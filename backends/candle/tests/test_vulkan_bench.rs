mod common;

use anyhow::Result;
use std::path::PathBuf;
use std::time::Instant;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, ModelType, Pool};

// Throughput micro-benchmark for the Vulkan backend (ignored by default).
//
// Runs the full GTE forward on a varlen batch of BENCH_NSEQ sequences of
// BENCH_SEQLEN tokens each and prints steady-state tokens/s. Sweep BENCH_SEQLEN
// at fixed total tokens to separate the linear (matmul/elementwise) cost from
// the quadratic (attention) cost.
//
//   BENCH_MODEL   model dir   (default Snowflake arctic-embed-m-v2.0)
//   BENCH_DTYPE   float16 | float32          (default float16)
//   BENCH_NSEQ    sequences per batch         (default 64)
//   BENCH_SEQLEN  tokens per sequence         (default 128)
//   BENCH_ITERS   timed iterations            (default 8)
//
// cargo test -p text-embeddings-backend-candle --features vulkan \
//   --test test_vulkan_bench -- --ignored --nocapture --test-threads=1
#[test]
#[ignore]
#[serial_test::serial]
#[cfg(feature = "vulkan")]
fn bench_vulkan_forward() -> Result<()> {
    let root: PathBuf = std::env::var("BENCH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/martin/tei-vulkan/Snowflake-snowflake-arctic-embed-m-v2.0")
        });
    anyhow::ensure!(root.join("model.safetensors").exists(), "model not found at {root:?}");
    let dtype = std::env::var("BENCH_DTYPE").unwrap_or_else(|_| "float16".to_string());
    let nseq: usize = env_usize("BENCH_NSEQ", 64);
    let seqlen: usize = env_usize("BENCH_SEQLEN", 128);
    let iters: usize = env_usize("BENCH_ITERS", 8);
    let total = nseq * seqlen;

    let backend = CandleBackend::new(&root, dtype.clone(), ModelType::Embedding(Pool::Cls), None)?;

    let make = || {
        let input_ids = vec![0u32; total];
        let token_type_ids = vec![0u32; total];
        let mut position_ids = Vec::with_capacity(total);
        let mut cumulative_seq_lengths = Vec::with_capacity(nseq + 1);
        cumulative_seq_lengths.push(0u32);
        for i in 0..nseq {
            position_ids.extend(0..seqlen as u32);
            cumulative_seq_lengths.push(((i + 1) * seqlen) as u32);
        }
        Batch {
            input_ids,
            token_type_ids,
            position_ids,
            cumulative_seq_lengths,
            max_length: seqlen as u32,
            pooled_indices: (0..nseq as u32).collect(),
            raw_indices: vec![],
        }
    };

    // Warmup: shader JIT + pipeline creation + allocator warm-up.
    for _ in 0..2 {
        backend.embed(make())?;
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        backend.embed(make())?;
    }
    let secs = t0.elapsed().as_secs_f64();
    let tps = (total * iters) as f64 / secs;
    println!(
        "BENCH dtype={dtype} nseq={nseq} seqlen={seqlen} total={total} \
         iters={iters} time={secs:.3}s  => {tps:.0} tok/s"
    );
    Ok(())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}
