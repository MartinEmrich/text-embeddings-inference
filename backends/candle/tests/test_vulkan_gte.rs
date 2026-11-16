mod common;

use anyhow::Result;
use common::{batch, load_tokenizer, sort_embeddings};
use std::path::PathBuf;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, ModelType, Pool};

fn model_root() -> PathBuf {
    std::env::var("TEI_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/martin/tei-vulkan/Snowflake-snowflake-arctic-embed-m-v2.0")
        })
}

#[test]
#[serial_test::serial]
#[cfg(feature = "vulkan")]
fn test_vulkan_gte_matches_cpu() -> Result<()> {
    let root = model_root();
    anyhow::ensure!(root.join("model.safetensors").exists(), "model not found at {root:?}");
    let tokenizer = load_tokenizer(&root)?;
    let sentences = [
        "What is Deep Learning?",
        "Deep Learning is a subset of machine learning.",
        "The quick brown fox jumps over the lazy dog.",
    ];
    let encodings: Vec<_> = sentences
        .iter()
        .map(|s| tokenizer.encode(*s, true).unwrap())
        .collect();
    let make_batch = || batch(encodings.clone(), vec![0, 1, 2], vec![]);

    // CPU reference
    std::env::set_var("TEI_FORCE_CPU", "1");
    let cpu = CandleBackend::new(
        &root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Cls),
        None,
    )?;
    std::env::remove_var("TEI_FORCE_CPU");
    let t0 = std::time::Instant::now();
    let (cpu_emb, _) = sort_embeddings(cpu.embed(make_batch())?);
    println!("cpu embed took {:?}", t0.elapsed());

    // Vulkan
    let vk = CandleBackend::new(
        &root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Cls),
        None,
    )?;
    // Warmup: pay RADV's first-use shader JIT + pipeline creation before timing
    // the steady-state forward.
    let _ = sort_embeddings(vk.embed(make_batch())?);
    let t0 = std::time::Instant::now();
    let (vk_emb, _) = sort_embeddings(vk.embed(make_batch())?);
    println!("vulkan embed took {:?}", t0.elapsed());

    for (i, (c, v)) in cpu_emb.iter().zip(vk_emb.iter()).enumerate() {
        let dot: f32 = c.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let nc: f32 = c.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nv: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        let cos = dot / (nc * nv);
        let max_abs = c
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("emb[{i}] cos={cos:.6} max_abs_diff={max_abs:.2e}");
    }

    // single-sentence (no padding) comparison
    for (i, s) in sentences.iter().enumerate() {
        let enc = tokenizer.encode(*s, true).unwrap();
        let single = || batch(vec![enc.clone()], vec![0], vec![]);
        std::env::set_var("TEI_FORCE_CPU", "1");
        let (c1, _) = sort_embeddings(cpu.embed(single())?);
        std::env::remove_var("TEI_FORCE_CPU");
        let (v1, _) = sort_embeddings(vk.embed(single())?);
        let c = &c1[0];
        let v = &v1[0];
        let dot: f32 = c.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let nc: f32 = c.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nv: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        let cos = dot / (nc * nv);
        println!("single[{i}] cos={cos:.6}");
        anyhow::ensure!(cos > 0.999, "single cosine too low for {i}: {cos}");
    }
    Ok(())
}

#[test]
#[serial_test::serial]
#[cfg(feature = "vulkan")]
fn test_vulkan_gte_f16_matches_cpu_f32() -> Result<()> {
    let root = model_root();
    anyhow::ensure!(root.join("model.safetensors").exists(), "model not found at {root:?}");
    let tokenizer = load_tokenizer(&root)?;
    let sentences = [
        "What is Deep Learning?",
        "Deep Learning is a subset of machine learning.",
        "The quick brown fox jumps over the lazy dog.",
    ];
    let encodings: Vec<_> = sentences
        .iter()
        .map(|s| tokenizer.encode(*s, true).unwrap())
        .collect();
    let make_batch = || batch(encodings.clone(), vec![0, 1, 2], vec![]);

    // CPU f32 reference
    std::env::set_var("TEI_FORCE_CPU", "1");
    let cpu = CandleBackend::new(
        &root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Cls),
        None,
    )?;
    std::env::remove_var("TEI_FORCE_CPU");
    let (cpu_emb, _) = sort_embeddings(cpu.embed(make_batch())?);

    // Vulkan f16 (f16 storage, f32 compute)
    let vk = CandleBackend::new(
        &root,
        "float16".to_string(),
        ModelType::Embedding(Pool::Cls),
        None,
    )?;
    let _ = sort_embeddings(vk.embed(make_batch())?); // warmup
    let t0 = std::time::Instant::now();
    let (vk_emb, _) = sort_embeddings(vk.embed(make_batch())?);
    println!("vulkan f16 embed took {:?}", t0.elapsed());

    for (i, (c, v)) in cpu_emb.iter().zip(vk_emb.iter()).enumerate() {
        let dot: f32 = c.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let nc: f32 = c.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nv: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        let cos = dot / (nc * nv);
        let max_abs = c
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("f16 emb[{i}] cos={cos:.6} max_abs_diff={max_abs:.2e}");
        anyhow::ensure!(cos > 0.999, "f16 batch cosine too low for {i}: {cos}");
    }

    // single-sentence (no padding) comparison
    for (i, s) in sentences.iter().enumerate() {
        let enc = tokenizer.encode(*s, true).unwrap();
        let single = || batch(vec![enc.clone()], vec![0], vec![]);
        std::env::set_var("TEI_FORCE_CPU", "1");
        let (c1, _) = sort_embeddings(cpu.embed(single())?);
        std::env::remove_var("TEI_FORCE_CPU");
        let (v1, _) = sort_embeddings(vk.embed(single())?);
        let c = &c1[0];
        let v = &v1[0];
        let dot: f32 = c.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let nc: f32 = c.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nv: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        let cos = dot / (nc * nv);
        println!("f16 single[{i}] cos={cos:.6}");
        anyhow::ensure!(cos > 0.999, "f16 single cosine too low for {i}: {cos}");
    }
    Ok(())
}

/// End-to-end memory + flash-attention guard: run a full GTE forward at the
/// router's default warmup size (two 8192-token sequences = max_batch_tokens
/// 16384). Before the deferred-batch byte budget this OOM'd; before the tiled FA
/// kernel the flash-attention launch tripped the amdgpu job timeout. Override
/// with GTE_TEST_SEQ / GTE_TEST_DTYPE for investigation.
#[test]
#[serial_test::serial]
#[cfg(feature = "vulkan")]
fn test_vulkan_gte_warmup_sized_batch() -> Result<()> {
    let root = model_root();
    anyhow::ensure!(root.join("model.safetensors").exists(), "model not found at {root:?}");

    let vk = CandleBackend::new(
        &root,
        std::env::var("GTE_TEST_DTYPE").unwrap_or_else(|_| "float16".to_string()),
        ModelType::Embedding(Pool::Cls),
        None,
    )?;

    let seq = std::env::var("GTE_TEST_SEQ").ok().and_then(|s| s.parse().ok()).unwrap_or(8192usize);
    let total = 2 * seq;
    let input_ids = vec![0u32; total];
    let token_type_ids = vec![0u32; total];
    let mut position_ids = Vec::with_capacity(total);
    for _ in 0..2 {
        position_ids.extend(0..seq as u32);
    }
    let b = Batch {
        input_ids,
        token_type_ids,
        position_ids,
        cumulative_seq_lengths: vec![0, seq as u32, total as u32],
        max_length: seq as u32,
        pooled_indices: vec![0, 1],
        raw_indices: vec![],
    };

    let t0 = std::time::Instant::now();
    let (emb, raw) = sort_embeddings(vk.embed(b)?);
    println!("warmup-sized batch ({} tokens) took {:?}", total, t0.elapsed());
    anyhow::ensure!(emb.len() == 2, "expected 2 pooled embeddings, got {}", emb.len());
    anyhow::ensure!(!emb[0].is_empty(), "empty embedding");
    anyhow::ensure!(raw.is_empty() || raw.first().map(|r| !r.is_empty()).unwrap_or(true));
    Ok(())
}
