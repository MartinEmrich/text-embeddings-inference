mod common;

use anyhow::Result;
use common::{batch, load_tokenizer, sort_embeddings};
use std::path::PathBuf;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

fn model_root() -> PathBuf {
    std::env::var("TEI_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/martin/tei-vulkan/Qwen--Qwen3-Embedding-0.6B"))
}

#[test]
#[serial_test::serial]
#[cfg(feature = "vulkan")]
fn test_vulkan_qwen3_matches_cpu() -> Result<()> {
    let root = model_root();
    anyhow::ensure!(
        root.join("model.safetensors").exists(),
        "model not found at {root:?}"
    );
    let tokenizer = load_tokenizer(&root)?;
    let sentences = [
        "What is Deep Learning?",
        "Deep Learning is a subset of machine learning that uses neural networks.",
        "The quick brown fox jumps over the lazy dog.",
    ];
    let encodings: Vec<_> = sentences
        .iter()
        .map(|s| tokenizer.encode(*s, true).unwrap())
        .collect();
    let make_batch = || batch(encodings.clone(), vec![0, 1, 2], vec![]);

    // CPU reference (non-flash Qwen3Model, padded).
    std::env::set_var("TEI_FORCE_CPU", "1");
    let cpu = CandleBackend::new(
        &root,
        "float32".to_string(),
        ModelType::Embedding(Pool::LastToken),
        None,
    )?;
    std::env::remove_var("TEI_FORCE_CPU");
    let t0 = std::time::Instant::now();
    let (cpu_emb, _) = sort_embeddings(cpu.embed(make_batch())?);
    println!("cpu embed took {:?}", t0.elapsed());

    // Vulkan (FlashQwen3Model, varlen) — flash is default-on for Vulkan.
    let vk = CandleBackend::new(
        &root,
        "float32".to_string(),
        ModelType::Embedding(Pool::LastToken),
        None,
    )?;
    let _ = sort_embeddings(vk.embed(make_batch())?); // warmup
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
        anyhow::ensure!(cos > 0.999, "batch cosine too low for {i}: {cos}");
    }

    // single-sentence comparison.
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
fn test_vulkan_qwen3_f16_matches_cpu_f32() -> Result<()> {
    let root = model_root();
    anyhow::ensure!(
        root.join("model.safetensors").exists(),
        "model not found at {root:?}"
    );
    let tokenizer = load_tokenizer(&root)?;
    let sentences = [
        "What is Deep Learning?",
        "Deep Learning is a subset of machine learning that uses neural networks.",
        "The quick brown fox jumps over the lazy dog.",
    ];
    let encodings: Vec<_> = sentences
        .iter()
        .map(|s| tokenizer.encode(*s, true).unwrap())
        .collect();
    let make_batch = || batch(encodings.clone(), vec![0, 1, 2], vec![]);

    // CPU f32 reference.
    std::env::set_var("TEI_FORCE_CPU", "1");
    let cpu = CandleBackend::new(
        &root,
        "float32".to_string(),
        ModelType::Embedding(Pool::LastToken),
        None,
    )?;
    std::env::remove_var("TEI_FORCE_CPU");
    let (cpu_emb, _) = sort_embeddings(cpu.embed(make_batch())?);

    // Vulkan f16 (f16 storage, f32 compute; GQA varlen flash).
    let vk = CandleBackend::new(
        &root,
        "float16".to_string(),
        ModelType::Embedding(Pool::LastToken),
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

    // single-sentence comparison.
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
