/// Standalone REVE encoder — produce EEG embeddings or classification outputs.

use std::{path::Path, time::Instant};

use anyhow::Context;
use burn::prelude::*;

use crate::{
    config::ModelConfig,
    data::{InputBatch, channel_wise_normalize},
    model::reve::Reve,
    weights::load_model,
};

/// Per-sample output from REVE.
pub struct ReveOutput {
    /// Output values (row-major f32).
    /// Classification mode: [n_outputs]
    pub output: Vec<f32>,
    /// Shape of the output.
    pub shape: Vec<usize>,
    pub n_channels: usize,
}

/// Collection of outputs.
pub struct EncodingResult {
    pub outputs: Vec<ReveOutput>,
    pub ms_load: f64,
    pub ms_encode: f64,
}

/// REVE encoder for EEG signal processing.
pub struct ReveEncoder<B: Backend> {
    model: Reve<B>,
    pub model_cfg: ModelConfig,
    device: B::Device,
}

impl<B: Backend> ReveEncoder<B> {
    /// Load model from config.json and weights safetensors.
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        device: B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("config: {}", config_path.display()))?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: ModelConfig = serde_json::from_value(
            hf_val.get("model").cloned().unwrap_or(hf_val.clone()),
        )
        .context("parsing model config")?;

        let t = Instant::now();
        let model = load_model::<B>(
            &model_cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            &device,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((
            Self {
                model,
                model_cfg,
                device,
            },
            ms,
        ))
    }

    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        format!(
            "REVE  embed_dim={}  depth={}  heads={}  head_dim={}  patch={}  outputs={}",
            c.embed_dim, c.depth, c.heads, c.head_dim, c.patch_size, c.n_outputs,
        )
    }

    /// Run inference on a prepared InputBatch.
    pub fn run_batch(&self, batch: &InputBatch<B>) -> anyhow::Result<ReveOutput> {
        let signal = channel_wise_normalize(batch.signal.clone());

        let output = self.model.forward(signal, batch.positions.clone());
        // [B, n_outputs] or [B, n_outputs]

        let shape = output.dims().to_vec();
        let output_vec = output
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("output→vec: {e:?}"))?;

        Ok(ReveOutput {
            output: output_vec,
            shape: shape[1..].to_vec(), // remove batch dim
            n_channels: batch.n_channels,
        })
    }

    /// Run on multiple batches.
    pub fn run_batches(
        &self,
        batches: &[InputBatch<B>],
    ) -> anyhow::Result<Vec<ReveOutput>> {
        batches.iter().map(|b| self.run_batch(b)).collect()
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}
