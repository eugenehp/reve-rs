/// Load pretrained REVE weights from a safetensors file.
///
/// Weight key patterns (from Python state_dict):
///
///   to_patch_embedding.0.weight                   [embed_dim, patch_size]
///   to_patch_embedding.0.bias                     [embed_dim]
///   mlp4d.0.weight                                [embed_dim, 4]
///   mlp4d.2.weight                                [embed_dim]
///   mlp4d.2.bias                                  [embed_dim]
///   ln.weight                                     [embed_dim]
///   ln.bias                                       [embed_dim]
///   transformer.layers.{i}.0.norm.weight          [embed_dim]          (RMSNorm)
///   transformer.layers.{i}.0.to_qkv.weight        [3*inner, embed_dim] (no bias)
///   transformer.layers.{i}.0.to_out.weight         [embed_dim, inner]   (no bias)
///   transformer.layers.{i}.1.net.0.weight          [embed_dim]          (RMSNorm)
///   transformer.layers.{i}.1.net.1.weight          [geglu_dim, embed_dim] (no bias)
///   transformer.layers.{i}.1.net.3.weight          [embed_dim, mlp_dim]   (no bias)
///   final_layer.1.weight                           [final_dim]           (LayerNorm)
///   final_layer.1.bias                             [final_dim]
///   final_layer.2.weight                           [n_outputs, final_dim]
///   final_layer.2.bias                             [n_outputs]
///
///   For attention_pooling:
///   cls_query_token                                [1, 1, embed_dim]
///   final_layer.0.weight                           [embed_dim]
///   final_layer.0.bias                             [embed_dim]
///   final_layer.1.weight                           [n_outputs, embed_dim]
///   final_layer.1.bias                             [n_outputs]

use std::collections::HashMap;
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;

use crate::model::reve::Reve;
use crate::config::ModelConfig;

// ── WeightMap ─────────────────────────────────────────────────────────────────

pub struct WeightMap {
    pub tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::with_capacity(st.len());

        for (raw_key, view) in st.tensors() {
            let key = raw_key
                .strip_prefix("model.")
                .unwrap_or(raw_key.as_str())
                .to_string();

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;
        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }
        Ok(Tensor::<B, N>::from_data(TensorData::new(data, shape), device))
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:80}  {s:?}");
        }
    }
}

// ── Weight assignment helpers ─────────────────────────────────────────────────

/// PyTorch [out, in] → burn [in, out]
fn set_linear_w<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
}

fn set_linear_wb<B: Backend>(linear: &mut burn::nn::Linear<B>, w: Tensor<B, 2>, b: Tensor<B, 1>) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = linear.bias {
        linear.bias = Some(bias.clone().map(|_| b));
    }
}

fn set_layernorm<B: Backend>(norm: &mut burn::nn::LayerNorm<B>, w: Tensor<B, 1>, b: Tensor<B, 1>) {
    norm.gamma = norm.gamma.clone().map(|_| w);
    if let Some(ref beta) = norm.beta {
        norm.beta = Some(beta.clone().map(|_| b));
    }
}

fn set_rmsnorm<B: Backend>(norm: &mut crate::model::rms_norm::RmsNorm<B>, w: Tensor<B, 1>) {
    norm.weight = norm.weight.clone().map(|_| w);
}

// ── Full model loader ─────────────────────────────────────────────────────────

/// Load a REVE model from a safetensors file.
pub fn load_model<B: Backend>(
    cfg: &ModelConfig,
    weights_path: &str,
    device: &B::Device,
) -> anyhow::Result<Reve<B>> {
    let mut wm = WeightMap::from_file(weights_path)?;
    eprintln!("Loading {} weight tensors...", wm.tensors.len());
    load_model_from_wm(cfg, &mut wm, device)
}

pub fn load_model_from_wm<B: Backend>(
    cfg: &ModelConfig,
    wm: &mut WeightMap,
    device: &B::Device,
) -> anyhow::Result<Reve<B>> {
    let mut model = Reve::new(
        cfg.n_outputs,
        cfg.n_chans,
        cfg.n_times,
        cfg.embed_dim,
        cfg.depth,
        cfg.heads,
        cfg.head_dim,
        cfg.mlp_dim_ratio,
        cfg.use_geglu,
        cfg.freqs,
        cfg.patch_size,
        cfg.patch_overlap,
        cfg.attention_pooling,
        device,
    );

    load_reve_weights(wm, &mut model, cfg, device)?;
    Ok(model)
}

fn load_reve_weights<B: Backend>(
    wm: &mut WeightMap,
    model: &mut Reve<B>,
    cfg: &ModelConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    // ── Patch embedding ─────────────────────────────────────────────────────
    // to_patch_embedding.0.weight [embed_dim, patch_size]
    // to_patch_embedding.0.bias   [embed_dim]
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 2>("to_patch_embedding.0.weight", device),
        wm.take::<B, 1>("to_patch_embedding.0.bias", device),
    ) {
        set_linear_wb(&mut model.patch_embed, w, b);
    }

    // ── MLP4D (positional encoding MLP) ─────────────────────────────────────
    // mlp4d.0.weight [embed_dim, 4] (no bias)
    if let Ok(w) = wm.take::<B, 2>("mlp4d.0.weight", device) {
        set_linear_w(&mut model.mlp4d_linear, w);
    }
    // mlp4d.2.weight [embed_dim], mlp4d.2.bias [embed_dim]  (LayerNorm)
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 1>("mlp4d.2.weight", device),
        wm.take::<B, 1>("mlp4d.2.bias", device),
    ) {
        set_layernorm(&mut model.mlp4d_ln, w, b);
    }

    // ── 4DPE output LayerNorm ───────────────────────────────────────────────
    // ln.weight [embed_dim], ln.bias [embed_dim]
    if let (Ok(w), Ok(b)) = (
        wm.take::<B, 1>("ln.weight", device),
        wm.take::<B, 1>("ln.bias", device),
    ) {
        set_layernorm(&mut model.pos_ln, w, b);
    }

    // ── Transformer layers ──────────────────────────────────────────────────
    for i in 0..cfg.depth {
        let block = &mut model.transformer.layers[i];

        // Attention: layers.{i}.0
        // layers.{i}.0.norm.weight  (RMSNorm, no bias)
        if let Ok(w) = wm.take::<B, 1>(
            &format!("transformer.layers.{i}.0.norm.weight"), device,
        ) {
            set_rmsnorm(&mut block.attn.norm, w);
        }
        // layers.{i}.0.to_qkv.weight [3*inner_dim, embed_dim] (no bias)
        if let Ok(w) = wm.take::<B, 2>(
            &format!("transformer.layers.{i}.0.to_qkv.weight"), device,
        ) {
            set_linear_w(&mut block.attn.to_qkv, w);
        }
        // layers.{i}.0.to_out.weight [embed_dim, inner_dim] (no bias)
        if let Ok(w) = wm.take::<B, 2>(
            &format!("transformer.layers.{i}.0.to_out.weight"), device,
        ) {
            set_linear_w(&mut block.attn.to_out, w);
        }

        // FeedForward: layers.{i}.1
        // layers.{i}.1.net.0.weight  (RMSNorm, no bias)
        if let Ok(w) = wm.take::<B, 1>(
            &format!("transformer.layers.{i}.1.net.0.weight"), device,
        ) {
            set_rmsnorm(&mut block.ff.norm, w);
        }
        // layers.{i}.1.net.1.weight [geglu_dim, embed_dim] (no bias)
        if let Ok(w) = wm.take::<B, 2>(
            &format!("transformer.layers.{i}.1.net.1.weight"), device,
        ) {
            set_linear_w(&mut block.ff.linear1, w);
        }
        // layers.{i}.1.net.3.weight [embed_dim, mlp_dim] (no bias)
        if let Ok(w) = wm.take::<B, 2>(
            &format!("transformer.layers.{i}.1.net.3.weight"), device,
        ) {
            set_linear_w(&mut block.ff.linear2, w);
        }
    }

    // ── Classification head ─────────────────────────────────────────────────
    if cfg.attention_pooling {
        // cls_query_token [1, 1, embed_dim]
        if let Ok(t) = wm.take::<B, 3>("cls_query_token", device) {
            if let Some(ref mut q) = model.cls_query_token {
                *q = q.clone().map(|_| t);
            }
        }
        // final_layer.0 = LayerNorm, final_layer.1 = Linear
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>("final_layer.0.weight", device),
            wm.take::<B, 1>("final_layer.0.bias", device),
        ) {
            set_layernorm(&mut model.final_ln, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>("final_layer.1.weight", device),
            wm.take::<B, 1>("final_layer.1.bias", device),
        ) {
            set_linear_wb(&mut model.final_linear, w, b);
        }
    } else {
        // final_layer: Flatten, LayerNorm, Linear
        // final_layer.1 = LayerNorm, final_layer.2 = Linear
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 1>("final_layer.1.weight", device),
            wm.take::<B, 1>("final_layer.1.bias", device),
        ) {
            set_layernorm(&mut model.final_ln, w, b);
        }
        if let (Ok(w), Ok(b)) = (
            wm.take::<B, 2>("final_layer.2.weight", device),
            wm.take::<B, 1>("final_layer.2.bias", device),
        ) {
            set_linear_wb(&mut model.final_linear, w, b);
        }
    }

    Ok(())
}
