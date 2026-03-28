/// REVE (Representation for EEG with Versatile Embeddings) — full model.
///
/// Architecture:
///   1. Patch extraction via unfold (overlapping temporal patches)
///   2. Linear patch embedding
///   3. 4D positional encoding (Fourier + MLP) for (x,y,z,t)
///   4. Transformer encoder (RMSNorm, GEGLU, multi-head attention)
///   5. Classification head (flatten + LayerNorm + Linear, or attention pooling)

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig};
use burn::tensor::activation::gelu;

use crate::model::transformer::TransformerBackbone;
use crate::model::fourier4d;

// ── REVE Model ────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Reve<B: Backend> {
    // Patch embedding: Linear(patch_size → embed_dim)
    pub patch_embed: Linear<B>,
    // 4D positional encoding MLP: Linear(4 → embed_dim) → GELU → LayerNorm
    pub mlp4d_linear: Linear<B>,
    pub mlp4d_ln: LayerNorm<B>,
    // 4DPE output LayerNorm
    pub pos_ln: LayerNorm<B>,
    // Transformer encoder
    pub transformer: TransformerBackbone<B>,
    // Classification head
    pub final_ln: LayerNorm<B>,
    pub final_linear: Linear<B>,
    // Attention pooling (optional)
    pub cls_query_token: Option<Param<Tensor<B, 3>>>,

    // Config
    pub embed_dim: usize,
    pub n_chans: usize,
    pub n_times: usize,
    pub n_outputs: usize,
    pub patch_size: usize,
    pub patch_overlap: usize,
    pub freqs: usize,
    pub use_attention_pooling: bool,
}

impl<B: Backend> Reve<B> {
    pub fn new(
        n_outputs: usize,
        n_chans: usize,
        n_times: usize,
        embed_dim: usize,
        depth: usize,
        heads: usize,
        head_dim: usize,
        mlp_dim_ratio: f64,
        use_geglu: bool,
        freqs: usize,
        patch_size: usize,
        patch_overlap: usize,
        attention_pooling: bool,
        device: &B::Device,
    ) -> Self {
        let mlp_dim = (embed_dim as f64 * mlp_dim_ratio) as usize;

        let patch_embed = LinearConfig::new(patch_size, embed_dim)
            .with_bias(true)
            .init(device);

        let mlp4d_linear = LinearConfig::new(4, embed_dim)
            .with_bias(false)
            .init(device);
        let mlp4d_ln = LayerNormConfig::new(embed_dim).init(device);
        let pos_ln = LayerNormConfig::new(embed_dim).init(device);

        let transformer = TransformerBackbone::new(
            embed_dim, depth, heads, head_dim, mlp_dim, use_geglu, device,
        );

        let n_patches = compute_n_patches(n_times, patch_size, patch_overlap);
        let final_dim = if attention_pooling {
            embed_dim
        } else {
            n_chans * n_patches * embed_dim
        };

        let final_ln = LayerNormConfig::new(final_dim).init(device);
        let final_linear = LinearConfig::new(final_dim, n_outputs)
            .with_bias(true)
            .init(device);

        let cls_query_token = if attention_pooling {
            Some(Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, 1, embed_dim], device),
            ))
        } else {
            None
        };

        Self {
            patch_embed,
            mlp4d_linear,
            mlp4d_ln,
            pos_ln,
            transformer,
            final_ln,
            final_linear,
            cls_query_token,
            embed_dim,
            n_chans,
            n_times,
            n_outputs,
            patch_size,
            patch_overlap,
            freqs,
            use_attention_pooling: attention_pooling,
        }
    }

    /// Forward pass.
    ///
    /// eeg: [B, C, T]
    /// pos: [B, C, 3] — (x, y, z) electrode positions
    /// Returns: [B, n_outputs]
    pub fn forward(
        &self,
        eeg: Tensor<B, 3>,
        pos: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let [batch, n_chans, n_times] = eeg.dims();
        let device = eeg.device();
        let step = self.patch_size - self.patch_overlap;

        // 1. Extract overlapping patches — matches torch.unfold(dim=2, size, step)
        // torch.unfold returns floor((size - kernel) / step) + 1 patches
        let n_patches = (n_times - self.patch_size) / step + 1;
        let patches = self.extract_patches(eeg, n_chans, n_patches, &device);
        // patches: [B, C, n_patches, patch_size]

        // 2. Build 4D positional encoding
        let pos_4d = fourier4d::add_time_patch(pos, n_patches, &device);
        // pos_4d: [B, C*n_patches, 4]

        let fourier_emb = fourier4d::fourier_embed_4d(
            pos_4d.clone(), self.embed_dim, self.freqs, 0.1, 0.4, &device,
        );
        let mlp_emb = self.mlp4d_ln.forward(
            gelu(self.mlp4d_linear.forward(pos_4d))
        );
        let pos_embed = self.pos_ln.forward(fourier_emb + mlp_emb);
        // pos_embed: [B, C*n_patches, embed_dim]

        // 3. Patch embedding
        // patches: [B, C, n_patches, patch_size] → [B, C*n_patches, patch_size]
        let patch_flat = patches.reshape([batch, n_chans * n_patches, self.patch_size]);
        let patch_embedded = self.patch_embed.forward(patch_flat);
        // [B, C*n_patches, embed_dim]

        // 4. Add positional encoding
        let x = patch_embedded + pos_embed;

        // 5. Transformer
        let x = self.transformer.forward(x);
        // [B, C*n_patches, embed_dim]

        // 6. Reshape back to [B, C, n_patches, embed_dim]
        let x = x.reshape([batch, n_chans, n_patches, self.embed_dim]);

        // 7. Classification head
        if self.use_attention_pooling {
            let x = self.attention_pooling(x);
            let x = self.final_ln.forward(x);
            self.final_linear.forward(x)
        } else {
            let x = x.reshape([batch, n_chans * n_patches * self.embed_dim]);
            let x = self.final_ln.forward(x);
            self.final_linear.forward(x)
        }
    }

    /// Extract overlapping patches from the signal, matching `torch.unfold`.
    ///
    /// eeg: [B, C, T] → [B, C, n_patches, patch_size]
    /// torch.unfold drops any remainder that doesn't fit a full patch.
    fn extract_patches(
        &self,
        eeg: Tensor<B, 3>,
        _n_chans: usize,
        n_patches: usize,
        _device: &B::Device,
    ) -> Tensor<B, 4> {
        let step = self.patch_size - self.patch_overlap;

        let mut patch_list = Vec::with_capacity(n_patches);
        for p in 0..n_patches {
            let start = p * step;
            let patch = eeg.clone().narrow(2, start, self.patch_size);
            patch_list.push(patch.unsqueeze_dim::<4>(2)); // [B, C, 1, patch_size]
        }
        Tensor::cat(patch_list, 2) // [B, C, n_patches, patch_size]
    }

    /// Attention pooling: use cls_query_token to attend to all encoder outputs.
    ///
    /// x: [B, C, S, E] → [B, E]
    fn attention_pooling(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, n_chans, seq_len, embed_dim] = x.dims();
        // Flatten channel and sequence: [B, C*S, E]
        let x_flat = x.reshape([batch, n_chans * seq_len, embed_dim]);

        let query = self.cls_query_token.as_ref().unwrap().val()
            .expand([batch, 1, embed_dim]); // [B, 1, E]

        // Attention scores: [B, 1, E] @ [B, E, C*S] = [B, 1, C*S]
        let scale = (embed_dim as f64).powf(-0.5) as f32;
        let scores = query.matmul(x_flat.clone().transpose()).mul_scalar(scale);
        let weights = burn::tensor::activation::softmax(scores, 2); // [B, 1, C*S]

        // Weighted sum: [B, 1, C*S] @ [B, C*S, E] = [B, 1, E]
        let out = weights.matmul(x_flat);
        out.reshape([batch, embed_dim])
    }
}

/// Compute number of patches matching `torch.unfold` semantics:
/// `floor((n_times - patch_size) / step) + 1`
fn compute_n_patches(n_times: usize, patch_size: usize, patch_overlap: usize) -> usize {
    let step = patch_size - patch_overlap;
    (n_times - patch_size) / step + 1
}
