/// Multi-Head Self-Attention with RMSNorm for REVE.
///
/// Python:
///   class Attention(nn.Module):
///       def __init__(self, dim, heads=8, head_dim=64):
///           inner_dim = head_dim * heads
///           self.norm = RMSNorm(dim)
///           self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
///           self.to_out = nn.Linear(inner_dim, dim, bias=False)
///           self.attend = ClassicalAttention(heads, use_sdpa=True)
///
///   class ClassicalAttention(nn.Module):
///       def forward(self, qkv):
///           q, k, v = qkv.chunk(3, dim=-1)
///           q,k,v = rearrange each 'b s (h d) -> b h s d'
///           out = scaled_dot_product_attention(q, k, v)
///           out = rearrange 'b h s d -> b s (h d)'

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use crate::model::rms_norm::RmsNorm;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub norm: RmsNorm<B>,
    pub to_qkv: Linear<B>,
    pub to_out: Linear<B>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, heads: usize, head_dim: usize, device: &B::Device) -> Self {
        let inner_dim = head_dim * heads;
        Self {
            norm: RmsNorm::new(dim, 1e-6, device),
            to_qkv: LinearConfig::new(dim, inner_dim * 3).with_bias(false).init(device),
            to_out: LinearConfig::new(inner_dim, dim).with_bias(false).init(device),
            n_heads: heads,
            head_dim,
        }
    }

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);

        let normed = self.norm.forward(x);
        let qkv = self.to_qkv.forward(normed); // [B, S, 3*inner_dim]

        // Split into Q, K, V: each [B, S, inner_dim]
        let inner = h * dh;
        let q = qkv.clone().narrow(2, 0, inner);
        let k = qkv.clone().narrow(2, inner, inner);
        let v = qkv.narrow(2, inner * 2, inner);

        // Reshape to [B, H, S, D]
        let q = q.reshape([b, s, h, dh]).swap_dims(1, 2);
        let k = k.reshape([b, s, h, dh]).swap_dims(1, 2);
        let v = v.reshape([b, s, h, dh]).swap_dims(1, 2);

        // Scaled dot-product attention
        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(q.matmul(k.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(v); // [B, H, S, D]

        // Merge heads: [B, H, S, D] → [B, S, H*D]
        let out = out.swap_dims(1, 2).reshape([b, s, h * dh]);
        self.to_out.forward(out)
    }
}
