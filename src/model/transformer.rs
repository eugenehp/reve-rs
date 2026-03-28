/// Transformer Backbone for REVE.
///
/// Python:
///   class TransformerBackbone(nn.Module):
///       def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):
///           self.layers = nn.ModuleList([
///               nn.ModuleList([Attention(dim, heads, head_dim), FeedForward(dim, mlp_dim, geglu)])
///               for _ in range(depth)
///           ])
///       def forward(self, x, return_out_layers=False):
///           for attn, ff in self.layers:
///               x = attn(x) + x
///               x = ff(x) + x
///           return x

use burn::prelude::*;
use crate::model::attention::Attention;
use crate::model::feedforward::FeedForward;

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attn: Attention<B>,
    pub ff: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        dim: usize,
        heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        use_geglu: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            attn: Attention::new(dim, heads, head_dim, device),
            ff: FeedForward::new(dim, mlp_dim, use_geglu, device),
        }
    }

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.attn.forward(x.clone()) + x;
        let x_res = x.clone();
        self.ff.forward(x) + x_res
    }
}

#[derive(Module, Debug)]
pub struct TransformerBackbone<B: Backend> {
    pub layers: Vec<TransformerBlock<B>>,
}

impl<B: Backend> TransformerBackbone<B> {
    pub fn new(
        dim: usize,
        depth: usize,
        heads: usize,
        head_dim: usize,
        mlp_dim: usize,
        use_geglu: bool,
        device: &B::Device,
    ) -> Self {
        let layers = (0..depth)
            .map(|_| TransformerBlock::new(dim, heads, head_dim, mlp_dim, use_geglu, device))
            .collect();
        Self { layers }
    }

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for block in &self.layers {
            x = block.forward(x);
        }
        x
    }

    /// Forward returning outputs from all layers (for secondary loss / attention pooling).
    pub fn forward_with_layers(&self, x: Tensor<B, 3>) -> Vec<Tensor<B, 3>> {
        let mut out = vec![x.clone()];
        let mut x = x;
        for block in &self.layers {
            x = block.forward(x);
            out.push(x.clone());
        }
        out
    }
}
