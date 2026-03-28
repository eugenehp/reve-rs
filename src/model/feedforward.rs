/// Feed-Forward Network for REVE transformer blocks.
///
/// Python:
///   class FeedForward(nn.Module):
///       def __init__(self, dim, hidden_dim, geglu):
///           self.net = nn.Sequential(
///               RMSNorm(dim),
///               nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
///               GEGLU() if geglu else nn.GELU(),
///               nn.Linear(hidden_dim, dim, bias=False),
///           )

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use crate::model::rms_norm::RmsNorm;
use crate::model::geglu::geglu;

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub norm: RmsNorm<B>,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub use_geglu: bool,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, hidden_dim: usize, use_geglu: bool, device: &B::Device) -> Self {
        let in_features = if use_geglu { hidden_dim * 2 } else { hidden_dim };
        Self {
            norm: RmsNorm::new(dim, 1e-6, device),
            linear1: LinearConfig::new(dim, in_features).with_bias(false).init(device),
            linear2: LinearConfig::new(hidden_dim, dim).with_bias(false).init(device),
            use_geglu,
        }
    }

    /// x: [B, S, dim] → [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.norm.forward(x);
        let h = self.linear1.forward(h);
        let h = if self.use_geglu {
            geglu(h)
        } else {
            burn::tensor::activation::gelu(h)
        };
        self.linear2.forward(h)
    }
}
