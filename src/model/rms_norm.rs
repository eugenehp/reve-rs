/// RMSNorm for REVE (used instead of LayerNorm in the Transformer).
///
/// Python:
///   class RMSNorm(nn.Module):
///       def _norm(self, x): return x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
///       def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight

use burn::prelude::*;
use burn::module::{Param, ParamId};

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
    pub dim: usize,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::initialized(ParamId::new(), Tensor::ones([dim], device)),
            eps,
            dim,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // x.pow(2).mean(-1, keepdim=True) → rsqrt → * x → * weight
        let x_sq = x.clone().powf_scalar(2.0);
        let mean_sq = x_sq.mean_dim(D - 1); // keepdim
        let rsqrt = (mean_sq + self.eps).sqrt().recip();
        let normed = x * rsqrt;
        let w = self.weight.val();
        normed * w.unsqueeze()
    }
}
