/// GEGLU activation for REVE.
///
/// Python:
///   class GEGLU(nn.Module):
///       def forward(self, x):
///           x, gates = x.chunk(2, dim=-1)
///           return F.gelu(gates) * x

use burn::prelude::*;
use burn::tensor::activation::gelu;

/// GEGLU: split input in half along last dim, GELU-gate one half by the other.
/// Input: [*, 2*D] → Output: [*, D]
pub fn geglu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dims = x.dims();
    let last = dims.as_ref()[D - 1];
    let half = last / 2;
    let x_part = x.clone().narrow(D - 1, 0, half);
    let gates = x.narrow(D - 1, half, half);
    gelu(gates) * x_part
}
