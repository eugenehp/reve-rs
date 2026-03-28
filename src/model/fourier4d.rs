/// 4D Fourier Positional Embedding for REVE.
///
/// Encodes (x, y, z, t) coordinates using multi-frequency sinusoidal basis.
///
/// Python:
///   class FourierEmb4D(nn.Module):
///       def forward(self, positions_):
///           positions = positions_.clone()
///           positions[:, :, -1] *= self.increment_time
///           freqs_w = arange(self.freqs)
///           # Build 4D grid of frequencies
///           p_x = 2*pi*freqs_x / width; ...
///           loc = pos[...,0]*p_x + pos[...,1]*p_y + pos[...,2]*p_z + pos[...,3]*p_w
///           emb = cat([cos(loc), sin(loc)], dim=-1)
///
///   @classmethod
///   def add_time_patch(cls, pos, num_patches):
///       # pos: [B, C, 3] → [B, C*num_patches, 4]

use std::f32::consts::PI;
use burn::prelude::*;

/// Compute 4D Fourier positional embedding.
///
/// positions: [B, S, 4] where last dim is (x, y, z, t)
/// Returns: [B, S, dimension]
pub fn fourier_embed_4d<B: Backend>(
    positions: Tensor<B, 3>,
    dimension: usize,
    freqs: usize,
    increment_time: f32,
    margin: f32,
    device: &B::Device,
) -> Tensor<B, 3> {
    // Scale time dimension
    let pos = positions;
    let time_col = pos.clone().narrow(2, 3, 1).mul_scalar(increment_time);
    let xyz = pos.narrow(2, 0, 3);
    let pos = Tensor::cat(vec![xyz, time_col], 2); // [B, S, 4]

    // Add margin
    let pos = pos + margin;
    let width = 1.0 + 2.0 * margin;

    // Build frequency grid: for each combo of (fx, fy, fz, fw) where each in 0..freqs
    // loc = x*px + y*py + z*pz + w*pw where p_dim = 2*pi*freq/width
    //
    // The Python builds a 4D grid [freqs, freqs, freqs, freqs] then contracts with pos,
    // resulting in freqs^4 components per position. Then truncates to dimension/2.
    let half_dim = dimension / 2;
    let n_freq4 = freqs.pow(4);

    // Precompute frequency coefficients for all 4D combos
    let mut freq_coeffs = Vec::with_capacity(n_freq4 * 4);
    for fx in 0..freqs {
        for fy in 0..freqs {
            for fz in 0..freqs {
                for fw in 0..freqs {
                    freq_coeffs.push(2.0 * PI * fx as f32 / width);
                    freq_coeffs.push(2.0 * PI * fy as f32 / width);
                    freq_coeffs.push(2.0 * PI * fz as f32 / width);
                    freq_coeffs.push(2.0 * PI * fw as f32 / width);
                }
            }
        }
    }

    // freq_coeffs_t: [1, 4, n_freq4]  (transposed so pos @ freq = [B, S, n_freq4])
    let freq_t = Tensor::<B, 2>::from_data(
        TensorData::new(freq_coeffs, vec![n_freq4, 4]),
        device,
    )
    .transpose()
    .unsqueeze_dim::<3>(0); // [1, 4, n_freq4]

    // pos: [B, S, 4] @ freq_t: [1, 4, n_freq4] → [B, S, n_freq4]
    let loc = pos.matmul(freq_t);

    // Truncate to half_dim if needed
    let loc = if n_freq4 > half_dim {
        loc.narrow(2, 0, half_dim)
    } else if n_freq4 < half_dim {
        panic!(
            "freqs^4 = {} < half_dim = {}. Increase freqs parameter.",
            n_freq4, half_dim
        );
    } else {
        loc
    };

    // emb = cat([cos(loc), sin(loc)], dim=-1)  → [B, S, dimension]
    let cos_part = loc.clone().cos();
    let sin_part = loc.sin();
    Tensor::cat(vec![cos_part, sin_part], 2)
}

/// Expand position tensor by adding temporal patch index.
///
/// pos: [B, C, 3] → [B, C*num_patches, 4]
/// Each channel position is repeated for each time patch with t = 0, 1, ..., num_patches-1
pub fn add_time_patch<B: Backend>(
    pos: Tensor<B, 3>,
    num_patches: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [batch, n_chans, _] = pos.dims();

    // pos_repeated: [B, C, num_patches, 3]
    let pos_4d = pos.unsqueeze_dim::<4>(2); // [B, C, 1, 3]
    let pos_repeated = pos_4d.repeat_dim(2, num_patches); // [B, C, num_patches, 3]

    // time_values: [1, 1, num_patches, 1]
    let time_data: Vec<f32> = (0..num_patches).map(|t| t as f32).collect();
    let time_values = Tensor::<B, 1>::from_data(
        TensorData::new(time_data, vec![num_patches]),
        device,
    )
    .reshape([1, 1, num_patches, 1])
    .repeat_dim(0, batch)
    .repeat_dim(1, n_chans); // [B, C, num_patches, 1]

    // Concatenate: [B, C, num_patches, 4]
    let pos_with_time = Tensor::cat(vec![pos_repeated, time_values], 3);

    // Reshape: [B, C*num_patches, 4]
    pos_with_time.reshape([batch, n_chans * num_patches, 4])
}
