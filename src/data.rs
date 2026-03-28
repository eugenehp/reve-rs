/// Data preparation for REVE inference.
///
/// REVE input: (B, C, T) signal + (B, C, 3) channel positions.

use burn::prelude::*;

/// A single prepared input for the REVE model.
pub struct InputBatch<B: Backend> {
    /// EEG signal: [1, C, T].
    pub signal: Tensor<B, 3>,
    /// Channel 3D positions: [1, C, 3].
    pub positions: Tensor<B, 3>,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of time samples.
    pub n_samples: usize,
}

/// Channel-wise z-score normalisation (matching REVE preprocessing).
///
/// REVE expects z-scored input clipped at 15 standard deviations.
pub fn channel_wise_normalize<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2); // [B, C, 1]
    let diff = x.clone() - mean.clone();
    let var = (diff.clone() * diff).mean_dim(2);
    let std = (var + 1e-8).sqrt();
    let normed = (x - mean) / std;
    // Clip at ±15 std
    normed.clamp(-15.0, 15.0)
}

/// Build InputBatch from raw arrays.
pub fn build_batch<B: Backend>(
    signal: Vec<f32>,      // [C, T] row-major
    positions: Vec<f32>,   // [C, 3] row-major
    n_channels: usize,
    n_samples: usize,
    device: &B::Device,
) -> InputBatch<B> {
    let signal = Tensor::<B, 2>::from_data(
        TensorData::new(signal, vec![n_channels, n_samples]),
        device,
    )
    .unsqueeze_dim::<3>(0); // [1, C, T]

    let positions = Tensor::<B, 2>::from_data(
        TensorData::new(positions, vec![n_channels, 3]),
        device,
    )
    .unsqueeze_dim::<3>(0); // [1, C, 3]

    InputBatch {
        signal,
        positions,
        n_channels,
        n_samples,
    }
}

/// Build InputBatch from channel names using a position bank.
pub fn build_batch_named<B: Backend>(
    signal: Vec<f32>,
    channel_names: &[&str],
    n_samples: usize,
    position_bank: &crate::position_bank::PositionBank,
    device: &B::Device,
) -> InputBatch<B> {
    let n_channels = channel_names.len();
    let positions = position_bank.get_positions(channel_names);
    build_batch(signal, positions, n_channels, n_samples, device)
}
