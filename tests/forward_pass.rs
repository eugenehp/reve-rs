/// Basic forward pass test — verify the model runs without panics
/// and produces the correct output shape.

use burn::backend::NdArray as B;
use burn::prelude::*;

use reve_rs::model::reve::Reve;

#[test]
fn test_forward_pass_basic() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let n_outputs = 4;
    let n_chans = 8;
    let n_times = 1000;
    let embed_dim = 64; // small for testing
    let depth = 2;
    let heads = 4;
    let head_dim = 16;

    let model = Reve::<B>::new(
        n_outputs,
        n_chans,
        n_times,
        embed_dim,
        depth,
        heads,
        head_dim,
        2.66,
        true,  // geglu
        4,     // freqs
        200,   // patch_size
        20,    // patch_overlap
        false, // no attention pooling
        &device,
    );

    // Random EEG signal: [1, 8, 1000]
    let eeg = Tensor::<B, 3>::ones([1, n_chans, n_times], &device).mul_scalar(0.1);

    // Random positions: [1, 8, 3]
    let pos = Tensor::<B, 3>::ones([1, n_chans, 3], &device).mul_scalar(0.05);

    let output = model.forward(eeg, pos);
    let dims = output.dims();

    assert_eq!(dims[0], 1, "batch dimension");
    assert_eq!(dims[1], n_outputs, "output dimension");
    eprintln!("Output shape: {:?}", dims);
}

#[test]
fn test_forward_pass_attention_pooling() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let n_outputs = 4;
    let n_chans = 8;
    let n_times = 1000;
    let embed_dim = 64;
    let depth = 2;
    let heads = 4;
    let head_dim = 16;

    let model = Reve::<B>::new(
        n_outputs,
        n_chans,
        n_times,
        embed_dim,
        depth,
        heads,
        head_dim,
        2.66,
        true,
        4,
        200,
        20,
        true, // attention pooling
        &device,
    );

    let eeg = Tensor::<B, 3>::ones([1, n_chans, n_times], &device).mul_scalar(0.1);
    let pos = Tensor::<B, 3>::ones([1, n_chans, 3], &device).mul_scalar(0.05);

    let output = model.forward(eeg, pos);
    let dims = output.dims();

    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], n_outputs);
    eprintln!("Attention pooling output shape: {:?}", dims);
}

#[test]
fn test_channel_invariance() {
    // REVE should handle different numbers of channels
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    for n_chans in [4, 16, 32] {
        let n_times = 1000;
        let embed_dim = 64;

        let model = Reve::<B>::new(
            4, n_chans, n_times, embed_dim,
            2, 4, 16, 2.66, true, 4, 200, 20, true, // use attention pooling for variable channels
            &device,
        );

        let eeg = Tensor::<B, 3>::ones([1, n_chans, n_times], &device).mul_scalar(0.1);
        let pos = Tensor::<B, 3>::ones([1, n_chans, 3], &device).mul_scalar(0.05);

        let output = model.forward(eeg, pos);
        assert_eq!(output.dims(), [1, 4]);
        eprintln!("n_chans={n_chans}: output shape {:?}", output.dims());
    }
}
