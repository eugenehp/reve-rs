/// Rust REVE benchmark binary.
///
/// Usage: benchmark <n_chans> <n_times> <warmup> <repeats>
/// Outputs JSON: { "times_ms": [...], "backend": "..." }

use burn::prelude::*;
use std::time::Instant;

// ── Backend selection ────────────────────────────────────────────────────────
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::Wgpu as B;
    pub fn device() -> burn::backend::wgpu::WgpuDevice {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }
    #[cfg(feature = "metal")]
    pub const NAME: &str = "wgpu-metal";
    #[cfg(feature = "vulkan")]
    pub const NAME: &str = "wgpu-vulkan";
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    pub const NAME: &str = "wgpu";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub fn device() -> burn::backend::ndarray::NdArrayDevice {
        burn::backend::ndarray::NdArrayDevice::Cpu
    }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "ndarray-accelerate";
    #[cfg(not(feature = "blas-accelerate"))]
    pub const NAME: &str = "ndarray";
}

use backend::{B, device, NAME};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: benchmark <n_chans> <n_times> <warmup> <repeats>");
        std::process::exit(1);
    }

    let n_chans: usize = args[1].parse().unwrap();
    let n_times: usize = args[2].parse().unwrap();
    let warmup: usize = args[3].parse().unwrap();
    let repeats: usize = args[4].parse().unwrap();

    let device = device();

    let model = reve_rs::model::reve::Reve::<B>::new(
        4,       // n_outputs
        n_chans,
        n_times,
        512,     // embed_dim
        2,       // depth
        8,       // heads
        64,      // head_dim
        2.66,
        true,    // geglu
        4,       // freqs
        200,     // patch_size
        20,      // patch_overlap
        true,    // attention pooling
        &device,
    );

    let eeg = Tensor::<B, 3>::ones([1, n_chans, n_times], &device).mul_scalar(0.1f32);
    let pos = Tensor::<B, 3>::ones([1, n_chans, 3], &device).mul_scalar(0.05f32);

    // Warmup
    for _ in 0..warmup {
        let _ = model.forward(eeg.clone(), pos.clone());
    }

    // Timed runs
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        let _ = model.forward(eeg.clone(), pos.clone());
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
    }

    let times_str: Vec<String> = times.iter().map(|t| format!("{:.4}", t)).collect();
    println!(
        "{{\"times_ms\": [{}], \"backend\": \"{}\"}}",
        times_str.join(", "),
        NAME
    );
}
