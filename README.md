# reve-rs

Pure-Rust inference for the **REVE** (Representation for EEG with Versatile Embeddings) foundation model, built on [Burn 0.20](https://burn.dev).

REVE is pretrained on **60,000+ hours** of EEG data from **92 datasets** spanning **25,000 subjects**. Its key innovation is a **4D Fourier positional encoding** scheme that enables generalization across arbitrary electrode configurations without retraining.

## Architecture

```
EEG [B, C, T]
    │
    ├─ Overlapping Patch Extraction (unfold)
    │  → [B, C, n_patches, patch_size]
    │
    ├─ Linear Patch Embedding
    │  → [B, C*n_patches, embed_dim]
    │
    ├─ 4D Positional Encoding (Fourier + MLP)
    │  (x, y, z, t) → [B, C*n_patches, embed_dim]
    │
    ├─ Transformer Encoder (RMSNorm, GEGLU, Multi-Head Attention)
    │  → [B, C*n_patches, embed_dim]
    │
    └─ Classification Head (Flatten+LN+Linear or Attention Pooling)
       → [B, n_outputs]
```

## Quick Start

```rust
use reve_rs::{ReveEncoder, data};
use std::path::Path;

// Load model
let (model, ms) = ReveEncoder::<B>::load(
    Path::new("config.json"),
    Path::new("model.safetensors"),
    device,
)?;

// Build input batch
let batch = data::build_batch::<B>(
    signal,     // [C, T] flat f32
    positions,  // [C, 3] flat f32
    n_channels,
    n_samples,
    &device,
);

// Run inference
let output = model.run_batch(&batch)?;
println!("Output: {:?}", output.shape);
```

## Build

```bash
# CPU (default — NdArray + Rayon)
cargo build --release

# CPU with Apple Accelerate BLAS (macOS)
cargo build --release --features blas-accelerate

# GPU (Metal on macOS)
cargo build --release --no-default-features --features metal

# GPU (Vulkan on Linux/Windows)
cargo build --release --no-default-features --features vulkan
```

## CLI Inference

```bash
# Download weights (requires HuggingFace access)
cargo run --release --features hf-download --bin download_weights -- --repo brain-bzh/reve-base

# Run inference
cargo run --release --bin infer -- --weights data/model.safetensors --config data/config.json -v
```

## Pretrained Weights

Weights are on [HuggingFace](https://huggingface.co/collections/brain-bzh/reve):

| Model | Params | Embed Dim | Layers |
|-------|--------|-----------|--------|
| `brain-bzh/reve-base` | 72M | 512 | 22 |
| `brain-bzh/reve-large` | ~400M | 1250 | — |

> **Note:** You must agree to the data usage terms on HuggingFace before downloading.

## Features

| Feature | Description |
|---------|-------------|
| `ndarray` (default) | CPU backend with Rayon multi-threading |
| `blas-accelerate` | Apple Accelerate BLAS (macOS) |
| `openblas-system` | System OpenBLAS (Linux) |
| `wgpu` | GPU via wgpu (auto-detect Metal/Vulkan/DX12) |
| `metal` | Native Metal shaders (macOS) |
| `vulkan` | Native Vulkan shaders (Linux/Windows) |
| `hf-download` | HuggingFace Hub weight download |

## References

- El Ouahidi et al. (2025). *REVE: A Foundation Model for EEG — Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects.* NeurIPS 2025.
- [braindecode Python implementation](https://github.com/braindecode/braindecode)

## License

Apache-2.0
