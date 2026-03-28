//! # reve-rs — REVE EEG Foundation Model inference in Rust
//!
//! Pure-Rust inference for the REVE (Representation for EEG with Versatile
//! Embeddings) foundation model, built on [Burn 0.20](https://burn.dev).
//!
//! REVE is a pretrained EEG model that generalizes across diverse electrode
//! configurations using 4D positional encoding (x, y, z, t). It was pretrained
//! on 60,000+ hours of EEG data from 92 datasets spanning 25,000 subjects.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use reve_rs::ReveEncoder;
//!
//! let (model, _ms) = ReveEncoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//! ```

pub mod config;
pub mod data;
pub mod encoder;
pub mod model;
pub mod position_bank;
pub mod weights;

// Flat re-exports
pub use config::ModelConfig;
pub use data::{InputBatch, build_batch, build_batch_named};
pub use encoder::{ReveEncoder, ReveOutput, EncodingResult};
pub use position_bank::PositionBank;
pub use weights::WeightMap;
