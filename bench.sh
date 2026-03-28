#!/bin/bash
set -e

cd /Users/Shared/reve-rs

echo "=== Building Rust backends ==="

echo "Building ndarray (default)..."
cargo build --release --example benchmark 2>&1 | tail -1
cp target/release/examples/benchmark target/release/examples/benchmark_ndarray

echo "Building ndarray + Accelerate BLAS..."
cargo build --release --example benchmark --features blas-accelerate 2>&1 | tail -1
cp target/release/examples/benchmark target/release/examples/benchmark_accelerate

echo "Building wgpu (Metal)..."
cargo build --release --example benchmark --no-default-features --features metal 2>&1 | tail -1
cp target/release/examples/benchmark target/release/examples/benchmark_metal || true

echo ""
echo "=== Running benchmarks ==="
python3 bench_full.py
