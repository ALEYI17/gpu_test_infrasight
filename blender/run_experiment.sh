#!/bin/bash
set -e

BENCH="/home/aleyi/Downloads/benchmark-launcher-cli"

echo "=== Starting Blender Benchmark (Interactive Mode) ==="
echo "Using CLI at: $BENCH"
echo

"$BENCH" benchmark monster --blender-version 4.4.0 --device-type CUDA

"$BENCH" benchmark classroom --blender-version 4.4.0 --device-type CUDA

echo
echo "=== Benchmark finished ==="
