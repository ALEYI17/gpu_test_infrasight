#!/usr/bin/env bash
# =============================================================================
# eBPF CUDA Tracer — Overhead Benchmark Runner
# =============================================================================
# Run as root: sudo bash run_benchmark.sh
#
# Before running:
#   Uncomment the experiments you want in CONFIG["experiments"] inside
#   measure_overhead.py — same list as your main runner.
# =============================================================================

set -euo pipefail

SCRIPT="sudo env PATH=\"$PATH\" python3 measure_overhead.py"
RESULTS_DIR="./results/overhead"
ITERATIONS=3
LOADER_WAIT=120

# No --time-window here — it is injected automatically per window
# from CONFIG["time_windows"] = [1, 2, 5]
LOADER="/home/aleyi/Documents/InfraSight_gpu/main \
  --tracer=fingerprint \
  --server-addr=localhost \
  --server-port=8080 \
  --cuda-lib=/usr/lib/x86_64-linux-gnu/libcuda.so"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Baseline (no probes, no data collection)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 1 — Baseline (no eBPF probes)"
echo "════════════════════════════════════════════════════════"
echo "  Make sure your loader is NOT running, then press Enter..."
read -r

sudo $SCRIPT run \
  --mode baseline \
  --iterations "$ITERATIONS" \
  --results-dir "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Instrumented + data collection
# Loader is started/stopped automatically per time window (1s, 2s, 5s).
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 2 — Instrumented + collect (tw=1s, 2s, 5s)"
echo "════════════════════════════════════════════════════════"

sudo $SCRIPT run \
  --mode instrumented \
  --collect \
  --iterations "$ITERATIONS" \
  --loader "$LOADER" \
  --loader-wait "$LOADER_WAIT" \
  --results-dir "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Per-probe PMU profiling (optional, bare metal only)
# Uncomment one block per experiment you want to profile.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 3 — Per-probe bpftool profiling (optional)"
echo "════════════════════════════════════════════════════════"

# sudo $SCRIPT probe-profile \
#   --name llm_roberta \
#   --duration 60 \
#   --loader "$LOADER" \
#   --loader-wait "$LOADER_WAIT" \
#   --results-dir "$RESULTS_DIR"

# sudo $SCRIPT probe-profile \
#   --name miner_xmrig \
#   --duration 60 \
#   --loader "$LOADER" \
#   --loader-wait "$LOADER_WAIT" \
#   --results-dir "$RESULTS_DIR"

# sudo $SCRIPT probe-profile \
#   --name dl_cnn_train \
#   --duration 60 \
#   --loader "$LOADER" \
#   --loader-wait "$LOADER_WAIT" \
#   --results-dir "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Overhead report
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 4 — Overhead Report"
echo "════════════════════════════════════════════════════════"

sudo env PATH="$PATH" python3 measure_overhead.py report --results-dir "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Merge datasets per time window
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 5 — Merge datasets"
echo "════════════════════════════════════════════════════════"

##python3 merge_dataset.py

# echo ""
# echo "[done]"
# echo "  results/overhead/                       — overhead JSON files"
# echo "  final_gpu_time_windows_tw1.parquet      — merged dataset tw=1s"
# echo "  final_gpu_time_windows_tw2.parquet      — merged dataset tw=2s"
# echo "  final_gpu_time_windows_tw5.parquet      — merged dataset tw=5s"
