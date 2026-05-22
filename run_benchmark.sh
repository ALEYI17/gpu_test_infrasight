#!/usr/bin/env bash
# =============================================================================
# eBPF CUDA Tracer — Overhead Measurement Quick-Start
# =============================================================================
# Place this next to measure_overhead.py at your project root.
# Run as root: sudo bash run_benchmark.sh
# (All child processes including the eBPF loader inherit root automatically.
#  Never add 'sudo' inside LOADER — nested sudo fails without a TTY.)
#
# Before running:
#   Edit CONFIG["experiments"] in measure_overhead.py to uncomment the same
#   workloads you have active in your main runner.
# =============================================================================

set -euo pipefail

SCRIPT="python3 measure_overhead.py"
RESULTS_DIR="./results/overhead"
ITERATIONS=5                     # runs per workload per mode
LOADER_WAIT=2                    # seconds to wait after loader starts before running workload

# Full loader command — no 'sudo' prefix here.
# This script is already run as root (sudo bash run_benchmark.sh), so every
# subprocess it spawns inherits root automatically. A nested sudo would fail
# because there is no TTY available for password input.
LOADER="./main \
  --tracer=fingerprint \
  --server-addr=localhost \
  --server-port=8080 \
  --cuda-lib=/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
  --time-window=2"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Baseline
# Make sure your eBPF loader is NOT running before this step.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 1 — Baseline (no eBPF probes)"
echo "════════════════════════════════════════════════════════"
echo "  Confirm your loader is stopped, then press Enter..."
read -r

sudo $SCRIPT run \
  --mode baseline \
  --iterations "$ITERATIONS" \
  --results-dir "$RESULTS_DIR"

# To run only specific experiments instead of all active ones:
# sudo $SCRIPT run --mode baseline --only miner_xmrig dl_cnn_train --iterations "$ITERATIONS" --results-dir "$RESULTS_DIR"

# To run only one category:
# sudo $SCRIPT run --mode baseline --categories other --iterations "$ITERATIONS" --results-dir "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Instrumented
# The script starts your loader automatically via --loader, waits for probes
# to attach, runs all workloads, then stops the loader.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 2 — Instrumented (eBPF probes active)"
echo "════════════════════════════════════════════════════════"

sudo $SCRIPT run \
  --mode instrumented \
  --loader "$LOADER" \
  --loader-wait "$LOADER_WAIT" \
  --iterations "$ITERATIONS" \
  --results-dir "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Per-probe PMU profiling (optional, needs perf support)
# Measures hardware counters (cycles, instructions, L1/LLC misses) per probe
# while a specific workload runs. Run once per experiment you care about.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 3 — Per-probe bpftool profiling (optional)"
echo "════════════════════════════════════════════════════════"

# Uncomment and adjust --name to match one of your CONFIG["experiments"] names:
# sudo $SCRIPT probe-profile \
#   --name dl_cnn_train \
#   --duration 60 \
#   --loader "$LOADER" \
#   --loader-wait "$LOADER_WAIT" \
#   --results-dir "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Report
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  PHASE 4 — Overhead Report"
echo "════════════════════════════════════════════════════════"

$SCRIPT report --results-dir "$RESULTS_DIR"

echo ""
echo "[done] Output files in $RESULTS_DIR/"
echo "  overhead_report.json        — full machine-readable combined report"
echo "  <name>_baseline.json        — per-workload baseline iterations + aggregate"
echo "  <name>_instrumented.json    — per-workload instrumented iterations + BPF stats"
echo "  probe_profiles.json         — per-probe PMU counters (if phase 3 was run)"
