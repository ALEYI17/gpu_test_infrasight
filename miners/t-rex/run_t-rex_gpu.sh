#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$0")"
DURATION=350

ALGORITHMS=(
  "octopus"
  "tensority"
  "blake3"
  "autolykos2"
  "progpow-veil"
)

for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running GPU stress test with algorithm: $ALGO"
  echo "=============================================="

  timeout ${DURATION}s $SCRIPT_DIR/t-rex \
    -B \
    -a "$ALGO" || true

  echo "Algorithm $ALGO finished (or timed out). Cooling down GPU for 60s..."
  sleep 60
done

echo "All GPU stress tests completed."
