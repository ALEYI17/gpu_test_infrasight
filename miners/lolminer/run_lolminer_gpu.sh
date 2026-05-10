#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$0")"

ALGORITHMS=(
  "ETHASH"
  "ETCHASH"
  "GRAM"
)

for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running lolMiner benchmark: $ALGO"
  echo "=============================================="

  $SCRIPT_DIR/lolMiner --benchmark "$ALGO" || true

  echo "Benchmark for $ALGO finished. Cooling down GPU for 60s..."
  sleep 60
done

echo "All lolMiner benchmarks completed."
