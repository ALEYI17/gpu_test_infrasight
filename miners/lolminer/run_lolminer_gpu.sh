#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$0")"
DURATION=120
COOLDOWN=25

ALGORITHMS=(
  "ETHASH"
  "ETCHASH"
  "GRAM"
)

TPID=""

cleanup() {
  echo "Interrupted — killing lolMiner..."
  if [ -n "$TPID" ]; then
    kill -- -$TPID 2>/dev/null || true
    wait $TPID 2>/dev/null || true
  fi
  # also pkill in case setsid didn't track it
  pkill -f lolMiner 2>/dev/null || true
  exit 1
}

trap cleanup INT TERM EXIT

for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running lolMiner benchmark: $ALGO (${DURATION}s)"
  echo "=============================================="

  setsid $SCRIPT_DIR/lolMiner --benchmark "$ALGO" &
  TPID=$!
  sleep $DURATION
  kill -- -$TPID 2>/dev/null || true
  wait $TPID 2>/dev/null || true
  TPID=""

  echo "Benchmark for $ALGO finished. Cooling down ${COOLDOWN}s..."
  sleep $COOLDOWN
done

trap - EXIT
echo "All lolMiner benchmarks completed."
