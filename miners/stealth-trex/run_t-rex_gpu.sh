#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")/../t-rex"
DURATION=360
COOLDOWN=120

ALGORITHMS=(
  "octopus 9"
  "blake3 12"
)

TPID=""

cleanup() {
  trap - INT TERM EXIT

  echo "Interrupted — killing t-rex..."

  if [ -n "$TPID" ]; then
    kill -- -"$TPID" 2>/dev/null || true
    wait "$TPID" 2>/dev/null || true
  fi

  pkill -f t-rex 2>/dev/null || true
  exit 1
}

trap cleanup INT TERM EXIT

for ENTRY in "${ALGORITHMS[@]}"; do
  IFS=' ' read -r ALGO INTENSITY <<< "$ENTRY"

  echo "=============================================="
  echo " Running t-rex: $ALGO (${DURATION}s)"
  echo " Intensity: $INTENSITY"
  echo "=============================================="

  # setsid puts t-rex in its own process group
  setsid "$SCRIPT_DIR/t-rex" \
    -B \
    -a "$ALGO" \
    -i "$INTENSITY" &
  
  TPID=$!

  sleep "$DURATION"

  # Kill the entire process group
  kill -- -"$TPID" 2>/dev/null || true
  wait "$TPID" 2>/dev/null || true

  TPID=""

  echo "Algorithm $ALGO finished. Cooling down ${COOLDOWN}s..."
  sleep "$COOLDOWN"
done

# Prevent cleanup() from running on normal exit
trap - EXIT

echo "All t-rex tests completed."
