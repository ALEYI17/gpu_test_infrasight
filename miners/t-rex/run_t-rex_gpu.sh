#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$0")"
DURATION=360
COOLDOWN=300 
ALGORITHMS=(
  "octopus"
  "tensority"
  "blake3"
)

TPID=""

cleanup() {
  echo "Interrupted — killing lolMiner..."
  if [ -n "$TPID" ]; then
    kill -- -$TPID 2>/dev/null || true
    wait $TPID 2>/dev/null || true
  fi
  # also pkill in case setsid didn't track it
  pkill -f t-rex 2>/dev/null || true
  exit 1
}

trap cleanup INT TERM EXIT

for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running t-rex: $ALGO (${DURATION}s)"
  echo "=============================================="

  # setsid puts t-rex in its own process group
  # kill -- -$PID sends SIGTERM to the entire group
  setsid $SCRIPT_DIR/t-rex -B -a "$ALGO" &
  TPID=$!
  sleep $DURATION
  kill -- -$TPID 2>/dev/null || true
  wait $TPID 2>/dev/null || true
  TPID=""


  echo "Algorithm $ALGO finished. Cooling down ${COOLDOWN}s..."
  sleep $COOLDOWN
done
trap - EXIT
echo "All t-rex tests completed."

