#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")"
DURATION=360
COOLDOWN=120

ALGORITHMS=(
  "etchash etc.2miners.com:1010 0x1234567890123456789012345678901234567890.worker1"
  "kawpow rvn.2miners.com:6060 RPZApvMRGMdSTgSNBMGXpspuc67pN1cJXt.worker1"
)

TPID=""

cleanup() {
  echo "Interrupted — killing NBMiner..."

  if [ -n "$TPID" ]; then
    kill -- -"$TPID" 2>/dev/null || true
    wait "$TPID" 2>/dev/null || true
  fi

  pkill -f nbminer 2>/dev/null || true
  exit 1
}

trap cleanup INT TERM EXIT

for ENTRY in "${ALGORITHMS[@]}"; do
  ALGO=$(echo "$ENTRY" | awk '{print $1}')
  POOL=$(echo "$ENTRY" | awk '{print $2}')
  WALLET=$(echo "$ENTRY" | awk '{print $3}')

  echo "=============================================="
  echo " Running NBMiner: $ALGO"
  echo "=============================================="

  # start in its own process group
  setsid "$SCRIPT_DIR/nbminer" \
    -a "$ALGO" \
    -o "stratum+tcp://$POOL" \
    -u "$WALLET" \
    --no-health &
  
  TPID=$!

  sleep "$DURATION"

  # stop entire process group cleanly
  kill -- -"$TPID" 2>/dev/null || true
  wait "$TPID" 2>/dev/null || true

  TPID=""

  echo "Algorithm $ALGO finished. Cooling down GPU for ${COOLDOWN}s..."
  sleep "$COOLDOWN"
done

# IMPORTANT: prevent EXIT trap from corrupting success status
trap - EXIT

echo "All NBMiner stress tests completed."
