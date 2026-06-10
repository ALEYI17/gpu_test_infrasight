#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")"
DURATION=360
COOLDOWN=120

ALGORITHMS=(
  "etchash etc.2miners.com:1010 0x1234567890123456789012345678901234567890"
  "kawpow rvn.2miners.com:6060 RPZApvMRGMdSTgSNBMGXpspuc67pN1cJXt"
  "autolykos2 pool.br.woolypooly.com:3101 9iPzeu6vvqgxR6huzzGt2PsJNs5fy4xYxUeEE3W3SH9NKALUT4T"
)

TPID=""

cleanup() {
  echo "Interrupted — killing SRBMiner..."

  if [ -n "$TPID" ]; then
    kill -- -"$TPID" 2>/dev/null || true
    wait "$TPID" 2>/dev/null || true
  fi

  pkill -f SRBMiner-MULTI 2>/dev/null || true
  exit 1
}

trap cleanup INT TERM EXIT

for ENTRY in "${ALGORITHMS[@]}"; do
  ALGO=$(echo "$ENTRY" | awk '{print $1}')
  POOL=$(echo "$ENTRY" | awk '{print $2}')
  WALLET=$(echo "$ENTRY" | awk '{print $3}')

  echo "=============================================="
  echo " Running SRBMiner: $ALGO"
  echo "=============================================="

  setsid "$SCRIPT_DIR/SRBMiner-MULTI" \
    --algorithm "$ALGO" \
    --pool "$POOL" \
    --wallet "$WALLET" \
    --worker worker1 \
    --cpu-threads 0 &
  
  TPID=$!

  sleep "$DURATION"

  # Kill entire process group
  kill -- -"$TPID" 2>/dev/null || true
  wait "$TPID" 2>/dev/null || true

  TPID=""

  echo "Algorithm $ALGO finished. Cooling down ${COOLDOWN}s..."
  sleep "$COOLDOWN"
done

# Prevent cleanup() from running on normal exit
trap - EXIT

echo "All SRBMiner stress tests completed."
