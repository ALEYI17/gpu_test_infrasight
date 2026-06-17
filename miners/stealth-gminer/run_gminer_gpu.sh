#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")/../gminer"

DURATION=360
COOLDOWN=120

ALGORITHMS=(
  "kawpow rvn.2miners.com:6060 RPZApvMRGMdSTgSNBMGXpspuc67pN1cJXt 75"
  "etchash etc.2miners.com:1010 0x1234567890123456789012345678901234567890 75"
  "autolykos2 pool.br.woolypooly.com:3101 9iPzeu6vvqgxR6huzzGt2PsJNs5fy4xYxUeEE3W3SH9NKALUT4T 75"
)

TPID=""

cleanup() {
  echo "Interrupted — killing GMiner..."

  if [ -n "$TPID" ]; then
    kill -- -"$TPID" 2>/dev/null || true
    wait "$TPID" 2>/dev/null || true
  fi

  pkill -f miner 2>/dev/null || true
  exit 1
}

trap cleanup INT TERM EXIT

for ENTRY in "${ALGORITHMS[@]}"; do
  ALGO=$(echo "$ENTRY" | awk '{print $1}')
  POOL=$(echo "$ENTRY" | awk '{print $2}')
  WALLET=$(echo "$ENTRY" | awk '{print $3}')
  INTENSITY=$(echo "$ENTRY" | awk '{print $4}')

  echo "=============================================="
  echo " Running GMiner: $ALGO (intensity=$INTENSITY)"
  echo "=============================================="

  setsid "$SCRIPT_DIR/miner" \
    --algo "$ALGO" \
    --server "$POOL" \
    --user "$WALLET" \
    --cuda 1 \
    --intensity "$INTENSITY" &
  
  TPID=$!

  sleep "$DURATION"

  kill -- -"$TPID" 2>/dev/null || true
  wait "$TPID" 2>/dev/null || true

  TPID=""

  echo "Algorithm $ALGO finished. Cooling down ${COOLDOWN}s..."
  sleep "$COOLDOWN"
done

trap - EXIT

echo "All GMiner stealth tests completed."
