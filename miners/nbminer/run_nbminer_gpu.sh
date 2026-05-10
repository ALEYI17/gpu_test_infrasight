#!/bin/bash
set -e
SCRIPT_DIR="$(dirname "$0")"
DURATION=350

ALGORITHMS=(
  "ethash eth.2miners.com:2020 0x1234567890123456789012345678901234567890.worker1"
  "etchash etc.2miners.com:1010 0x1234567890123456789012345678901234567890.worker1"
  "kawpow rvn.2miners.com:6060 RPZApvMRGMdSTgSNBMGXpspuc67pN1cJXt.worker1"
)

for ENTRY in "${ALGORITHMS[@]}"; do
  ALGO=$(echo $ENTRY | awk '{print $1}')
  POOL=$(echo $ENTRY | awk '{print $2}')
  WALLET=$(echo $ENTRY | awk '{print $3}')

  echo "=============================================="
  echo " Running NBMiner with algorithm: $ALGO"
  echo "=============================================="

  timeout ${DURATION}s $SCRIPT_DIR/nbminer \
    -a "$ALGO" \
    -o stratum+tcp://$POOL \
    -u $WALLET \
    --no-health || true

  echo "Algorithm $ALGO finished (or timed out). Cooling down GPU for 60s..."
  sleep 60
done

echo "All NBMiner stress tests completed."
