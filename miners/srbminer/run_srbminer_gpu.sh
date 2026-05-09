#!/bin/bash
set -e
IMAGE="alejandrosalamanca17/srbminer-gpu-test:latest"
CONTAINER_NAME="srbminer-gpu-test"
DURATION=350

cleanup() {
  echo "Cleaning up container..."
  docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -dit \
  --name $CONTAINER_NAME \
  --gpus all \
  --network mining-net \
  -w /SRBMiner-Multi-3-2-8 \
  $IMAGE bash

ALGORITHMS=(
  "ethash eth.2miners.com:2020 0x1234567890123456789012345678901234567890"
  "etchash etc.2miners.com:1010 0x1234567890123456789012345678901234567890"
  "kawpow rvn.2miners.com:6060 RPZApvMRGMdSTgSNBMGXpspuc67pN1cJXt"
)

for ENTRY in "${ALGORITHMS[@]}"; do
  ALGO=$(echo $ENTRY | awk '{print $1}')
  POOL=$(echo $ENTRY | awk '{print $2}')
  WALLET=$(echo $ENTRY | awk '{print $3}')

  echo "=============================================="
  echo " Running SRBMiner with algorithm: $ALGO"
  echo "=============================================="

  timeout ${DURATION}s docker exec -i $CONTAINER_NAME \
    ./SRBMiner-MULTI --algorithm "$ALGO" \
      --pool $POOL \
      --wallet $WALLET \
      --worker worker1 \
      --cpu-threads 0 || true

  echo "Algorithm $ALGO finished (or timed out). Cooling down GPU for 60s..."
  sleep 60
done

echo "All SRBMiner stress tests completed."
