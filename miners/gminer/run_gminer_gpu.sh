#!/bin/bash
set -e
IMAGE="alejandrosalamanca17/gminer-gpu-test:latest"
CONTAINER_NAME="gminer-gpu-test"
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
  -w / \
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
  echo " Running GMiner with algorithm: $ALGO"
  echo "=============================================="

  timeout ${DURATION}s docker exec -i $CONTAINER_NAME \
    ./miner --algo "$ALGO" \
      --server $POOL \
      --user $WALLET \
      --cuda 1 --nvml 0 || true

  echo "Algorithm $ALGO finished (or timed out). Cooling down GPU for 60s..."
  sleep 60
done

echo "All GMiner stress tests completed."
