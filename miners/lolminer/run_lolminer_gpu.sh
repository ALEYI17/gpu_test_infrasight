#!/bin/bash
set -e

IMAGE="alejandrosalamanca17/lolminer-gpu:latest"
CONTAINER_NAME="lolminer-gpu-test"

cleanup() {
  echo "Cleaning up container..."
  docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -dit \
  --name $CONTAINER_NAME \
  --gpus all \
  -w /workspace/lolminer/1.98 \
  $IMAGE bash

ALGORITHMS=(
  "ETHASH"
  "ETCHASH"
  "GRAM"
)

# Loop through each algorithm
for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running lolMiner benchmark: $ALGO"
  echo "=============================================="

  docker exec -it $CONTAINER_NAME \
    ./lolMiner --benchmark "$ALGO"

  echo "Benchmark for $ALGO finished. Cooling down GPU for 60s..."
  sleep 60
done

echo "All lolMiner benchmarks completed."
cleanup
