#!/bin/bash
set -e

IMAGE="alejandrosalamanca17/trex-gpu:latest"
CONTAINER_NAME="t-rex-gpu-test"
DURATION=350   # seconds per algorithm

cleanup() {
  echo "Cleaning up container..."
  docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -dit \
  --name $CONTAINER_NAME \
  --gpus all \
  -w /trex \
  $IMAGE bash

ALGORITHMS=(
  "octopus"
  "tensority"
  "blake3"
  "autolykos2"
  "progpow-veil"
)

for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running GPU stress test with algorithm: $ALGO"
  echo "=============================================="

  timeout ${DURATION}s docker exec -i $CONTAINER_NAME \
    ./t-rex \
      -B \
      -a "$ALGO" || true

  echo "Algorithm $ALGO finished (or timed out). Cooling down GPU for 60s..."
  sleep 60
done

echo "All GPU stress tests completed."
