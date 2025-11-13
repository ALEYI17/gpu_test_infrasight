#!/bin/bash
set -e

IMAGE="alejandrosalamanca17/xmrig-gpu:latest"
CONTAINER_NAME="xmrig-gpu-test"

cleanup() {
  echo "Cleaning up container..."
  docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -dit \
  --name $CONTAINER_NAME \
  --gpus all \
  -w /workspace/xmrig/build \
  $IMAGE bash

ALGORITHMS=(
  "rx/arq"
  "rx/sfx"
)

# Run each algorithm sequentially
for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running GPU stress test with algorithm: $ALGO"
  echo "=============================================="

  docker exec -it $CONTAINER_NAME \
    ./xmrig \
      --cuda \
      --cuda-loader=/workspace/xmrig-cuda/build/libxmrig-cuda.so \
      --no-cpu \
      --algo "$ALGO" \
      --stress

  echo "Algorithm $ALGO finished. Cooling down GPU for 60s..."
  sleep 60
done

echo "All GPU stress tests completed."
cleanup
