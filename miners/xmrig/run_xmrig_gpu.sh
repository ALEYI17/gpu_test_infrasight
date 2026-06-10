#!/bin/bash
set -e
IMAGE="alejandrosalamanca17/xmrig-gpu:linuxmint"
CONTAINER_NAME="xmrig-gpu-test"
DURATION=120
COOLDOWN=25

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
  "rx/0"
  "rx/wow"
  "rx/sfx"
)

for ALGO in "${ALGORITHMS[@]}"; do
  echo "=============================================="
  echo " Running GPU stress test with algorithm: $ALGO (${DURATION}s)"
  echo "=============================================="

  # docker exec runs inside the container — kill the docker exec process
  # and the container runtime will clean up xmrig inside
  docker exec $CONTAINER_NAME \
    ./xmrig \
      --cuda \
      --cuda-loader=/workspace/xmrig-cuda/build/libxmrig-cuda.so \
      --no-cpu \
      --algo "$ALGO" \
      --stress &
  TPID=$!
  sleep $DURATION
  kill $TPID 2>/dev/null || true
  wait $TPID 2>/dev/null || true
  # also stop xmrig inside the container
  docker exec $CONTAINER_NAME pkill -f xmrig 2>/dev/null || true

  echo "Algorithm $ALGO finished. Cooling down ${COOLDOWN}s..."
  sleep $COOLDOWN
done

echo "All GPU stress tests completed."
cleanup
