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

docker exec -it $CONTAINER_NAME \
  ./xmrig --cuda \
  --cuda-loader=/workspace/xmrig-cuda/build/libxmrig-cuda.so \
  --stress

echo "Stress test finished."
