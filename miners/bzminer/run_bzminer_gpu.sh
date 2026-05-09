#!/bin/bash
set -e
IMAGE="alejandrosalamanca17/bzminer-gpu-test:latest"
CONTAINER_NAME="bzminer-gpu-test"
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
  -w /bzminer_v24.0.2_linux \
  $IMAGE bash

ALGORITHMS=(
  "ethash 0x1234567890123456789012345678901234567890 stratum+tcp://eth.2miners.com:2020"
  "etchash 0x1234567890123456789012345678901234567890 stratum+tcp://etc.2miners.com:1010"
  "rvn RPZApvMRGMdSTgSNBMGXpspuc67pN1cJXt stratum+tcp://rvn.2miners.com:6060"
)

for ENTRY in "${ALGORITHMS[@]}"; do
  ALGO=$(echo $ENTRY | awk '{print $1}')
  WALLET=$(echo $ENTRY | awk '{print $2}')
  POOL=$(echo $ENTRY | awk '{print $3}')

  echo "=============================================="
  echo " Running BZMiner with algorithm: $ALGO"
  echo "=============================================="

  timeout ${DURATION}s docker exec -i $CONTAINER_NAME \
    ./bzminer -a "$ALGO" \
      -w $WALLET \
      -p $POOL \
      -r worker1 --nc 1 --no_watchdog || true

  echo "Algorithm $ALGO finished (or timed out). Cooling down GPU for 60s..."
  sleep 60
done

echo "All BZMiner stress tests completed."
