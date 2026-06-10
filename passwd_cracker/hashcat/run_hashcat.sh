#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHCAT_BIN="$SCRIPT_DIR/hashcat/hashcat"
GEN_BASE="$SCRIPT_DIR/generated"          # ← fixed path
BENCHMARK_DURATION=300
COOLDOWN=60

if [ ! -x "$HASHCAT_BIN" ]; then
    echo "Hashcat binary not found: $HASHCAT_BIN"
    exit 1
fi

# --- CUDA toolchain ---
export CUDA_HOME=/usr/local/cuda-13      # ← fixed version
export CUDA_PATH=/usr/local/cuda-13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PTXAS="$(command -v ptxas || true)"
if [ -z "$PTXAS" ]; then
    echo "ERROR: ptxas not found in PATH"
    exit 1
fi

TPID=""
cleanup() {
  if [ -n "$TPID" ]; then
    kill -- -$TPID 2>/dev/null || true
    wait $TPID 2>/dev/null || true
  fi
  pkill -f hashcat 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# 1) Generate hashes
python3 "$SCRIPT_DIR/generate_hashes.py"

# 2) Find most recent generated dir
GEN_DIR="$(find "$GEN_BASE" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
if [ -z "${GEN_DIR:-}" ]; then
  echo "No generated directory found in $GEN_BASE"
  exit 1
fi
echo "Using hashes from: $GEN_DIR"

# 3) Benchmark with timeout
echo "=============================================="
echo " Running hashcat benchmark (${BENCHMARK_DURATION}s)"
echo "=============================================="
setsid "$HASHCAT_BIN" -b -D 2 --backend-ignore-opencl \
  > "$GEN_DIR/hashcat_benchmark.txt" 2>&1 &
TPID=$!
sleep $BENCHMARK_DURATION
kill -- -$TPID 2>/dev/null || true
pkill -f "hashcat.*-b" 2>/dev/null || true
wait $TPID 2>/dev/null || true
TPID=""
echo "Benchmark done. Cooling down ${COOLDOWN}s..."
sleep $COOLDOWN
# 4) Brute-force mask attacks — run to completion
MASK='?l?l?l'
echo "Running mask brute-force on MD5 hashes..."
"$HASHCAT_BIN" \
  -m 0 -a 3 -D 2 \
  "$GEN_DIR/hashes_md5.txt" "$MASK" \
  --outfile="$GEN_DIR/md5_mask_cracked.txt" \
  --potfile-disable --quiet || true

echo "Running mask brute-force on SHA1 hashes..."
"$HASHCAT_BIN" \
  -m 100 -a 3 -D 2 \
  "$GEN_DIR/hashes_sha1.txt" "$MASK" \
  --outfile="$GEN_DIR/sha1_mask_cracked.txt" \
  --potfile-disable --quiet || true

echo "Running mask brute-force on SHA256 hashes..."
"$HASHCAT_BIN" \
  -m 1400 -a 3 -D 2 \
  "$GEN_DIR/hashes_sha256.txt" "$MASK" \
  --outfile="$GEN_DIR/sha256_mask_cracked.txt" \
  --potfile-disable --quiet || true

trap - EXIT
echo "Done. Results in: $GEN_DIR"
