#!/usr/bin/env bash
set -euo pipefail

# 1) Generate hashes
python3 generate_hashes.py

# 2) Find most recent generated dir
GEN_DIR=$(ls -d passwd_cracker/hashcat/generated/* | tail -n 1)
if [ -z "$GEN_DIR" ]; then
  echo "No generated directory found"
  exit 1
fi
echo "Using generated hashes from: $GEN_DIR"

# 3) Benchmark -> write to file in generated dir
hashcat -b |& tee "$GEN_DIR/hashcat_benchmark.txt"

# 4) Brute-force mask attacks (small masks for quick tests)
MASK='?l?l?l'   # ~17k combinations

echo "Running mask brute-force on MD5 hashes..."
hashcat -m 0 -a 3 "$GEN_DIR/hashes_md5.txt" "$MASK" --outfile="$GEN_DIR/md5_mask_cracked.txt" --potfile-disable --quiet || true

echo "Running mask brute-force on SHA1 hashes..."
hashcat -m 100 -a 3 "$GEN_DIR/hashes_sha1.txt" "$MASK" --outfile="$GEN_DIR/sha1_mask_cracked.txt" --potfile-disable --quiet || true

echo "Running mask brute-force on SHA256 hashes..."
hashcat -m 1400 -a 3 "$GEN_DIR/hashes_sha256.txt" "$MASK" --outfile="$GEN_DIR/sha256_mask_cracked.txt" --potfile-disable --quiet || true

echo "Done. Results are in: $GEN_DIR"

