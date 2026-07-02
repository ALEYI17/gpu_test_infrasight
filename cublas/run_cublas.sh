#!/bin/bash
TIME_MINUTES=5
TIME=$((TIME_MINUTES * 60))

# Matrix sizes to sweep (square NxN)
SIZES="1024 2048 4096 8192"

# Dtypes to test
DTYPES="fp32 tf32 fp16 bf16"

python3 bench.py \
    --time "$TIME" \
    --sizes $SIZES \
    --dtypes $DTYPES
