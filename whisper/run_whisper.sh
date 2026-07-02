#!/bin/bash

TIME=300

# Available Whisper models (largest -> smallest)
#
# openai/whisper-large-v3   (~1.55B params)
# openai/whisper-large-v2
# openai/whisper-medium     (~769M params)
# openai/whisper-small      (~244M params)
# openai/whisper-base       (~74M params)
# openai/whisper-tiny       (~39M params)
#
# Larger models:
#   + Higher GPU utilization
#   + Better transcription accuracy
#   - Slower inference

MODEL="openai/whisper-large-v3"

python3 infer.py \
    --time "$TIME" \
    --model "$MODEL"
