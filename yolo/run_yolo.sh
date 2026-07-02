#!/bin/bash

TIME=300

# Available models (largest -> smallest)
#
# yolov8x.pt  - Extra Large (~68M parameters)
# yolov8l.pt  - Large       (~44M parameters)
# yolov8m.pt  - Medium      (~26M parameters)
# yolov8s.pt  - Small       (~11M parameters)
# yolov8n.pt  - Nano        (~3M parameters)
#
# Larger models:
#   + More GPU utilization
#   + Higher accuracy
#   - Slower inference
#
# Smaller models:
#   + Faster inference
#   + Lower GPU usage
#   - Lower accuracy

MODEL="yolov8x.pt"

python3 infer.py \
    --time "$TIME" \
    --model "$MODEL"
