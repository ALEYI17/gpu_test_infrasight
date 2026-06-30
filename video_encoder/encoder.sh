#!/bin/bash

set -e

VIDEO="input4k.mp4"
COOLDOWN=60

# ------------------------------------------------------------------
# Create the synthetic input video if it doesn't already exist
# ------------------------------------------------------------------
if [ ! -f "$VIDEO" ]; then
    echo "Input video not found."
    echo "Generating a synthetic 4K 60 FPS test video..."

    ffmpeg \
        -y \
        -f lavfi \
        -i testsrc2=duration=360:size=3840x2160:rate=60 \
        -pix_fmt yuv420p \
        "$VIDEO"

    echo "Video created."
else
    echo "Using existing video: $VIDEO"
fi

# ------------------------------------------------------------------
# List of transcoding experiments
# ------------------------------------------------------------------

OUTPUTS=(
    "output_h264.mp4"
    "output_hevc.mp4"
    "output_1080p.mp4"
    "output_720p.mp4"
)

COMMANDS=(
'-hwaccel cuda -hwaccel_output_format cuda -i input4k.mp4 -c:v h264_nvenc output_h264.mp4'

'-hwaccel cuda -hwaccel_output_format cuda -i input4k.mp4 -c:v hevc_nvenc output_hevc.mp4'

'-hwaccel cuda -hwaccel_output_format cuda -i input4k.mp4 -vf scale_cuda=1920:1080 -c:v h264_nvenc output_1080p.mp4'

'-hwaccel cuda -hwaccel_output_format cuda -i input4k.mp4 -vf scale_cuda=1280:720 -c:v h264_nvenc output_720p.mp4'
)

# ------------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------------

for i in "${!COMMANDS[@]}"; do

    echo
    echo "=========================================="
    echo "Experiment $((i+1))/4"
    echo "=========================================="

    rm -f "${OUTPUTS[$i]}"

    ffmpeg -y ${COMMANDS[$i]}

    if [ "$i" -lt $((${#COMMANDS[@]}-1)) ]; then
        echo
        echo "Cooling down GPU for ${COOLDOWN} seconds..."
        sleep "$COOLDOWN"
    fi

done

echo
echo "All transcoding experiments completed."
