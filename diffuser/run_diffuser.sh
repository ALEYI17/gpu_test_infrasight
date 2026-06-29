#!/bin/bash

COOLDOWN=60

PROMPTS=(
"A futuristic city at sunset with flying cars"
"A medieval castle on top of snowy mountains"
"A cyberpunk street full of neon lights at night"
#"A photorealistic astronaut walking through a tropical rainforest"
)

mkdir -p outputs

for i in "${!PROMPTS[@]}"; do

    echo "======================================"
    echo "Experiment $((i+1))/${#PROMPTS[@]}"
    echo "======================================"

    python generate.py \
        --prompt "${PROMPTS[$i]}" \
        --output "outputs/image_$((i+1)).png"

    if [ "$i" -lt $((${#PROMPTS[@]} - 1)) ]; then
        echo "Cooling down GPU for ${COOLDOWN}s..."
        sleep "$COOLDOWN"
    fi

done

echo "All prompts completed."
