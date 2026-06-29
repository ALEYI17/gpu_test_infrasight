import argparse

import torch
from diffusers import StableDiffusionXLPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to generate",
    )

    parser.add_argument(
        "--output",
        default="generated_image.png",
        help="Output image filename",
    )

    args = parser.parse_args()

    model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    print("Loading Stable Diffusion model...")

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Current device:", torch.cuda.current_device())
    print("GPU:", torch.cuda.get_device_name(0))

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )

    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    pipe = pipe.to("cuda")

    print(f"Prompt: {args.prompt}")

    image = pipe(
        args.prompt,
        height=768,
        width=768,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]

    image.save(args.output)

    print(f"Image saved as {args.output}")


if __name__ == "__main__":
    main()
