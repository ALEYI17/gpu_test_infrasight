import argparse
import shutil
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--time",
        type=int,
        default=300,
        help="Benchmark duration in seconds",
    )

    parser.add_argument(
        "--images",
        type=int,
        default=200,
        help="Number of images to download",
    )

    parser.add_argument(
        "--model",
        default="yolov8x.pt",
        help="YOLO model",
    )

    args = parser.parse_args()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    print(f"Loading {args.model}...")
    model = YOLO(args.model)

    tmpdir = Path(tempfile.mkdtemp())

    try:
        print(f"Downloading {args.images} images...")

        dataset = load_dataset(
            "beans",
            split=f"train[:{args.images}]",
        )

        image_dir = tmpdir / "images"
        image_dir.mkdir()

        for i, sample in enumerate(dataset):
            sample["image"].save(image_dir / f"{i:05d}.jpg")

        images = sorted(image_dir.glob("*.jpg"))

        print(f"Running inference for {args.time} seconds...")

        start = time.time()
        total_predictions = 0

        while time.time() - start < args.time:
            for image in images:
                model.predict(
                    source=str(image),
                    device=0,
                    save=False,
                    verbose=False,
                )

                total_predictions += 1

                if time.time() - start >= args.time:
                    break

        elapsed = time.time() - start

        print()
        print("======================================")
        print("Benchmark finished")
        print("======================================")
        print(f"Elapsed time : {elapsed:.1f} s")
        print(f"Images used  : {len(images)}")
        print(f"Predictions  : {total_predictions}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
