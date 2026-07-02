import argparse
import time
import numpy as np
import torch
from transformers import pipeline

SAMPLE_RATE = 16000
# Use a handful of different durations so you're not just benchmarking one fixed-length clip
DURATIONS_SEC = [4, 6, 8, 10, 12]

def make_synthetic_audio(duration_sec: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate audio that looks like speech-ish energy (not just pure sine/silence)."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Layer a few tones + noise so it's not degenerate silence/pure-tone
    signal = (
        0.3 * np.sin(2 * np.pi * 110 * t)
        + 0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.1 * np.random.randn(n_samples)
    )
    # Normalize to roughly [-1, 1] like real audio
    signal = signal / np.max(np.abs(signal))
    return signal.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=300, help="Benchmark duration in seconds")
    parser.add_argument("--model", default="openai/whisper-large-v3", help="Whisper model")
    args = parser.parse_args()

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Loading {args.model}...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=0,
        dtype=torch.float16,
    )

    print("Generating synthetic audio samples...")
    audio_files = [make_synthetic_audio(d) for d in DURATIONS_SEC]
    print(f"Generated {len(audio_files)} synthetic clips.")

    print(f"Running inference for {args.time} seconds...")
    start = time.time()
    total = 0
    while time.time() - start < args.time:
        for audio in audio_files:
            pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
            total += 1
            if time.time() - start >= args.time:
                break
    elapsed = time.time() - start

    print()
    print("======================================")
    print("Benchmark finished")
    print("======================================")
    print(f"Elapsed time : {elapsed:.1f} s")
    print(f"Audio clips  : {len(audio_files)}")
    print(f"Inferences   : {total}")

if __name__ == "__main__":
    main()
