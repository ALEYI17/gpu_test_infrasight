import argparse
import time
import torch

def benchmark_gemm(size: int, dtype: torch.dtype, duration: float, device="cuda"):
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.time()
    iters = 0
    while time.time() - start < duration:
        torch.matmul(a, b)
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.time() - start

    flops_per_matmul = 2 * (size ** 3)  # multiply-add
    total_flops = flops_per_matmul * iters
    tflops = total_flops / elapsed / 1e12
    return iters, elapsed, tflops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=int, default=300, help="Total benchmark duration in seconds (split across all configs)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "tf32", "fp16", "bf16"])
    args = parser.parse_args()

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    dtype_map = {
        "fp32": torch.float32,
        "tf32": torch.float32,  # enabled via backend flag below
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtypes = {name: dtype_map[name] for name in args.dtypes}

    n_configs = len(dtypes) * len(args.sizes)
    per_config_time = max(args.time / n_configs, 1)
    print(f"Total time: {args.time}s across {n_configs} configs -> {per_config_time:.1f}s each")

    print("\n{:<8} {:<8} {:>10} {:>10} {:>12}".format("dtype", "size", "iters", "sec", "TFLOPs"))
    results = []
    overall_start = time.time()
    for name, dtype in dtypes.items():
        torch.backends.cuda.matmul.allow_tf32 = (name == "tf32")
        for size in args.sizes:
            iters, elapsed, tflops = benchmark_gemm(size, dtype, per_config_time)
            print(f"{name:<8} {size:<8} {iters:>10} {elapsed:>10.1f} {tflops:>12.2f}")
            results.append((name, size, iters, elapsed, tflops))
    overall_elapsed = time.time() - overall_start

    print()
    print("======================================")
    print("Benchmark finished")
    print("======================================")
    print(f"Total elapsed : {overall_elapsed:.1f} s")
    print(f"Configs run   : {n_configs}")
    best = max(results, key=lambda r: r[4])
    print(f"Peak TFLOPs   : {best[4]:.2f} ({best[0]}, size={best[1]})")

if __name__ == "__main__":
    main()
