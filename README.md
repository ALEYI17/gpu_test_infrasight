# gpu_test_infrasight


# Benchmark readme

# eBPF CUDA Tracer — Overhead Benchmark

Measures the performance overhead added by the eBPF kprobes and tracepoints to
each workload. Produces a wall-time delta table, a per-probe nanosecond breakdown,
and an eBPF-time-as-%-of-runtime column for every experiment.

---

## Files

| File | Purpose |
|---|---|
| `measure_overhead.py` | Main benchmark script — drop next to your runner |
| `run_benchmark.sh` | Convenience wrapper that runs all four phases |

---

## Setup

**1. Edit `CONFIG["experiments"]`** in `measure_overhead.py` to uncomment the same
workloads you have active in your main runner. The tuple format is identical:

```python
"experiments": [
    ("other", "miners/xmrig",    "miner_xmrig"),
    ("dl_ml", "dl/cnn/train.py", "dl_cnn_train"),
    ("llm",   "llm/bert",        "llm_bert"),
    # ...
],
```

**2. Check the loader command** in `CONFIG["loader"]`. It is pre-set to:

```
./main
  --tracer=fingerprint
  --server-addr=localhost
  --server-port=8080
  --cuda-lib=/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so
  --time-window=2
```

No `sudo` prefix is needed — the benchmark itself is run as root, so all child
processes inherit root automatically. A nested `sudo` would fail because there is
no TTY available.

---

## How It Measures Overhead

### 1. Wall-clock delta (end-to-end)
Each workload runs N times without probes (baseline) and N times with probes
(instrumented). Reports mean ± stdev and the delta in both milliseconds and percent.

### 2. In-kernel BPF stats — per-probe nanoseconds (most precise)
Enabled via `kernel.bpf_stats_enabled=1` (set automatically when run as root).
The kernel accumulates `run_cnt` and `run_time_ns` per BPF program. The script
snapshots these before and after every workload iteration:

```
probe_overhead_ns = run_time_ns_after  − run_time_ns_before
probe_overhead_%  = probe_overhead_ns  / wall_time_ns × 100
```

This is pure CPU time inside the JIT'd eBPF code — independent of wall-clock noise.

### 3. `bpftool prog profile` — hardware PMU counters (optional)
The `probe-profile` subcommand uses hardware perf counters per probe:
cycles, instructions, L1d loads, LLC misses. Useful for understanding *why* a
probe is slow, not just *how much* time it takes.

### 4. GPU metrics
`nvidia-smi` is polled every 200 ms during each run to verify probes don't
perturb GPU utilisation, VRAM, power draw, or SM clock frequency.

---

## Tracked Programs

### kprobes (CUDA driver)
| Program | Hook |
|---|---|
| `handle_cuCtxSync` | `cuCtxSynchronize` entry |
| `handle_cuCtxSync_ret` | `cuCtxSynchronize` return |
| `handle_cuLaunchkernel` | `cuLaunchKernel` entry |
| `handle_cuMemAlloc` | `cuMemAlloc` entry |
| `handle_cuMemcpy_dtoh` | `cuMemcpyDtoH` entry |
| `handle_cuMemcpy_dtohAsync` | `cuMemcpyDtoHAsync` entry |
| `handle_cuMemcpy_htod` | `cuMemcpyHtoD` entry |
| `handle_cuMemcpy_htod_async` | `cuMemcpyHtoDAsync` entry |
| `handle_cuStreamSync` | `cuStreamSynchronize` entry |
| `handle_cuStreamSynchronize_ret` | `cuStreamSynchronize` return |

### tracepoints
| Program | Tracepoint |
|---|---|
| `watchdog_ioctl` | `syscalls/sys_enter_ioctl` |
| `handle_process_exit` | `sched/sched_process_exit` |

Names match the ELF symbol names reported by `bpftool prog show`, not the
bpf2go-generated Go struct field names.

---

## Usage

### Quick start

```bash
sudo bash run_benchmark.sh
```

### Manual — step by step

```bash
# Phase 1 — baseline (loader must NOT be running)
sudo python3 measure_overhead.py run --mode baseline

# Phase 2 — instrumented (loader started/stopped automatically)
sudo python3 measure_overhead.py run --mode instrumented

# Phase 3 — report
python3 measure_overhead.py report

# Phase 4 — per-probe PMU profiling for one experiment (optional)
sudo python3 measure_overhead.py probe-profile --name dl_cnn_train --duration 60
```

### Filtering

```bash
# Run only specific experiments
sudo python3 measure_overhead.py run --mode baseline --only miner_xmrig dl_cnn_train

# Run only one category
sudo python3 measure_overhead.py run --mode instrumented --categories other

# Mix filters
sudo python3 measure_overhead.py run --mode baseline --categories dl_ml llm
```

### All flags

| Flag | Default | Applies to | Description |
|---|---|---|---|
| `--mode` | required | `run` | `baseline` or `instrumented` |
| `--iterations` | `5` | `run` | Runs per workload |
| `--loader` | see CONFIG | `run`, `probe-profile` | Full loader command string |
| `--loader-wait` | `2.0` | `run`, `probe-profile` | Seconds to wait after loader starts |
| `--only` | all | `run`, `probe-profile` | Space-separated experiment names |
| `--categories` | all | `run`, `probe-profile` | `dl_ml`, `llm`, and/or `other` |
| `--results-dir` | `./results/overhead` | all | Output directory |
| `--name` | required | `probe-profile` | Experiment name from CONFIG |
| `--duration` | `30` | `probe-profile` | Profiling window in seconds |

---

## Example Report Output

```
════════════════════════════════════════════════════════════════════════════════════════════
  eBPF CUDA Tracer — Overhead Report   2026-05-21 20:55:01
════════════════════════════════════════════════════════════════════════════════════════════

── Wall-Time Overhead ──────────────────────────────────────────────────────────────────
Workload                       Baseline (s)   Instrumented (s)      Δ (ms)      Δ (%)
──────────────────────────────────────────────────────────────────────────────────────
dl_cnn_train              45.231±0.121    45.247±0.134         +16.0      +0.04%
miner_xmrig               12.344±0.089    12.349±0.091          +5.1      +0.04%
llm_bert                 183.412±0.821   183.449±0.856         +37.2      +0.02%

── Per-Probe Execution Cost (mean across all instrumented workloads) ───────────────────
Probe                                Calls/run    Avg (ns)   Total (µs)   Total (ms)
────────────────────────────────────────────────────────────────────────────────────
handle_cuCtxSync                           892        48.2        43014       0.0430
handle_cuCtxSync_ret                       892       163.4       145803       0.1458
handle_cuLaunchkernel                    18432       220.1      4054579       4.0546
handle_cuMemAlloc                           12        89.3         1072       0.0011
handle_cuMemcpy_dtoh                      3421       135.2       462519       0.4625
handle_cuMemcpy_dtohAsync                 1205       131.8       158769       0.1588
handle_cuMemcpy_htod                      3421       137.6       470780       0.4708
handle_cuMemcpy_htod_async               1205       133.9       161360       0.1614
handle_cuStreamSync                        412        48.2        19859       0.0199
handle_cuStreamSynchronize_ret             412       163.4        67321       0.0673
handle_process_exit                      12043        41.3       497375       0.4974
watchdog_ioctl                           48210       198.7      9579327       9.5793
────────────────────────────────────────────────────────────────────────────────────
TOTAL                                                          15661778      15.6618

── eBPF Time as % of Wall-Clock Runtime ────────────────────────────────────────────────
Workload                      eBPF (ms)   Wall (s)     eBPF %
────────────────────────────────────────────────────────────
dl_cnn_train                    15.6618    45.247     0.0346%
miner_xmrig                     15.6618    12.349     0.1268%
llm_bert                        15.6618   183.449     0.0085%
```

---

## Output Files

| File | Contents |
|---|---|
| `<name>_baseline.json` | Per-iteration wall times and GPU metrics |
| `<name>_instrumented.json` | Per-iteration wall times + per-probe BPF stats + aggregates |
| `overhead_report.json` | Combined machine-readable report for both modes |
| `probe_profiles.json` | PMU counter output per probe (only if `probe-profile` was run) |

---

## Tips

- **Run as root** — required for `bpf_stats_enabled` sysctl and `bpftool`
- **Loader scope** — the loader starts once and stays up for the entire batch. If your
  loader needs a restart between workloads, move `LoaderContext` inside the per-experiment
  loop in `cmd_run`
- **High variance?** — increase `--iterations` or pin the CPU governor:
  `cpupower frequency-set -g performance`
- **Miners** — use benchmark/fixed-work modes (`--benchmark`) rather than timed runs
  so the amount of work is constant regardless of overhead
- **Confirm probes attached** — `LoaderContext` calls `bpftool prog show` after
  `--loader-wait` seconds and prints how many of the 12 programs are visible; if it
  shows 0, check `--cuda-lib` path and loader stderr output
