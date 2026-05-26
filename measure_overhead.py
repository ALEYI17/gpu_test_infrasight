#!/usr/bin/env python3
"""
eBPF CUDA Tracer — Overhead Benchmark
======================================
Drop this file next to your existing runner script.
It reuses your exact CONFIG["experiments"] format and runner functions so there is
no duplication of experiment definitions or execution logic.

Usage
-----
  # 1. Baseline  — make sure your eBPF loader is NOT running
  sudo python3 measure_overhead.py run --mode baseline

  # 2. Instrumented — script starts/stops your loader around each workload
  #    Pass the full loader command (already running as root via sudo, no nested sudo needed)
  sudo python3 measure_overhead.py run --mode instrumented \
      --loader "./main --tracer=fingerprint --server-addr=localhost --server-port=8080 \
                --cuda-lib=/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so"

  # 3. Report — compare the two and show overhead table
  python3 measure_overhead.py report

  # Filter to specific experiments or categories:
  sudo python3 measure_overhead.py run --mode baseline --only miner_xmrig dl_cnn_train
  sudo python3 measure_overhead.py run --mode baseline --categories other

  # Per-probe perf-counter profiling (while a workload runs):
  sudo python3 measure_overhead.py probe-profile --name dl_cnn_train --duration 60
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors your runner's CONFIG — edit the same way you edit that file)
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "clickhouse_container": "clickhouse",
    "clickhouse_client_args": "",
    "dataset_dir": "dataset",
    "tables": [
        "audit.gpu_time_window_events",
        "audit.gpu_event_tokens",
    ],
    # ── Same format as your runner: (kind, path, exp_name) ──────────────────
    # Uncomment / comment exactly the same experiments you want to benchmark.
    "experiments": [
        #("other", "passwd_cracker/hashcat", "passwd_hashcat"),
        #("other", "miners/xmrig",           "miner_xmrig"),
        #("other", "miners/lolminer",         "miner_lolminer"),
        #("other", "blender/",               "blender"),
        #("other", "miners/nbminer",          "miner_nbminer"),
        #("other", "miners/gminer",           "miner_gminer"),
        #("other", "miners/bzminer",          "miner_bzminer"),
        #("other", "miners/srbminer",         "miner_srbminer"),
        #("other", "miners/t-rex",            "miner_trex"),
        # ("dl_ml", "dl/cnn/train.py",         "dl_cnn_train"),
        # ("dl_ml", "dl/lstm/train.py",        "dl_lstm_train"),
        # ("llm",   "llm/bert",                "llm_bert"),
        # ("llm",   "llm/bloom",               "llm_bloom"),
        # ("llm",   "llm/gpt",                 "llm_gpt"),
        # ("llm",   "llm/gpt-neo",             "llm_gpt_neo"),
         ("llm",   "llm/roberta",             "llm_roberta"),
        # ("dl_ml", "ml/logistic_regression/train.py", "ml_logreg"),
        # ("dl_ml", "ml/random_forest/train.py",       "ml_forest"),
        # ("dl_ml", "ml/svm/train.py",                 "ml_svm"),
    ],
    # ── Benchmark-specific settings ─────────────────────────────────────────
    "iterations":    5,          # runs per workload per mode
    # Full loader command WITHOUT --time-window (injected automatically per run).
    # No 'sudo' prefix needed: this script is already run as root via sudo,
    # so every subprocess it spawns inherits root — nested sudo would fail.
    "loader": (
        "./main"
        " --tracer=fingerprint"
        " --server-addr=localhost"
        " --server-port=8080"
        " --cuda-lib=/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so"
    ),
    "loader_wait":   2.0,        # seconds to wait after loader starts before running workload
    # Time windows (seconds) to collect separate datasets for.
    # Each window restarts the loader with --time-window=N and saves to
    # dataset/<exp_name>/tw<N>/<timestamp>/.
    # Only used when --collect is passed in instrumented mode.
    # Set to [] or omit --time-windows flag to use a single default window.
    "time_windows":  [1, 2, 5],
    "results_dir":   "./results/overhead",
    "stop_on_error": False,
}

REPO_ROOT = Path(__file__).resolve().parent
BASE_ENV  = REPO_ROOT / ".env"

# ──────────────────────────────────────────────────────────────────────────────
# CLICKHOUSE DATA COLLECTION  (from your runner — used when --collect is passed)
# ──────────────────────────────────────────────────────────────────────────────

def ch_query_truncate(container: str, extra_args: str = "") -> None:
    """Truncate all tracked ClickHouse tables before each instrumented run."""
    for table in CONFIG["tables"]:
        q   = f"TRUNCATE TABLE {table}"
        cmd = (f"docker exec -i {shlex.quote(container)} "
               f"clickhouse-client {extra_args} --query \"{q}\"")
        run_cmd_live(cmd)


def ch_export_table_parquet(container: str, table: str,
                             out_path: Path, extra_args: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q   = f"SELECT * FROM {table} FORMAT Parquet"
    cmd = (f"docker exec -i {shlex.quote(container)} "
           f"clickhouse-client {extra_args} --query \"{q}\"")
    print(f"  [collect] Exporting {table} -> {out_path}")
    with open(out_path, "wb") as f:
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"  [collect] ClickHouse export failed: "
                  f"{stderr.decode(errors='ignore')}")
            raise subprocess.CalledProcessError(p.returncode, cmd)
        f.write(stdout)


def ch_export_all(container: str, out_dir: Path, extra_args: str = "") -> None:
    for table in CONFIG["tables"]:
        safe = table.replace(".", "_")
        ch_export_table_parquet(container, table,
                                out_dir / f"{safe}.parquet", extra_args)


def ch_save_metadata(out_dir: Path, exp_name: str, path: str,
                     iteration: int, overhead_result: Dict,
                     time_window: Optional[int] = None) -> None:
    meta = {
        "experiment":    exp_name,
        "script_or_path": path,
        "iteration":     iteration,
        "time_window_s": time_window,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overhead": {
            "wall_time_s":  overhead_result["wall_time_s"],
            "exit_code":    overhead_result["exit_code"],
            "bpf_total_ns": sum(
                v["total_ns"] for v in overhead_result.get("bpf_delta", {}).values()
            ),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

# ──────────────────────────────────────────────────────────────────────────────
# RUNNER FUNCTIONS  (verbatim copy from your runner so they stay in sync)
# ──────────────────────────────────────────────────────────────────────────────

def run_cmd_live(cmd, cwd=None):
    """Run shell command streaming output (raises on non-zero)."""
    print(f"  > {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)


def build_runner_command(script_path: Path) -> str:
    script_parent = str(script_path.parent)
    script_name   = str(script_path.name)

    if BASE_ENV.exists():
        if BASE_ENV.is_dir() and (BASE_ENV / "bin" / "activate").exists():
            source_cmd = f"source {shlex.quote(str(BASE_ENV / 'bin' / 'activate'))}"
        elif BASE_ENV.is_file():
            source_cmd = f"source {shlex.quote(str(BASE_ENV))}"
        else:
            source_cmd = ""
    else:
        source_cmd = ""

    # Always use plain python3 — if a virtualenv was sourced above, its python3
    # is already first on PATH. We never use 'uv run' here because sudo strips
    # the user PATH so uv is not reachable even if installed for that user.
    run_python_cmd = f"python3 {shlex.quote(script_name)}"

    if source_cmd:
        return f"bash -lc 'cd {shlex.quote(script_parent)} && {source_cmd} && {run_python_cmd}'"
    return f"bash -lc 'cd {shlex.quote(script_parent)} && {run_python_cmd}'"


def run_experiment_llm(script_dir, exp_name):
    print(f"  [llm] {exp_name}")
    train_script = Path(script_dir) / "train.py"
    infer_script = Path(script_dir) / "infer.py"
    if not train_script.exists():
        raise FileNotFoundError(train_script)
    run_cmd_live(build_runner_command(train_script))
    if infer_script.exists():
        run_cmd_live(build_runner_command(infer_script))
    else:
        print("  No infer.py found; skipping inference.")


def run_experiment_dl_ml(script_path, exp_name):
    print(f"  [dl_ml] {exp_name}")
    script = Path(script_path)
    if script.is_dir():
        script = script / "train.py"
    if not script.exists():
        raise FileNotFoundError(script)
    run_cmd_live(build_runner_command(script))


def run_experiment_other(path, exp_name):
    print(f"  [other] {exp_name}")
    p = Path(path)

    def exec_sh(script_path: Path):
        cmd = f"bash -lc 'cd {shlex.quote(str(script_path.parent))} && bash {shlex.quote(str(script_path.name))}'"
        run_cmd_live(cmd)

    if p.is_file():
        if p.suffix == ".sh":
            exec_sh(p)
            return
        raise FileNotFoundError(f"{p} is not a .sh script")
    elif p.is_dir():
        preferred = ["run.sh", "run_hashcat.sh", "run_hashcat_minimal.sh"]
        for name in preferred:
            candidate = p / name
            if candidate.exists():
                exec_sh(candidate)
                return
        sh_files = sorted(f for f in p.glob("*.sh") if f.is_file())
        if sh_files:
            exec_sh(sh_files[0])
            return
        raise FileNotFoundError(f"No .sh entrypoint found in {p}. Expected one of: {preferred}")
    else:
        raise FileNotFoundError(f"Path {p} does not exist")


def dispatch_experiment(kind: str, path: str, exp_name: str):
    """Call the right runner for a (kind, path, exp_name) triple."""
    if kind == "llm":
        run_experiment_llm(path, exp_name)
    elif kind == "dl_ml":
        run_experiment_dl_ml(path, exp_name)
    elif kind == "other":
        run_experiment_other(path, exp_name)
    else:
        raise ValueError(f"Unknown experiment kind: {kind!r}")


# ──────────────────────────────────────────────────────────────────────────────
# BPF STATS
# ──────────────────────────────────────────────────────────────────────────────

BPF_STATS_SYSCTL = "/proc/sys/kernel/bpf_stats_enabled"

PROBE_NAMES = {
    # kprobes — CUDA driver calls
    "handle_cuCtxSync",
    "handle_cuCtxSync_ret",
    "handle_cuLaunchkernel",
    "handle_cuMemAlloc",
    "handle_cuMemcpy_dtoh",
    "handle_cuMemcpy_dtohAsync",
    "handle_cuMemcpy_htod",
    "handle_cuMemcpy_htod_async",
    "handle_cuStreamSync",
    "handle_cuStreamSynchronize_ret",
    # tracepoints
    "watchdog_ioctl",       # syscalls/sys_enter_ioctl
    "handle_process_exit",  # sched/sched_process_exit
}


def enable_bpf_stats() -> bool:
    """Enable kernel BPF stats (needs root). Returns True if already enabled."""
    try:
        prev = Path(BPF_STATS_SYSCTL).read_text().strip()
        Path(BPF_STATS_SYSCTL).write_text("1\n")
        print(f"[bpf_stats] Enabled (was {prev})")
        return True
    except PermissionError:
        print("[bpf_stats] WARNING: Cannot set sysctl — run as root for per-probe timing.")
        return False
    except FileNotFoundError:
        print("[bpf_stats] WARNING: sysctl not found (kernel too old?)")
        return False


def snapshot_bpf_stats() -> Dict[str, Dict]:
    """Snapshot run_cnt / run_time_ns for every tracked probe via bpftool."""
    try:
        r = subprocess.run(
            ["bpftool", "prog", "show", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return {}
        return {
            p["name"]: {
                "run_cnt":     p.get("run_cnt", 0),
                "run_time_ns": p.get("run_time_ns", 0),
                "id":          p.get("id"),
            }
            for p in json.loads(r.stdout)
            if p.get("name") in PROBE_NAMES
        }
    except Exception as e:
        print(f"[bpf_stats] bpftool error: {e}")
        return {}


def delta_bpf_stats(before: Dict, after: Dict) -> Dict:
    delta = {}
    for name in PROBE_NAMES:
        b = before.get(name, {"run_cnt": 0, "run_time_ns": 0})
        a = after.get(name, {"run_cnt": 0, "run_time_ns": 0})
        calls = a["run_cnt"]     - b["run_cnt"]
        ns    = a["run_time_ns"] - b["run_time_ns"]
        delta[name] = {
            "calls":    calls,
            "total_ns": ns,
            "avg_ns":   (ns / calls) if calls > 0 else 0,
        }
    return delta


# ──────────────────────────────────────────────────────────────────────────────
# GPU MONITOR
# ──────────────────────────────────────────────────────────────────────────────

class GpuMonitor:
    def __init__(self, interval_ms: int = 200):
        self._samples: List[Dict] = []
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._poll, args=(interval_ms,), daemon=True)

    def start(self): self._thread.start()

    def stop(self) -> List[Dict]:
        self._stop.set()
        self._thread.join(timeout=5)
        return self._samples

    def _poll(self, interval_ms):
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    break
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) == 5:
                    try:
                        self._samples.append({
                            "gpu_util":     float(parts[0]),
                            "mem_util":     float(parts[1]),
                            "mem_used_mb":  float(parts[2]),
                            "power_w":      float(parts[3]),
                            "sm_clock_mhz": float(parts[4]),
                        })
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass  # nvidia-smi not available


def summarize_gpu(samples: List[Dict]) -> Dict:
    if not samples:
        return {}
    return {k: {"mean": statistics.mean(s[k] for s in samples),
                "max":  max(s[k] for s in samples),
                "min":  min(s[k] for s in samples)}
            for k in samples[0]}


# ──────────────────────────────────────────────────────────────────────────────
# TIMED EXPERIMENT WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

def run_timed(kind: str, path: str, exp_name: str, bpf_stats_ok: bool) -> Dict:
    """
    Run one experiment iteration, collecting wall time, BPF stats delta, and GPU metrics.
    """
    gpu = GpuMonitor()
    gpu.start()

    bpf_before = snapshot_bpf_stats() if bpf_stats_ok else {}
    t0 = time.perf_counter()

    exit_code = 0
    error_msg = ""
    try:
        dispatch_experiment(kind, path, exp_name)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        error_msg = str(e)
    except Exception as e:
        exit_code = -1
        error_msg = str(e)

    wall = time.perf_counter() - t0
    bpf_after   = snapshot_bpf_stats() if bpf_stats_ok else {}
    gpu_samples = gpu.stop()

    return {
        "wall_time_s": wall,
        "exit_code":   exit_code,
        "error":       error_msg,
        "bpf_delta":   delta_bpf_stats(bpf_before, bpf_after) if bpf_before else {},
        "gpu":         summarize_gpu(gpu_samples),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LOADER CONTEXT MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class LoaderContext:
    """
    Start the eBPF loader process before the block, stop it after.

    No 'sudo' prefix is needed in the command: this benchmark is already
    invoked as root (sudo python3 measure_overhead.py ...), so all child
    processes inherit root automatically. A nested sudo would fail because
    there is no TTY available for password prompting.
    """

    def __init__(self, loader_cmd: Optional[str], wait: float = 2.0):
        self.loader_cmd   = loader_cmd
        self.wait         = wait
        self._proc        = None
        self._output_lines: List[str] = []

    def __enter__(self):
        if not self.loader_cmd:
            return self

        print(f"[loader] Starting: {self.loader_cmd}")
        self._proc = subprocess.Popen(
            self.loader_cmd,
            shell=True,
            stdout=subprocess.PIPE,   # capture so we can show it on failure
            stderr=subprocess.STDOUT, # merge stderr into stdout
            text=True,
        )

        # Drain loader output in a background thread so the pipe never blocks
        def _drain():
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                self._output_lines.append(line.rstrip())
        threading.Thread(target=_drain, daemon=True).start()

        # Give the loader time to initialise, then make sure it didn't crash
        time.sleep(self.wait)
        rc = self._proc.poll()
        if rc is not None:
            output = "\n  ".join(self._output_lines[-20:]) or "(no output)"
            raise RuntimeError(
                f"[loader] Process exited unexpectedly (rc={rc}) after {self.wait}s.\n"
                f"  Last output:\n  {output}\n"
                f"  Command was: {self.loader_cmd}"
            )

        # Confirm probes actually attached via bpftool before running workloads
        snap = snapshot_bpf_stats()
        attached = [n for n in PROBE_NAMES if n in snap]
        if attached:
            print(f"[loader] Running (PID {self._proc.pid}) — "
                  f"{len(attached)}/{len(PROBE_NAMES)} probes confirmed attached")
        else:
            print(f"[loader] WARNING: PID {self._proc.pid} is alive but no probes "
                  f"visible in bpftool yet — check --cuda-lib path and loader output")

        return self

    def __exit__(self, *_):
        if self._proc:
            print(f"[loader] Stopping PID {self._proc.pid}")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            if self._output_lines:
                print("[loader] Final output:")
                for line in self._output_lines[-10:]:
                    print(f"  {line}")
            self._proc = None


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_iterations(iterations: List[Dict]) -> Dict:
    ok         = [it for it in iterations if it["exit_code"] == 0]
    wall_times = [it["wall_time_s"] for it in ok]

    bpf_agg: Dict[str, Dict[str, List]] = {
        n: {"calls": [], "total_ns": [], "avg_ns": []} for n in PROBE_NAMES
    }
    for it in ok:
        for probe, stats in it.get("bpf_delta", {}).items():
            bpf_agg[probe]["calls"].append(stats["calls"])
            bpf_agg[probe]["total_ns"].append(stats["total_ns"])
            if stats["calls"] > 0:
                bpf_agg[probe]["avg_ns"].append(stats["avg_ns"])

    bpf_summary = {
        probe: {
            "calls_mean":    statistics.mean(d["calls"])    if d["calls"]    else 0,
            "total_ns_mean": statistics.mean(d["total_ns"]) if d["total_ns"] else 0,
            "avg_ns_mean":   statistics.mean(d["avg_ns"])   if d["avg_ns"]   else 0,
        }
        for probe, d in bpf_agg.items()
    }

    return {
        "wall_mean_s":  statistics.mean(wall_times)   if wall_times          else None,
        "wall_stdev_s": statistics.stdev(wall_times)  if len(wall_times) > 1 else 0.0,
        "wall_min_s":   min(wall_times)                if wall_times          else None,
        "wall_max_s":   max(wall_times)                if wall_times          else None,
        "n_ok":         len(ok),
        "n_total":      len(iterations),
        "bpf":          bpf_summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SUBCOMMANDS
# ──────────────────────────────────────────────────────────────────────────────

def cmd_run(args):
    """Run all active experiments in one mode (baseline or instrumented)."""
    experiments = _filter_experiments(args)
    if not experiments:
        print("No experiments selected. "
              "Check CONFIG['experiments'] and --only / --categories filters.")
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    collect     = args.collect and args.mode == "instrumented"
    container   = CONFIG["clickhouse_container"]
    ch_extra    = CONFIG["clickhouse_client_args"]
    dataset_dir = Path(CONFIG["dataset_dir"])

    # Time windows: use CLI value if given, else CONFIG default, else [None] (single pass)
    if collect:
        time_windows = args.time_windows if args.time_windows else CONFIG.get("time_windows", [])
        if not time_windows:
            time_windows = [None]   # single pass, no --time-window injection
        dataset_dir.mkdir(exist_ok=True)
        print(f"[collect] Data collection ON  container={container!r}  "
              f"time_windows={time_windows}")
    else:
        time_windows = [None]       # overhead-only: one pass, no window looping
        if args.collect and args.mode == "baseline":
            print("[collect] NOTE: --collect ignored in baseline mode (no probes -> no data)")

    bpf_stats_ok = False
    if args.mode == "instrumented":
        bpf_stats_ok = enable_bpf_stats()

    base_loader_cmd = args.loader if args.mode == "instrumented" else None

    for tw in time_windows:
        # Build loader command for this time window
        if base_loader_cmd and tw is not None:
            loader_cmd = f"{base_loader_cmd} --time-window={tw}"
            tw_label   = f"tw{tw}"
        else:
            loader_cmd = base_loader_cmd
            tw_label   = None

        if tw is not None:
            print(f"\n{'*'*60}")
            print(f"  TIME WINDOW: {tw}s")
            print(f"{'*'*60}")

        with LoaderContext(loader_cmd, wait=args.loader_wait):
            for kind, path, exp_name in experiments:
                print(f"\n{'='*60}")
                print(f"  {exp_name}  [{kind}]  mode={args.mode}"
                      + (f"  tw={tw}s" if tw is not None else ""))
                print(f"{'='*60}")

                iterations = []
                for i in range(args.iterations):
                    print(f"\n  -- iteration {i+1}/{args.iterations} --")

                    # pre-run: truncate ClickHouse so each iteration is clean
                    if collect:
                        print("  [collect] Truncating ClickHouse tables...")
                        try:
                            ch_query_truncate(container, ch_extra)
                        except Exception as e:
                            print(f"  [collect] WARNING: truncate failed: {e}")

                    result = run_timed(kind, path, exp_name, bpf_stats_ok)
                    result["iteration"]    = i + 1
                    result["time_window"]  = tw
                    iterations.append(result)

                    probe_ns = sum(v["total_ns"] for v in result["bpf_delta"].values()) \
                               if result["bpf_delta"] else 0
                    pct = (probe_ns / 1e9) / result["wall_time_s"] * 100 \
                          if result["wall_time_s"] > 0 and probe_ns else 0
                    print(f"  wall={result['wall_time_s']:.3f}s  exit={result['exit_code']}"
                          + (f"  probe_total={probe_ns/1e6:.3f}ms ({pct:.4f}%)" if probe_ns else ""))

                    # post-run: export tables + metadata into tw-specific subdir
                    if collect and result["exit_code"] == 0:
                        ts      = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        # dataset/<exp_name>/tw<N>/<timestamp>/  (or /<timestamp>/ if no window)
                        run_dir = (dataset_dir / exp_name / tw_label / ts
                                   if tw_label else dataset_dir / exp_name / ts)
                        run_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            ch_export_all(container, run_dir, ch_extra)
                            ch_save_metadata(run_dir, exp_name, path, i + 1, result, tw)
                            print(f"  [collect] Saved -> {run_dir}")
                        except Exception as e:
                            print(f"  [collect] WARNING: export failed: {e}")

                    if result["exit_code"] != 0 and CONFIG["stop_on_error"]:
                        print(f"  ERROR: {result['error']}")
                        break

                agg = aggregate_iterations(iterations)
                # overhead results are per time-window so they don't overwrite each other
                tw_suffix = f"_tw{tw}" if tw is not None and collect else ""
                record = {
                    "name":        exp_name,
                    "kind":        kind,
                    "path":        path,
                    "mode":        args.mode,
                    "time_window": tw,
                    "timestamp":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "iterations":  iterations,
                    "aggregate":   agg,
                }

                out = results_dir / f"{exp_name}{tw_suffix}_{args.mode}.json"
                out.write_text(json.dumps(record, indent=2))

                print(f"\n  [OK] Overhead saved -> {out}")
                if agg["wall_mean_s"] is not None:
                    print(f"      wall = {agg['wall_mean_s']:.3f}s "
                          f"+/- {agg['wall_stdev_s']:.3f}s  "
                          f"(n={agg['n_ok']}/{agg['n_total']})")

def cmd_report(args):
    """Print overhead comparison table from saved baseline + instrumented results."""
    results_dir = Path(args.results_dir)

    baselines:    Dict[str, Dict] = {}
    instrumented: Dict[str, Dict] = {}

    for f in results_dir.glob("*_baseline.json"):
        d = json.loads(f.read_text())
        baselines[d["name"]] = d

    for f in results_dir.glob("*_instrumented.json"):
        d = json.loads(f.read_text())
        instrumented[d["name"]] = d

    if not baselines and not instrumented:
        print(f"No result files found in {results_dir}")
        return

    all_names = sorted(set(list(baselines) + list(instrumented)))

    print("\n" + "═"*92)
    print("  eBPF CUDA Tracer — Overhead Report")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═"*92)

    # ── Wall-time table ──────────────────────────────────────────────────────
    print("\n── Wall-Time Overhead ──────────────────────────────────────────────────────────────")
    print(f"{'Workload':<28} {'Baseline (s)':>16} {'Instrumented (s)':>18} {'Δ (ms)':>10} {'Δ (%)':>10}")
    print("─"*86)

    for name in all_names:
        ba = baselines.get(name, {}).get("aggregate", {})
        ia = instrumented.get(name, {}).get("aggregate", {})
        bm = ba.get("wall_mean_s")
        im = ia.get("wall_mean_s")

        if bm is not None and im is not None:
            d_ms  = (im - bm) * 1000
            d_pct = (im - bm) / bm * 100
            b_str = f"{bm:.3f}±{ba.get('wall_stdev_s', 0):.3f}"
            i_str = f"{im:.3f}±{ia.get('wall_stdev_s', 0):.3f}"
            print(f"{name:<28} {b_str:>16} {i_str:>18} {d_ms:>+9.1f}  {d_pct:>+8.2f}%")
        elif bm is not None:
            print(f"{name:<28} {bm:>16.3f} {'—':>18} {'—':>10} {'—':>10}")
        elif im is not None:
            print(f"{name:<28} {'—':>16} {im:>18.3f} {'—':>10} {'—':>10}")

    # ── Per-probe table ──────────────────────────────────────────────────────
    print("\n\n── Per-Probe Execution Cost (mean across all instrumented workloads) ───────────────")
    print(f"{'Probe':<36} {'Calls/run':>11} {'Avg (ns)':>10} {'Total (µs)':>12} {'Total (ms)':>12}")
    print("─"*84)

    probe_calls:    Dict[str, List[float]] = {n: [] for n in PROBE_NAMES}
    probe_avg_ns:   Dict[str, List[float]] = {n: [] for n in PROBE_NAMES}
    probe_total_ns: Dict[str, List[float]] = {n: [] for n in PROBE_NAMES}

    for data in instrumented.values():
        for probe, stats in data.get("aggregate", {}).get("bpf", {}).items():
            probe_calls[probe].append(stats["calls_mean"])
            probe_total_ns[probe].append(stats["total_ns_mean"])
            if stats["avg_ns_mean"] > 0:
                probe_avg_ns[probe].append(stats["avg_ns_mean"])

    grand_total_ns = 0.0
    for probe in sorted(PROBE_NAMES):
        calls  = statistics.mean(probe_calls[probe])    if probe_calls[probe]    else 0
        avg_ns = statistics.mean(probe_avg_ns[probe])   if probe_avg_ns[probe]   else 0
        tot_ns = statistics.mean(probe_total_ns[probe]) if probe_total_ns[probe] else 0
        grand_total_ns += tot_ns
        print(f"{probe:<36} {calls:>11.0f} {avg_ns:>10.1f} {tot_ns/1e3:>12.2f} {tot_ns/1e6:>12.4f}")

    print("─"*84)
    print(f"{'TOTAL':<36} {'':>11} {'':>10} "
          f"{grand_total_ns/1e3:>12.2f} {grand_total_ns/1e6:>12.4f}")

    # ── eBPF % of wall time ──────────────────────────────────────────────────
    print("\n\n── eBPF Time as % of Wall-Clock Runtime ────────────────────────────────────────────")
    print(f"{'Workload':<28} {'eBPF (ms)':>12} {'Wall (s)':>10} {'eBPF %':>10}")
    print("─"*64)
    for name in all_names:
        if name not in instrumented:
            continue
        ia = instrumented[name].get("aggregate", {})
        im = ia.get("wall_mean_s")
        if im is None:
            continue
        ebpf_ns = sum(v["total_ns_mean"] for v in ia.get("bpf", {}).values())
        frac    = (ebpf_ns / 1e9) / im * 100 if im else 0
        print(f"{name:<28} {ebpf_ns/1e6:>12.4f} {im:>10.3f} {frac:>9.4f}%")

    # ── GPU metrics comparison ────────────────────────────────────────────────
    # GPU data is stored per-iteration; aggregate it here from raw iterations.
    def _avg_gpu_across_iterations(data: Dict) -> Dict[str, float]:
        """Mean of per-iteration GPU means for each metric."""
        acc: Dict[str, List[float]] = {}
        for it in data.get("iterations", []):
            if it.get("exit_code") != 0:
                continue
            for metric, vals in it.get("gpu", {}).items():
                acc.setdefault(metric, []).append(vals["mean"])
        return {k: statistics.mean(v) for k, v in acc.items() if v}

    gpu_metrics = {
        "gpu_util":     ("GPU util (%)",    "{:>8.1f}"),
        "mem_util":     ("Mem BW util (%)", "{:>8.1f}"),
        "mem_used_mb":  ("VRAM used (MB)",  "{:>8.0f}"),
        "power_w":      ("Power (W)",       "{:>8.1f}"),
        "sm_clock_mhz": ("SM clock (MHz)",  "{:>8.0f}"),
    }

    # Collect GPU data for workloads that have it in both modes
    gpu_rows = []
    for name in all_names:
        b_gpu = _avg_gpu_across_iterations(baselines.get(name, {}))
        i_gpu = _avg_gpu_across_iterations(instrumented.get(name, {}))
        if b_gpu or i_gpu:
            gpu_rows.append((name, b_gpu, i_gpu))

    if gpu_rows:
        print("\n\n── GPU Metrics (mean across iterations) ────────────────────────────────────────────")
        col_w = 18
        header_label = f"{'Workload':<28} {'Mode':<14}"
        for _, (label, _) in gpu_metrics.items():
            header_label += f" {label:>{col_w}}"
        print(header_label)
        print("─" * (28 + 14 + col_w * len(gpu_metrics) + len(gpu_metrics)))

        for name, b_gpu, i_gpu in gpu_rows:
            for mode_label, gpu_data in [("baseline", b_gpu), ("instrumented", i_gpu)]:
                if not gpu_data:
                    continue
                row = f"{name:<28} {mode_label:<14}"
                for key, (_, fmt) in gpu_metrics.items():
                    val = gpu_data.get(key)
                    row += f" {(fmt.format(val) if val is not None else '—'):>{col_w}}"
                print(row)
            print()  # blank line between workloads
    else:
        print("\n\n── GPU Metrics ─────────────────────────────────────────────────────────────────────")
        print("  No GPU samples recorded. nvidia-smi may not be available on this host.")

    print("\n" + "═"*92)

    # Save machine-readable report
    report_path = results_dir / "overhead_report.json"
    payload = json.dumps({
        "generated":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baselines":    baselines,
        "instrumented": instrumented,
    }, indent=2)
    try:
        report_path.write_text(payload)
        print(f"[✓] Full report → {report_path}")
    except PermissionError:
        print(f"[!] Cannot write {report_path} — results directory is owned by root.")
        print(f"    Either run with sudo, or fix permissions once:")
        print(f"      sudo chown -R $USER {results_dir}")
        print(f"    The report above is complete; re-run 'report' to also save the JSON.")


def cmd_probe_profile(args):
    """
    Profile individual probes with hardware PMU counters via
    'bpftool prog profile' while a named experiment runs.
    The loader is started automatically via --loader.
    """
    exp = next(
        ((k, p, n) for k, p, n in CONFIG["experiments"] if n == args.name),
        None,
    )
    if exp is None:
        print(f"Experiment {args.name!r} not found in CONFIG['experiments'].")
        return

    _, path, exp_name = exp

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with LoaderContext(args.loader, wait=args.loader_wait):
        # Look up probe IDs AFTER the loader has started and probes are attached
        r = subprocess.run(["bpftool", "prog", "show", "--json"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("bpftool failed — are you root?")
            return

        progs = [p for p in json.loads(r.stdout) if p.get("name") in PROBE_NAMES]
        if not progs:
            print("No matching probes found after loader started — check --cuda-lib path.")
            return

        print(f"  Found {len(progs)} probes to profile: {[p['name'] for p in progs]}")

        profiles = {}
        for prog in progs:
            pid  = prog["id"]
            name = prog["name"]
            print(f"\n  Profiling {name} (id={pid}) for {args.duration}s ...")

            workload_proc = subprocess.Popen(
                build_runner_command(Path(path)),
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            profile_result = subprocess.run(
                ["bpftool", "prog", "profile", "id", str(pid),
                 "duration", str(args.duration),
                 "cycles", "instructions", "l1d_loads", "llc_misses"],
                capture_output=True, text=True, timeout=args.duration + 15,
            )
            workload_proc.terminate()
            out = (profile_result.stdout + profile_result.stderr).strip()
            profiles[name] = out
            print(f"    {out}")

    out_path = results_dir / "probe_profiles.json"
    out_path.write_text(json.dumps({
        "experiment": exp_name,
        "timestamp":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profiles":   profiles,
    }, indent=2))
    print(f"\n[✓] Probe profiles → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _filter_experiments(args) -> List:
    exps = list(CONFIG["experiments"])
    if getattr(args, "only", None):
        only = set(args.only)
        exps = [(k, p, n) for k, p, n in exps if n in only]
    if getattr(args, "categories", None):
        cats = set(args.categories)
        exps = [(k, p, n) for k, p, n in exps if k in cats]
    return exps


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="eBPF CUDA Tracer — Overhead Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--results-dir",  default=CONFIG["results_dir"])
        sp.add_argument("--loader",       default=CONFIG["loader"],
                        help="eBPF loader binary (instrumented mode only)")
        sp.add_argument("--loader-wait",  type=float, default=CONFIG["loader_wait"])
        sp.add_argument("--only",         nargs="+", metavar="EXP_NAME",
                        help="Run only these experiment names")
        sp.add_argument("--categories",   nargs="+", choices=["dl_ml", "llm", "other"],
                        help="Filter by category")

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run active experiments in baseline or instrumented mode")
    p_run.add_argument("--mode",         required=True, choices=["baseline", "instrumented"])
    p_run.add_argument("--iterations",   type=int, default=CONFIG["iterations"])
    p_run.add_argument("--collect",      action="store_true",
                       help="Truncate ClickHouse before each run and export Parquet after. "
                            "Only active in instrumented mode.")
    p_run.add_argument("--time-windows", type=int, nargs="+",
                       metavar="N",
                       default=None,
                       help="Time windows in seconds to collect separate datasets for "
                            "(e.g. --time-windows 2 5 10). Overrides CONFIG[\"time_windows\"]. "
                            "Only used with --collect in instrumented mode.")
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    # ── report ───────────────────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Print overhead comparison from saved results")
    p_rep.add_argument("--results-dir", default=CONFIG["results_dir"])
    p_rep.set_defaults(func=cmd_report)

    # ── probe-profile ─────────────────────────────────────────────────────────
    p_pp = sub.add_parser("probe-profile",
                          help="Per-probe PMU profiling via bpftool prog profile")
    p_pp.add_argument("--name",     required=True, help="Experiment name from CONFIG")
    p_pp.add_argument("--duration", type=int, default=30,
                      help="Profiling duration in seconds")
    add_common(p_pp)
    p_pp.set_defaults(func=cmd_probe_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
