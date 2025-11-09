#!/usr/bin/env python3
"""
Automated experiment-runner for infrasight stack using Dockerized ClickHouse.

Now sources a single base .env at repo root (either a file or a virtualenv dir).
"""

import os
import subprocess
import datetime
import json
import shlex
from pathlib import Path

# -------- CONFIGURE THIS --------
CONFIG = {
    "clickhouse_container": "clickhouse",
    "clickhouse_client_args": "",
    "dataset_dir": "dataset",
    "tables": [
        "audit.gpu_time_window_events",
        "audit.gpu_event_tokens",
    ],
    "experiments": [
        ("dl_ml", "dl/cnn/train.py", "dl_cnn_train"),
        ("dl_ml", "dl/lstm/train.py", "dl_lstm_train"),
        ("llm", "llm/bert", "llm_bert"),
        ("llm", "llm/gpt", "llm_gpt"),
        ("dl_ml", "ml/logistic_regression/train.py", "ml_logreg"),
        ("other", "passwd_cracker/hashcat", "passwd_hashcat"),
    ],
    "stop_on_error": True,
}
# --------------------------------

REPO_ROOT = Path(__file__).resolve().parent
BASE_ENV = REPO_ROOT / ".env"

def run_cmd_live(cmd, cwd=None):
    """Run shell command streaming output (raises on non-zero)."""
    print(f"> {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd)

def ch_query_truncate(container, extra_args=""):
    for table in CONFIG["tables"]:
        q = f"TRUNCATE TABLE {table}"
        cmd = f"docker exec -i {shlex.quote(container)} clickhouse-client {extra_args} --query \"{q}\""
        run_cmd_live(cmd)

def ch_export_table_parquet(container, table, out_path, extra_args=""):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q = f"SELECT * FROM {table} FORMAT Parquet"
    cmd = f"docker exec -i {shlex.quote(container)} clickhouse-client {extra_args} --query \"{q}\""
    print(f"Exporting {table} → {out_path}")
    with open(out_path, "wb") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print("ClickHouse export failed:", stderr.decode(errors="ignore"))
            raise subprocess.CalledProcessError(p.returncode, cmd)
        f.write(stdout)

def build_runner_command(script_path: Path):
    """
    Build a shell command that:
      - cds to script parent
      - sources repo-root .env (file or virtualenv dir/bin/activate) if present
      - runs uv run python3 <script_name> (or python3 if no env)
    Returns a string suitable for bash -lc execution.
    """
    script_parent = str(script_path.parent)
    script_name = str(script_path.name)

    # detect base env and how to source it
    if BASE_ENV.exists():
        if BASE_ENV.is_dir() and (BASE_ENV / "bin" / "activate").exists():
            # virtualenv dir: source bin/activate
            source_cmd = f"source {shlex.quote(str(BASE_ENV / 'bin' / 'activate'))}"
        elif BASE_ENV.is_file():
            # plain env file: source it
            source_cmd = f"source {shlex.quote(str(BASE_ENV))}"
        else:
            source_cmd = ""
    else:
        source_cmd = ""

    # prefer uv run if we sourced an env (user uses uv); otherwise fallback python3
    if source_cmd:
        run_python_cmd = f"uv run python3 {shlex.quote(script_name)}"
    else:
        run_python_cmd = f"python3 {shlex.quote(script_name)}"

    # combine into a single shell execution so sourcing affects the run
    # cd into script dir, then source env (if any), then run the command
    if source_cmd:
        cmd = f"bash -lc 'cd {shlex.quote(script_parent)} && {source_cmd} && {run_python_cmd}'"
    else:
        cmd = f"bash -lc 'cd {shlex.quote(script_parent)} && {run_python_cmd}'"
    return cmd

def run_experiment_llm(script_dir, exp_name):
    print(f"--- Running LLM experiment {exp_name} ---")
    train_script = Path(script_dir) / "train.py"
    infer_script = Path(script_dir) / "infer.py"
    if not train_script.exists():
        raise FileNotFoundError(train_script)
    run_cmd_live(build_runner_command(train_script))
    if infer_script.exists():
        run_cmd_live(build_runner_command(infer_script))
    else:
        print("No infer.py found; skipping inference.")

def run_experiment_dl_ml(script_path, exp_name):
    print(f"--- Running DL/ML experiment {exp_name} ---")
    script = Path(script_path)
    if script.is_dir():
        script = script / "train.py"
    if not script.exists():
        raise FileNotFoundError(script)
    run_cmd_live(build_runner_command(script))

def run_experiment_other(path, exp_name):
    print(f"--- Running OTHER experiment {exp_name} ---")
    p = Path(path)
    if p.is_file():
        if p.suffix == ".py":
            run_cmd_live(build_runner_command(p))
        else:
            # generic execution - cd into dir if file in a dir
            run_cmd_live(f"bash -lc 'cd {shlex.quote(str(p.parent))} && {shlex.quote(str(p))}'")
    elif (p / "run.sh").exists():
        run_cmd_live(f"bash -lc 'cd {shlex.quote(str(p))} && bash run.sh'")
    else:
        raise FileNotFoundError(f"No runnable entry in {p}")

def save_metadata(out_dir, script_ref, exp_name):
    meta = {
        "experiment": exp_name,
        "script_or_path": script_ref,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(Path(out_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def export_all_tables(container, out_dir, extra_args=""):
    for t in CONFIG["tables"]:
        safe_name = t.replace(".", "_")
        out_file = Path(out_dir) / f"{safe_name}.parquet"
        ch_export_table_parquet(container, t, out_file, extra_args)

def main():
    container = CONFIG["clickhouse_container"]
    extra_args = CONFIG["clickhouse_client_args"]
    base_dataset_dir = Path(CONFIG["dataset_dir"])
    base_dataset_dir.mkdir(exist_ok=True)

    print(f"Repo root: {REPO_ROOT}")
    print(f"Using base env: {BASE_ENV} (exists={BASE_ENV.exists()})")

    for kind, path, exp_name in CONFIG["experiments"]:
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_dir = base_dataset_dir / exp_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== START {exp_name} @ {timestamp} ===")

        try:
            print("Truncating ClickHouse tables...")
            ch_query_truncate(container, extra_args=extra_args)

            if kind == "llm":
                run_experiment_llm(path, exp_name)
            elif kind == "dl_ml":
                run_experiment_dl_ml(path, exp_name)
            elif kind == "other":
                run_experiment_other(path, exp_name)
            else:
                raise ValueError(f"Unknown kind: {kind}")

            print("Exporting tables to Parquet...")
            export_all_tables(container, run_dir, extra_args=extra_args)
            save_metadata(run_dir, path, exp_name)

            print(f"=== DONE {exp_name} -> {run_dir} ===")

        except Exception as e:
            print(f"⚠️  Error in {exp_name}: {e}")
            if CONFIG["stop_on_error"]:
                raise

if __name__ == "__main__":
    main()

