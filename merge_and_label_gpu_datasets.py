import pandas as pd
from pathlib import Path

DATASET_ROOT = Path("dataset")

# 0 = benign, 1 = malign
LABEL_MAP = {
    "passwd_hashcat": 1,  # malign (password cracker)
    "dl_cnn_train": 0,
    "dl_lstm_train": 0,
    "llm_bert": 0,
    "llm_gpt": 0,
    "llm_gpt_neo": 0,
    "llm_roberta": 0,
    "ml_forest": 0,
    "ml_logreg": 0,
    "ml_svm": 0,
}

# Output files
OUT_TIME_WINDOWS = "final_gpu_time_windows.parquet"
OUT_EVENT_TOKENS = "final_gpu_event_tokens.parquet"


def load_and_label_parquets(app_dir: Path, label: int, app_name: str):
    labeled_dfs = {"time_windows": None, "event_tokens": None}

    for exp_dir in app_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        time_window_path = exp_dir / "audit_gpu_time_window_events.parquet"
        token_path = exp_dir / "audit_gpu_event_tokens.parquet"

        if time_window_path.exists():
            df_tw = pd.read_parquet(time_window_path)
            df_tw["app_name"] = app_name
            df_tw["label"] = label
            df_tw["experiment_time"] = exp_dir.name
            labeled_dfs["time_windows"] = (
                pd.concat([labeled_dfs["time_windows"], df_tw], ignore_index=True)
                if labeled_dfs["time_windows"] is not None
                else df_tw
            )

        if token_path.exists():
            df_tok = pd.read_parquet(token_path)
            df_tok["app_name"] = app_name
            df_tok["label"] = label
            df_tok["experiment_time"] = exp_dir.name
            labeled_dfs["event_tokens"] = (
                pd.concat([labeled_dfs["event_tokens"], df_tok], ignore_index=True)
                if labeled_dfs["event_tokens"] is not None
                else df_tok
            )

    return labeled_dfs


def main():
    all_time_windows = []
    all_event_tokens = []

    for app_name, label in LABEL_MAP.items():
        app_dir = DATASET_ROOT / app_name
        if not app_dir.exists():
            print(f"[!] Skipping missing app dir: {app_dir}")
            continue

        labeled = load_and_label_parquets(app_dir, label, app_name)
        if labeled["time_windows"] is not None:
            all_time_windows.append(labeled["time_windows"])
        if labeled["event_tokens"] is not None:
            all_event_tokens.append(labeled["event_tokens"])

    # Concatenate everything and save unified parquet files
    if all_time_windows:
        final_tw = pd.concat(all_time_windows, ignore_index=True)
        final_tw.to_parquet(OUT_TIME_WINDOWS)
        print(f"[✓] Saved time window dataset → {OUT_TIME_WINDOWS} ({len(final_tw)} rows)")
    else:
        print("[!] No time window data found.")

    if all_event_tokens:
        final_et = pd.concat(all_event_tokens, ignore_index=True)
        final_et.to_parquet(OUT_EVENT_TOKENS)
        print(f"[✓] Saved event tokens dataset → {OUT_EVENT_TOKENS} ({len(final_et)} rows)")
    else:
        print("[!] No event token data found.")


if __name__ == "__main__":
    main()

