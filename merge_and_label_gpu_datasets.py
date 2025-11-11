import pandas as pd
from pathlib import Path
from fastparquet import write

DATASET_ROOT = Path("dataset")

# 0 = benign, 1 = malign
LABEL_MAP = {
    "passwd_hashcat": 1,  # malign
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

OUT_TIME_WINDOWS = "final_gpu_time_windows.parquet"
OUT_EVENT_TOKENS = "final_gpu_event_tokens.parquet"


def append_parquet(df: pd.DataFrame, out_path: str, schema_set: set):
    """Append a dataframe to a Parquet file incrementally."""
    if df is None or df.empty:
        return

    # Ensure consistent columns and dtypes
    df = df.convert_dtypes()

    if out_path not in schema_set:
        # First write
        write(out_path, df, compression="SNAPPY")
        schema_set.add(out_path)
    else:
        # Append to existing file
        write(out_path, df, append=True)


def process_app(app_name: str, label: int):
    app_dir = DATASET_ROOT / app_name
    if not app_dir.exists():
        print(f"[!] Skipping missing app dir: {app_dir}")
        return None, None

    all_tw, all_tok = [], []

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
            all_tw.append(df_tw)

        if token_path.exists():
            df_tok = pd.read_parquet(token_path)
            df_tok["app_name"] = app_name
            df_tok["label"] = label
            df_tok["experiment_time"] = exp_dir.name
            all_tok.append(df_tok)

    if all_tw:
        df_tw = pd.concat(all_tw, ignore_index=True)
    else:
        df_tw = None

    if all_tok:
        df_tok = pd.concat(all_tok, ignore_index=True)
    else:
        df_tok = None

    print(f"[✓] Processed {app_name}")
    return df_tw, df_tok


def main():
    schema_written = set()

    for app_name, label in LABEL_MAP.items():
        df_tw, df_tok = process_app(app_name, label)

        if df_tw is not None:
            append_parquet(df_tw, OUT_TIME_WINDOWS, schema_written)

        if df_tok is not None:
            append_parquet(df_tok, OUT_EVENT_TOKENS, schema_written)


if __name__ == "__main__":
    main()
    print("[✅] Merge complete.")

