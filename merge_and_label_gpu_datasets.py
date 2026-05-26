import pandas as pd
from pathlib import Path
from fastparquet import write

DATASET_ROOT = Path("dataset")

# 0 = benign, 1 = malign
LABEL_MAP = {
    "passwd_hashcat": 1,
    "miner_xmrig":    1,
    "miner_lolminer": 1,
    "miner_nbminer":  1,
    "miner_gminer":   1,
    "miner_bzminer":  1,
    "miner_srbminer": 1,
    "miner_trex":     1,
    "dl_cnn_train":   0,
    "dl_lstm_train":  0,
    "llm_bert":       0,
    "llm_gpt":        0,
    "llm_gpt_neo":    0,
    "llm_roberta":    0,
    "llm_bloom":      0,
    "ml_forest":      0,
    "ml_logreg":      0,
    "ml_svm":         0,
    "blender":        0,
}

# Time windows that were used during collection (must match CONFIG["time_windows"])
TIME_WINDOWS = [1, 2, 5]

# Output filenames — one per time window
OUT_TIME_WINDOWS_TPL = "final_gpu_time_windows_tw{tw}.parquet"
OUT_EVENT_TOKENS_TPL = "final_gpu_event_tokens_tw{tw}.parquet"


def append_parquet(df: pd.DataFrame, out_path: str, schema_set: set) -> None:
    """Append a DataFrame to a Parquet file, creating it on first write."""
    if df is None or df.empty:
        return
    df = df.convert_dtypes()
    if out_path not in schema_set:
        write(out_path, df, compression="SNAPPY")
        schema_set.add(out_path)
    else:
        write(out_path, df, append=True)


def process_app_for_window(app_name: str, label: int,
                            tw: int) -> tuple[pd.DataFrame | None,
                                              pd.DataFrame | None]:
    """
    Read all iteration folders under dataset/<app_name>/tw<N>/
    and return concatenated DataFrames for time_window_events and event_tokens.
    """
    tw_dir = DATASET_ROOT / app_name / f"tw{tw}"
    if not tw_dir.exists():
        print(f"  [!] Missing: {tw_dir}")
        return None, None

    all_tw, all_tok = [], []

    for exp_dir in sorted(tw_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        time_window_path = exp_dir / "audit_gpu_time_window_events.parquet"
        token_path       = exp_dir / "audit_gpu_event_tokens.parquet"

        if time_window_path.exists():
            df = pd.read_parquet(time_window_path)
            df["app_name"]        = app_name
            df["label"]           = label
            df["time_window_s"]   = tw
            df["experiment_time"] = exp_dir.name
            all_tw.append(df)

        if token_path.exists():
            df = pd.read_parquet(token_path)
            df["app_name"]        = app_name
            df["label"]           = label
            df["time_window_s"]   = tw
            df["experiment_time"] = exp_dir.name
            all_tok.append(df)

    df_tw  = pd.concat(all_tw,  ignore_index=True) if all_tw  else None
    df_tok = pd.concat(all_tok, ignore_index=True) if all_tok else None
    return df_tw, df_tok


def main() -> None:
    for tw in TIME_WINDOWS:
        print(f"\n{'='*50}")
        print(f"  Processing time window: {tw}s")
        print(f"{'='*50}")

        out_tw  = OUT_TIME_WINDOWS_TPL.format(tw=tw)
        out_tok = OUT_EVENT_TOKENS_TPL.format(tw=tw)
        schema_written: set = set()

        for app_name, label in LABEL_MAP.items():
            df_tw, df_tok = process_app_for_window(app_name, label, tw)

            if df_tw is not None:
                append_parquet(df_tw, out_tw, schema_written)
                print(f"  [✓] {app_name:20s} → {out_tw}  ({len(df_tw)} rows)")
            else:
                print(f"  [-] {app_name:20s}   no time_window data")

            # Uncomment to also merge event_tokens:
            # if df_tok is not None:
            #     append_parquet(df_tok, out_tok, schema_written)

        if schema_written:
            print(f"\n  [✅] Written: {out_tw}")
        else:
            print(f"\n  [!]  No data found for tw={tw}s — skipping output file")

    print("\n[✅] Merge complete.")
    print("Output files:")
    for tw in TIME_WINDOWS:
        p = Path(OUT_TIME_WINDOWS_TPL.format(tw=tw))
        if p.exists():
            mb = p.stat().st_size / 1e6
            print(f"  {p}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
