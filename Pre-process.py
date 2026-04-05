import os
import glob
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal

# Title: A Signal Preprocessing and Segmentation Pipeline for Surface EMG Gesture Recognition



FS                 = 2000.0
ADC_MID            = 2048.0
VALID_LABELS       = {0, 1, 2, 3, 4, 5, 6}
GAP_THRESHOLD_MS   = 100
ENVELOPE_CUTOFF_HZ = 10.0

TIME_COL      = "Time_ms"
LABEL_COL     = "Label"
# Unified column names for the four channels
CH_COLS       = ["CH1", "CH2", "CH3", "CH4"]
CH_ENV_COLS   = ["CH1_Env", "CH2_Env", "CH3_Env", "CH4_Env"]

DATA_DIR     = "."
FILE_PATTERN = "EMG_Dataset_V9*.csv"
# The output filename can be adjusted as needed
OUT_CSV      = "timeseries_all_sets_unified.csv"

FILTER_MODE  = "range"
SET_RANGE    = (1, 16)
SET_ID_LIST  = {1, 3, 5, 8}

ENVELOPE_MODE   = "ac_centered"

GESTURE_NAMES = {
    0: "Rest (Static Rest)",
    1: "Fist",
    2: "Spread",
    3: "Flexion",
    4: "Extension",
    5: "Pronation",
    6: "Supination",
}

# =========================
# Utility functions
# =========================

def compute_envelope(signal: np.ndarray, fs: float = FS,
                     cutoff: float = ENVELOPE_CUTOFF_HZ) -> np.ndarray:
    x_abs = np.abs(signal.astype(float))
    nyq   = fs / 2.0
    wn    = cutoff / nyq
    if not (0.0 < wn < 1.0):
        raise ValueError(f"Invalid envelope cutoff frequency: cutoff={cutoff}, fs={fs}")
    b, a = scipy.signal.butter(4, wn, btype="low")
    return scipy.signal.filtfilt(b, a, x_abs)


def extract_set_num(path: str) -> Optional[int]:
    m = re.search(r"_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def is_selected_set(set_num: int) -> bool:
    if FILTER_MODE == "all":
        return True
    if FILTER_MODE == "range":
        return SET_RANGE[0] <= set_num <= SET_RANGE[1]
    if FILTER_MODE == "list":
        return set_num in SET_ID_LIST
    raise ValueError(f"Unknown FILTER_MODE: {FILTER_MODE}")


def assign_segment_ids(time_ms: np.ndarray,
                       labels:   np.ndarray,
                       gap_threshold_ms: float = GAP_THRESHOLD_MS) -> np.ndarray:
    """
    Start a new segment when any of the following conditions occurs:
      1. Adjacent time difference > gap_threshold_ms (sampling interruption)
      2. Label changes (gesture switch)
    This ensures each segment contains a single Label, so sliding windows never cross gesture boundaries.
    """
    if len(time_ms) == 0:
        return np.array([], dtype=np.int64)

    dt           = np.diff(time_ms.astype(float))
    label_change = np.diff(labels.astype(int)) != 0
    breaks       = (dt > gap_threshold_ms) | label_change

    segment_id = np.ones(len(time_ms), dtype=np.int64)
    if len(time_ms) > 1:
        segment_id[1:] += np.cumsum(breaks).astype(np.int64)

    return segment_id


def compute_envelope_by_segments(signal: np.ndarray,
                                  segment_ids: np.ndarray,
                                  fs: float = FS,
                                  envelope_mode: str = ENVELOPE_MODE) -> np.ndarray:
    env         = np.zeros_like(signal, dtype=float)
    min_seg_len = int(fs * 0.05)

    for sid in np.unique(segment_ids):
        idx     = np.where(segment_ids == sid)[0]
        seg_sig = signal[idx].astype(float)

        if envelope_mode == "ac_centered":
            env_in = seg_sig - ADC_MID
        elif envelope_mode == "legacy_direct_abs":
            env_in = seg_sig
        else:
            raise ValueError(f"Unknown ENVELOPE_MODE: {envelope_mode}")

        env[idx] = np.abs(env_in) if len(env_in) < min_seg_len \
                   else compute_envelope(env_in, fs=fs)

    return env


# =========================
# Single-file processing
# =========================
def preprocess_one_file(csv_path: str, set_id: int) -> Tuple[Optional[pd.DataFrame], dict]:
    print(f"Processing: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read file: {e}")
        return None, {}

    required = [TIME_COL, LABEL_COL] + CH_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[Error] Missing required columns: {missing}")
        return None, {}

    print(f"  Original rows: {len(df):,}")
    print(f"  Original label distribution: {dict(df[LABEL_COL].value_counts().sort_index())}")

    # FIX-2  Filter Label first, then split into segments
    df = df[df[LABEL_COL].isin(VALID_LABELS)].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"  Rows after filtering: {len(df):,}")
    if df.empty:
        print("  [Warning] Data is empty after filtering, skipping")
        return None, {}

    # FIX-1 + FIX-2: Use the new function to split segments
    df["segment_id"] = assign_segment_ids(
        df[TIME_COL].to_numpy(dtype=float),
        df[LABEL_COL].to_numpy(dtype=int),
        gap_threshold_ms=GAP_THRESHOLD_MS,
    )
    seg_ids = df["segment_id"].to_numpy(dtype=np.int64)

    # Uniformly process envelope extraction for all 4 channels
    for raw_col, env_col in zip(CH_COLS, CH_ENV_COLS):
        sig        = df[raw_col].to_numpy(dtype=float)
        df[env_col] = compute_envelope_by_segments(sig, seg_ids, fs=FS,
                                                   envelope_mode=ENVELOPE_MODE)

    # Arrange output columns
    out_cols = [TIME_COL, "segment_id", LABEL_COL, *CH_COLS, *CH_ENV_COLS]
    df = df[out_cols].copy()
    df.insert(0, "set_id", int(set_id))

    seg_lengths = df.groupby("segment_id").size()

    # FIX-3  Print point-count quantiles for each segment
    pct = seg_lengths.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).astype(int)
    print(f"  Number of segments: {df['segment_id'].nunique()}")
    print(f"  Segment length quantiles (points):")
    print(f"    p10={pct[0.10]}  p25={pct[0.25]}  p50={pct[0.50]}"
          f"  p75={pct[0.75]}  p90={pct[0.90]}")
    print(f"  Shortest: {int(seg_lengths.min())}  Longest: {int(seg_lengths.max())}")

    for ws in [200, 300, 400]:
        usable = (seg_lengths >= ws).sum()
        total  = len(seg_lengths)
        print(f"  window_size={ws}: {usable}/{total} segments can contribute at least 1 window")

    print("  Sample count per gesture:")
    for lbl, cnt in df[LABEL_COL].value_counts().sort_index().items():
        print(f"    Label {lbl} ({GESTURE_NAMES.get(lbl, '?')}): {cnt:,} rows")

    info = {
        "n_rows":          int(len(df)),
        "n_segments":      int(df["segment_id"].nunique()),
        "min_segment_len": int(seg_lengths.min()),
        "max_segment_len": int(seg_lengths.max()),
        "p50_segment_len": int(pct[0.50]),
    }
    return df, info


# =========================
# Batch processing (unchanged)
# =========================
def run_all(data_dir: str = DATA_DIR, pattern: str = FILE_PATTERN,
            out_csv: str = OUT_CSV) -> Optional[pd.DataFrame]:
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        print(f"[Error] No files matching {data_dir}' were found under {pattern}")
        return None

    filtered_files, skipped_bad_name = [], []
    for f in files:
        set_num = extract_set_num(f)
        if set_num is None:
            skipped_bad_name.append(os.path.basename(f))
            continue
        if is_selected_set(set_num):
            filtered_files.append((set_num, f))

    filtered_files.sort(key=lambda x: x[0])

    if skipped_bad_name:
        print(f"[Warning] Filename format mismatch, skipped: {skipped_bad_name}")
    if not filtered_files:
        print("[Error] No files matched the filtering criteria")
        return None

    print(f"Files to process: {len(filtered_files)}\n")

    all_dfs, summary = [], {}
    for set_num, fpath in filtered_files:
        df, info = preprocess_one_file(fpath, set_num)
        if df is not None:
            all_dfs.append(df)
            summary[str(set_num)] = info
            print(f"  ✓ set {set_num} completed\n")

    if not all_dfs:
        print("[Error] No files were processed successfully")
        return None

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(out_csv, index=False)

    print("===== Final Summary =====")
    print(f"Total rows: {len(final_df):,}")
    print(f"Number of sets: {final_df['set_id'].nunique()}")
    seg_len_all = final_df.groupby(["set_id", "segment_id"]).size()
    print(f"Total segments: {len(seg_len_all)}")
    print(f"Global median segment length: {int(seg_len_all.median())} points")
    print("Total rows per gesture:")
    for lbl, cnt in final_df[LABEL_COL].value_counts().sort_index().items():
        print(f"  Label {lbl} ({GESTURE_NAMES.get(lbl, '?')}): {cnt:,} rows")
    print(f"\n✓ Saved: {out_csv}")
    return final_df


if __name__ == "__main__":
    run_all()


