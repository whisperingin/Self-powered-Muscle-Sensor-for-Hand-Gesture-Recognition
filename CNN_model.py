

import os
import time
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")

# Title: A One-Dimensional Convolutional Neural Network Framework for Surface EMG Gesture Classification

# ================= Configuration =================
CSV_FILE = "timeseries_all_sets_unified.csv"

WINDOW_SIZE = 300           # 100 ms @ 2000 Hz
STRIDE = WINDOW_SIZE // 4   # = 75
MIN_LABEL_PURITY = 0.90
CHANNELS = ["CH1", "CH2", "CH3", "CH4"]

N_CLASSES = 7
RANDOM_STATE = 42
VAL_CHECK_EVERY = 5

GESTURE_NAMES = {
    0: "Rest",
    1: "Fist",
    2: "Spread",
    3: "Flexion",
    4: "Extension",
    5: "Pronation",
    6: "Supination",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() or 0)
AMP_ENABLED = (DEVICE.type == "cuda")
FOLD_CACHE_ENABLED = True

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
# ========================================


# ========================================
# Utility functions
# ========================================
def set_seed(seed: int = RANDOM_STATE, deterministic: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def to_py_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f} s"
    return f"{sec / 60:.2f} min"


def stage_done(stage_name: str, t0: float):
    dt = time.time() - t0
    print(f"[Timing] {stage_name}: {format_seconds(dt)}")


def validate_dataframe(df: pd.DataFrame):
    required_cols = ["set_id", "segment_id", "Label"] + CHANNELS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if df.empty:
        raise ValueError("CSV is empty and cannot be used for training")

    labels = df["Label"].dropna().astype(int).unique().tolist()
    invalid = sorted(set(labels) - set(range(N_CLASSES)))
    if invalid:
        raise ValueError(
            f"Found invalid labels {invalid}. Please ensure Label only contains values from 0 to {N_CLASSES - 1}"
        )

    bad_numeric = []
    for c in CHANNELS:
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad_numeric.append(c)
    if bad_numeric:
        raise ValueError(f"The following channel columns are not numeric: {bad_numeric}")


def majority_label_and_purity(lbl_seg: np.ndarray):
    """
    Returns:
      maj_label: Majority label within the window
      purity   : Proportion of the majority label, ranging from 0 to 1
    """
    counts = np.bincount(lbl_seg, minlength=N_CLASSES)
    maj = int(np.argmax(counts))
    purity = float(counts[maj] / max(len(lbl_seg), 1))
    return maj, purity


def split_time_order(X, y, ratio=0.8):
    """
    Single-set scenario; split in chronological order
    """
    if len(y) < 2:
        raise ValueError("Too few windows to perform a chronological split")

    split = int(len(y) * ratio)
    if split <= 0 or split >= len(y):
        raise ValueError(
            f"Chronological split failed. Total windows={len(y)}, split index={split}. Please increase the data size or reduce WINDOW_SIZE"
        )

    return X[:split], y[:split], X[split:], y[split:]


def fit_channel_zscore(X: np.ndarray):
    """
    Fit z-score statistics per channel on windowed data
    X shape: (N, T, C)
    Compute statistics across the (N, T) dimensions to obtain one mean/std per channel
    """
    if X.ndim != 3:
        raise ValueError(f"Invalid X dimensions: expected 3D, got {X.ndim}")

    mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)  # (1,1,C)
    std = X.std(axis=(0, 1), keepdims=True).astype(np.float32)    # (1,1,C)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_channel_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Apply z-score to windowed data using the provided channel statistics
    """
    return ((X.astype(np.float32) - mean) / std).astype(np.float32)


def standardise_by_train(X_tr: np.ndarray, X_other: np.ndarray):
    """
    Fit z-score using only the training set, then apply it to the training set and another dataset
    Returns:
      X_tr_z, X_other_z, mean, std
    """
    mean, std = fit_channel_zscore(X_tr)
    X_tr_z = apply_channel_zscore(X_tr, mean, std)
    X_other_z = apply_channel_zscore(X_other, mean, std)
    return X_tr_z, X_other_z, mean, std


# ========================================
# 1. Build the dataset with sliding windows
# ========================================
def build_windows(df: pd.DataFrame):
    """
    Group by (set_id, segment_id) and apply sliding windows. Returns:
      X      : (N, WINDOW_SIZE, n_channels)
      y      : (N,)
      groups : (N,) Corresponding set_id values, used for LOSO
    """
    n_dropped_impure = 0
    X, y, groups = [], [], []

    n_dropped_nan = 0
    n_skipped_short_segments = 0

    for (set_id, segment_id), group in df.groupby(["set_id", "segment_id"], sort=True):
        vals = group[CHANNELS].to_numpy(dtype=np.float32)
        labels = group["Label"].to_numpy(dtype=np.int64)
        n = len(group)

        if n < WINDOW_SIZE:
            print(
                f"  Skipping set {set_id}, segment {segment_id}，"
                f"length {n} < WINDOW_SIZE {WINDOW_SIZE}"
            )
            n_skipped_short_segments += 1
            continue

        for i in range(0, n - WINDOW_SIZE + 1, STRIDE):
            seg = vals[i:i + WINDOW_SIZE]
            lbl_seg = labels[i:i + WINDOW_SIZE]

            if np.isnan(seg).any():
                n_dropped_nan += 1
                continue

            seg_label, purity = majority_label_and_purity(lbl_seg)

            if purity < MIN_LABEL_PURITY:
                n_dropped_impure += 1
                continue

            X.append(seg)
            y.append(seg_label)
            groups.append(int(set_id))

    if len(X) == 0:
        raise ValueError(
            "No windows were generated. Please check the data length, WINDOW_SIZE, STRIDE, or whether there are many NaN values"
        )

    print(f"  Total windows: {len(X)}")
    print(f"  Dropped NaN windows: {n_dropped_nan}")
    print(f"  Dropped low-purity windows (<{MIN_LABEL_PURITY:.2f}): {n_dropped_impure}")
    print(f"  Skipped short segments: {n_skipped_short_segments}")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups, dtype=np.int64),
    )


def arrays_to_tensors(X, y):
    Xt = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1))).float()
    yt = torch.from_numpy(y.astype(np.int64))
    return Xt, yt


def to_tensor_dataset(X, y):
    Xt, yt = arrays_to_tensors(X, y)
    return TensorDataset(Xt, yt)


def build_loader(dataset, batch_size, shuffle):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs)


def make_eval_loader(X, y, batch_size=256):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Evaluation set is empty")

    ds = to_tensor_dataset(X, y)
    return build_loader(ds, batch_size=batch_size, shuffle=False)


def make_loaders(X_tr, y_tr, X_te, y_te, batch_size):
    """
    X: (N, T, C) -> (N, C, T)
    """
    if len(X_tr) == 0 or len(y_tr) == 0:
        raise ValueError("Training set is empty")
    if len(X_te) == 0 or len(y_te) == 0:
        raise ValueError("Validation set or test set is empty")

    tr_ds = to_tensor_dataset(X_tr, y_tr)
    te_ds = to_tensor_dataset(X_te, y_te)

    tr_loader = build_loader(tr_ds, batch_size=batch_size, shuffle=True)
    te_loader = build_loader(te_ds, batch_size=256, shuffle=False)

    return tr_loader, te_loader


def split_final_train_val(X, y, groups=None, ratio=0.85):
    """
    In the final training stage, split off an additional validation subset from the training set,
    used only to select the best epoch, without using the test set to choose the model.

    Priority strategy:
      1. If groups exist and there are multiple sets, use the last set as validation
      2. Otherwise split in chronological order
    """
    if groups is not None:
        groups = np.asarray(groups)

        if len(groups) != len(y):
            raise ValueError("train_groups length does not match the number of training samples")

        uniq = np.unique(groups)

        if len(uniq) > 1:
            val_sid = uniq[-1]
            mask_va = (groups == val_sid)
            mask_tr = ~mask_va

            if not mask_tr.any():
                raise ValueError("Final training subset is empty")
            if not mask_va.any():
                raise ValueError("Final validation subset is empty")

            return (
                X[mask_tr], y[mask_tr],
                X[mask_va], y[mask_va],
                {
                    "final_val_mode": "set_holdout",
                    "final_val_set_id": to_py_scalar(val_sid),
                }
            )

    X_fit, y_fit, X_va, y_va = split_time_order(X, y, ratio=ratio)
    return (
        X_fit, y_fit,
        X_va, y_va,
        {
            "final_val_mode": "time_order",
            "final_val_set_id": None,
        }
    )


def plot_confusion_matrix_heatmap(cm, class_names, save_path):
    """
    Plot and save a confusion matrix heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_optimizer(model, lr, weight_decay):
    kwargs = dict(lr=lr, weight_decay=weight_decay)

    if DEVICE.type == "cuda":
        try:
            return optim.Adam(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass

    return optim.Adam(model.parameters(), **kwargs)


def get_amp_dtype():
    return torch.float16


# ========================================
# 2. Dynamic CNN model
# ========================================
class DynamicEMGNet(nn.Module):
    """
    Configurable 1D-CNN:
      Input: (B, C, T)
      N layers of Conv1d -> BN -> ReLU -> MaxPool(2)
      AdaptiveAvgPool -> Flatten
      FC -> ReLU -> Dropout -> FC(n_classes)
    """

    def __init__(self, n_channels, n_classes, params: dict):
        super().__init__()

        n_layers = params["n_layers"]
        filters = params["filters"]
        kernel = params["kernel"]
        fc_hidden = params["fc_hidden"]
        dropout1 = params["dropout1"]
        dropout2 = params["dropout2"]

        layers = []
        in_ch = n_channels

        for out_ch in filters:
            pad = kernel // 2

            layers += [
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel,
                    padding=pad,
                    bias=False,
                ),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_ch, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x


def expand_params(best_params: dict) -> dict:
    """
    Reconstruct the full parameter set from Optuna best_params
    """
    n_layers = int(best_params["n_layers"])
    filter_base = int(best_params["filter_base"])

    filters = []
    f = filter_base
    for _ in range(n_layers):
        filters.append(f)
        f = min(f * 2, 256)

    return {
        "n_layers": n_layers,
        "filters": filters,
        "kernel": int(best_params["kernel"]),
        "fc_hidden": int(best_params["fc_hidden"]),
        "dropout1": float(best_params["dropout1"]),
        "dropout2": float(best_params["dropout2"]),
        "lr": float(best_params["lr"]),
        "weight_decay": float(best_params["weight_decay"]),
        "batch_size": int(best_params["batch_size"]),
    }


# ========================================
# 3. Training and evaluation
# ========================================
@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0

    amp_dtype = get_amp_dtype()

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=amp_dtype,
            enabled=AMP_ENABLED
        ):
            logits = model(xb)

        pred = logits.argmax(dim=1)

        correct += (pred == yb).sum().item()
        total += len(yb)

    if total == 0:
        raise ValueError("Evaluation set is empty; accuracy cannot be computed")

    return correct / total


def train_eval(X_tr, y_tr, X_va, y_va, params, n_epochs, trial=None, report_offset=0):
    """
    Train a CNN with the given params and return the best validation accuracy
    Here z-score is fitted only on X_tr, then applied to X_tr / X_va
    """
    set_seed(RANDOM_STATE, deterministic=False)

    X_tr_z, X_va_z, _, _ = standardise_by_train(X_tr, X_va)

    loader_tr, loader_va = make_loaders(
        X_tr_z, y_tr, X_va_z, y_va, params["batch_size"]
    )

    model = DynamicEMGNet(
        n_channels=len(CHANNELS),
        n_classes=N_CLASSES,
        params=params,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    amp_dtype = get_amp_dtype()

    best_acc = 0.0

    for ep in range(n_epochs):
        model.train()

        for xb, yb in loader_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=amp_dtype,
                enabled=AMP_ENABLED
            ):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if (ep + 1) % VAL_CHECK_EVERY == 0 or ep == n_epochs - 1:
            acc = evaluate_accuracy(model, loader_va)
            best_acc = max(best_acc, acc)

            if trial is not None:
                step = report_offset + ep + 1
                trial.report(acc, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_acc


def prepare_fold_cache(X, y, groups):
    """
    Precompute the following for each fixed fold:
      1. train/val split
      2. train-only z-score
      3. numpy -> torch tensor
    Prepare everything in advance to avoid repeated processing for each trial
    """
    caches = []
    inner_set_ids = np.unique(groups)

    if len(inner_set_ids) > 1:
        for val_sid in inner_set_ids:
            mask_va = (groups == val_sid)
            mask_tr = ~mask_va

            if not mask_va.any():
                raise ValueError(f"Fold validation set is empty, set_id={val_sid}")
            if not mask_tr.any():
                raise ValueError(f"fold Training set is empty，set_id={val_sid}")

            X_tr_z, X_va_z, _, _ = standardise_by_train(X[mask_tr], X[mask_va])

            Xt_tr, yt_tr = arrays_to_tensors(X_tr_z, y[mask_tr])
            Xt_va, yt_va = arrays_to_tensors(X_va_z, y[mask_va])

            caches.append({
                "val_sid": int(val_sid),
                "Xt_tr": Xt_tr,
                "yt_tr": yt_tr,
                "Xt_va": Xt_va,
                "yt_va": yt_va,
                "n_tr": int(mask_tr.sum()),
                "n_va": int(mask_va.sum()),
            })
    else:
        X_tr, y_tr, X_va, y_va = split_time_order(X, y, ratio=0.8)
        X_tr_z, X_va_z, _, _ = standardise_by_train(X_tr, X_va)

        Xt_tr, yt_tr = arrays_to_tensors(X_tr_z, y_tr)
        Xt_va, yt_va = arrays_to_tensors(X_va_z, y_va)

        caches.append({
            "val_sid": None,
            "Xt_tr": Xt_tr,
            "yt_tr": yt_tr,
            "Xt_va": Xt_va,
            "yt_va": yt_va,
            "n_tr": len(y_tr),
            "n_va": len(y_va),
        })

    return caches


def make_loaders_from_cache(cache, batch_size):
    tr_ds = TensorDataset(cache["Xt_tr"], cache["yt_tr"])
    va_ds = TensorDataset(cache["Xt_va"], cache["yt_va"])

    tr_loader = build_loader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = build_loader(va_ds, batch_size=256, shuffle=False)

    return tr_loader, va_loader


def train_eval_cached(cache, params, n_epochs, trial=None, report_offset=0):
    set_seed(RANDOM_STATE, deterministic=False)

    loader_tr, loader_va = make_loaders_from_cache(cache, params["batch_size"])

    model = DynamicEMGNet(
        n_channels=len(CHANNELS),
        n_classes=N_CLASSES,
        params=params,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    amp_dtype = get_amp_dtype()

    best_acc = 0.0

    for ep in range(n_epochs):
        model.train()

        for xb, yb in loader_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=amp_dtype,
                enabled=AMP_ENABLED
            ):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if (ep + 1) % VAL_CHECK_EVERY == 0 or ep == n_epochs - 1:
            acc = evaluate_accuracy(model, loader_va)
            best_acc = max(best_acc, acc)

            if trial is not None:
                step = report_offset + ep + 1
                trial.report(acc, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_acc


# ========================================
# 4. Optuna objective
# ========================================
def make_objective(X, y, groups, n_epochs):
    """
    Build the objective for the tuning set
      If the inner split has multiple sets -> LOSO
      If the inner split has only 1 set -> chronological 80/20
    """
    inner_set_ids = np.unique(groups)
    inner_multi = len(inner_set_ids) > 1

    fold_cache = None
    if FOLD_CACHE_ENABLED:
        t_cache = time.time()
        fold_cache = prepare_fold_cache(X, y, groups)
        stage_done("Prebuild fold cache", t_cache)

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 4)
        filter_base = trial.suggest_categorical("filter_base", [16, 32, 64])

        filters = []
        f = filter_base
        for _ in range(n_layers):
            filters.append(f)
            f = min(f * 2, 256)

        params = {
            "n_layers": n_layers,
            "filters": filters,
            "kernel": trial.suggest_categorical("kernel", [3, 5, 7, 11]),
            "fc_hidden": trial.suggest_categorical("fc_hidden", [64, 128, 256]),
            "dropout1": trial.suggest_float("dropout1", 0.2, 0.6, step=0.1),
            "dropout2": trial.suggest_float("dropout2", 0.1, 0.4, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        }

        if inner_multi:
            fold_accs = []
            trial_t0 = time.time()

            print(f"\n[Trial {trial.number}] started, params: {params}")

            if FOLD_CACHE_ENABLED and fold_cache is not None:
                for fold_idx, cache in enumerate(fold_cache):
                    val_sid = cache["val_sid"]
                    fold_t0 = time.time()

                    print(
                        f"[Trial {trial.number}] Fold {fold_idx + 1}/{len(fold_cache)}  "
                        f"val_set={val_sid}  "
                        f"train_n={cache['n_tr']}  "
                        f"val_n={cache['n_va']}"
                    )

                    acc = train_eval_cached(
                        cache=cache,
                        params=params,
                        n_epochs=n_epochs,
                        trial=trial,
                        report_offset=fold_idx * n_epochs,
                    )
                    fold_accs.append(acc)

                    print(
                        f"[Trial {trial.number}] Fold {fold_idx + 1} completed  "
                        f"acc={acc:.4f}  "
                        f"time={format_seconds(time.time() - fold_t0)}"
                    )

            else:
                for fold_idx, val_sid in enumerate(inner_set_ids):
                    fold_t0 = time.time()

                    mask_va = (groups == val_sid)
                    mask_tr = ~mask_va

                    if not mask_va.any():
                        raise ValueError(f"Fold validation set is empty, set_id={val_sid}")
                    if not mask_tr.any():
                        raise ValueError(f"fold Training set is empty，set_id={val_sid}")

                    print(
                        f"[Trial {trial.number}] Fold {fold_idx + 1}/{len(inner_set_ids)}  "
                        f"val_set={val_sid}  "
                        f"train_n={mask_tr.sum()}  "
                        f"val_n={mask_va.sum()}"
                    )

                    acc = train_eval(
                        X[mask_tr], y[mask_tr],
                        X[mask_va], y[mask_va],
                        params=params,
                        n_epochs=n_epochs,
                        trial=trial,
                        report_offset=fold_idx * n_epochs,
                    )
                    fold_accs.append(acc)

                    print(
                        f"[Trial {trial.number}] Fold {fold_idx + 1} completed  "
                        f"acc={acc:.4f}  "
                        f"time={format_seconds(time.time() - fold_t0)}"
                    )

            mean_acc = float(np.mean(fold_accs))
            print(
                f"[Trial {trial.number}] completed  "
                f"mean_acc={mean_acc:.4f}  "
                f"total_time={format_seconds(time.time() - trial_t0)}"
            )

            return mean_acc

        if FOLD_CACHE_ENABLED and fold_cache is not None:
            return train_eval_cached(
                cache=fold_cache[0],
                params=params,
                n_epochs=n_epochs,
                trial=trial,
                report_offset=0,
            )

        X_tr, y_tr, X_va, y_va = split_time_order(X, y, ratio=0.8)

        return train_eval(
            X_tr, y_tr,
            X_va, y_va,
            params=params,
            n_epochs=n_epochs,
            trial=trial,
            report_offset=0,
        )

    return objective


# ========================================
# 5. Final training
# ========================================
def final_train(
    X_tr, y_tr, X_te, y_te,
    best_params,
    n_epochs,
    save_path="model_cnn_tuned.pt",
    extra_meta=None,
    train_groups=None,
    final_val_ratio=0.85,
):
    """
    Perform final training using the best hyperparameters
    Key points:
      1. Split an extra validation subset from X_tr to choose the best epoch
      2. Fit z-score only on the actual training subset X_fit used for fitting
      3. Save the weights from the epoch with the best validation accuracy
      4. Perform the final evaluation only once on the external test set X_te
      5. Print the confusion matrix and save the heatmap
    """
    set_seed(RANDOM_STATE, deterministic=False)
    final_t0 = time.time()

    params = expand_params(best_params)

    X_fit, y_fit, X_va, y_va, final_val_meta = split_final_train_val(
        X_tr, y_tr,
        groups=train_groups,
        ratio=final_val_ratio,
    )

    X_fit_z, X_va_z, z_mean, z_std = standardise_by_train(X_fit, X_va)
    X_te_z = apply_channel_zscore(X_te, z_mean, z_std)

    loader_tr = build_loader(
        to_tensor_dataset(X_fit_z, y_fit),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    loader_va = make_eval_loader(X_va_z, y_va, batch_size=256)
    loader_te = make_eval_loader(X_te_z, y_te, batch_size=256)

    model = DynamicEMGNet(
        n_channels=len(CHANNELS),
        n_classes=N_CLASSES,
        params=params,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    amp_dtype = get_amp_dtype()

    best_acc = -1.0
    best_epoch = 0
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    print(f"\nFinal training ({n_epochs} epochs) ...")
    print("z-score is fitted only on the final training subset")
    print(
        f"Final training subset: {len(y_fit)} windows | "
        f"Final validation subset: {len(y_va)} windows | "
        f"Final test set: {len(y_te)} windows"
    )
    print(
        f"Final validation strategy: {final_val_meta['final_val_mode']} | "
        f"val_set_id={final_val_meta['final_val_set_id']}"
    )

    for ep in range(n_epochs):
        ep_t0 = time.time()
        model.train()

        ep_loss = 0.0
        n_correct = 0
        n_total = 0

        for xb, yb in loader_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=amp_dtype,
                enabled=AMP_ENABLED
            ):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item() * len(yb)
            n_correct += (logits.detach().argmax(dim=1) == yb).sum().item()
            n_total += len(yb)

        scheduler.step()

        train_acc = n_correct / max(n_total, 1)
        val_acc = evaluate_accuracy(model, loader_va)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = ep + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if (ep + 1) % 10 == 0 or ep == n_epochs - 1:
            print(
                f"  Epoch {ep + 1:>3}/{n_epochs}  "
                f"loss={ep_loss / max(n_total, 1):.4f}  "
                f"train={train_acc:.4f}  "
                f"val={val_acc:.4f}  "
                f"best_val={best_acc:.4f}  "
                f"time={format_seconds(time.time() - ep_t0)}"
            )

    print(f"\nUsing the best validation checkpoint: epoch {best_epoch}, val_acc={best_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in loader_te:
            xb = xb.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=amp_dtype,
                enabled=AMP_ENABLED
            ):
                logits = model(xb)

            pred = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(pred)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    final_acc = accuracy_score(y_true, y_pred)

    print(f"\n{'=' * 55}")
    print(f"  Final test accuracy: {final_acc:.4f}")
    print(f"{'=' * 55}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(N_CLASSES)),
            target_names=[GESTURE_NAMES[i] for i in range(N_CLASSES)],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    print("Confusion Matrix:")
    print(cm)

    try:
        cm_df = pd.DataFrame(
            cm,
            index=[f"True_{GESTURE_NAMES[i]}" for i in range(N_CLASSES)],
            columns=[f"Pred_{GESTURE_NAMES[i]}" for i in range(N_CLASSES)],
        )
        print("\nConfusion Matrix DataFrame:")
        print(cm_df)
    except Exception:
        pass

    heatmap_path = str(Path(save_path).with_name(Path(save_path).stem + "_confusion_matrix.png"))
    plot_confusion_matrix_heatmap(
        cm=cm,
        class_names=[GESTURE_NAMES[i] for i in range(N_CLASSES)],
        save_path=heatmap_path,
    )
    print(f"Confusion matrix heatmap saved: {heatmap_path}")

    save_payload = {
        "model_state_dict": best_state,
        "model_params": params,
        "n_channels": len(CHANNELS),
        "n_classes": N_CLASSES,
        "channels": CHANNELS,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "gesture_names": GESTURE_NAMES,
        "best_val_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "best_optuna_params": best_params,
        "min_label_purity": MIN_LABEL_PURITY,
        "final_val_mode": final_val_meta["final_val_mode"],
        "final_val_set_id": final_val_meta["final_val_set_id"],
        "final_test_acc": float(final_acc),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_heatmap_path": heatmap_path,
        "zscore_mean": z_mean.reshape(-1).astype(np.float32).tolist(),
        "zscore_std": z_std.reshape(-1).astype(np.float32).tolist(),
    }

    if extra_meta is not None:
        save_payload.update(extra_meta)

    torch.save(save_payload, save_path)
    print(f"Model saved: {save_path}")
    stage_done("Final training + testing + save model", final_t0)
    return final_acc


# ========================================
# 6. Main flow
# ========================================
if __name__ == "__main__":
    total_t0 = time.time()

    parser = argparse.ArgumentParser(
        description="EMG CNN hyperparameter search (accelerated: AMP + TF32 + DataLoader workers + fold cache)"
    )
    parser.add_argument(
        "--trials",
        default=50,
        type=int,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        help="Training epochs per trial (default: 30)",
    )
    parser.add_argument(
        "--final-epochs",
        default=80,
        type=int,
        help="Final training epochs (default: 80)",
    )
    parser.add_argument(
        "--csv",
        default=CSV_FILE,
        help="Path to the preprocessed CSV file",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"AMP enabled: {AMP_ENABLED}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"FOLD_CACHE_ENABLED: {FOLD_CACHE_ENABLED}")
    print(f"Optuna trials: {args.trials}, each with {args.epochs} epochs")
    print(f"Final training: {args.final_epochs} epochs\n")

    t_load = time.time()
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    validate_dataframe(df)

    print(f"Total rows: {len(df):,}")
    print(f"Number of sets: {df['set_id'].nunique()}")
    print(f"Number of segments: {df[['set_id', 'segment_id']].drop_duplicates().shape[0]}")
    stage_done("Load CSV + validation", t_load)

    t_windows = time.time()
    print("\nBuilding windows ...")
    X, y, groups = build_windows(df)
    stage_done("Sliding-window construction", t_windows)

    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(
        "Window count per gesture:",
        {
            GESTURE_NAMES[int(k)]: int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        }
    )

    set_ids = np.unique(groups)
    n_sets = len(set_ids)

    # 3) Outer strict test split
    t_split = time.time()
    if n_sets >= 2:
        holdout_sid = set_ids[-1]

        mask_tune = (groups != holdout_sid)
        mask_test = (groups == holdout_sid)

        if not mask_tune.any():
            raise ValueError("Tuning set is empty")
        if not mask_test.any():
            raise ValueError("Independent test set is empty")

        X_tune, y_tune, g_tune = X[mask_tune], y[mask_tune], groups[mask_tune]
        X_test, y_test = X[mask_test], y[mask_test]

        inner_set_ids = np.unique(g_tune)

        print(f"\nDetected {n_sets} sets")
        print(f"Outer strict test set: set {holdout_sid}")
        print(f"Tuning set(s): {list(inner_set_ids)}")

        if len(inner_set_ids) > 1:
            print("Inner tuning strategy: Leave-One-Set-Out")
        else:
            print("Inner tuning strategy: single-set chronological 80/20")

    else:
        holdout_sid = None
        X_tune, y_tune, g_tune = X, y, groups
        X_test, y_test = None, None

        print("\nOnly 1 set detected")
        print("Unable to reserve a strict independent test set")
        print("Both tuning and final evaluation will use chronological 80/20")
    stage_done("Outer train/test split", t_split)

    # 4) Optuna search
    print(f"\n{'=' * 55}")
    print("  Starting Optuna hyperparameter search")
    print(f"{'=' * 55}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(10, args.trials),
        n_warmup_steps=max(5, VAL_CHECK_EVERY),
    )
    sampler = TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(
        X_tune,
        y_tune,
        g_tune,
        n_epochs=args.epochs,
    )

    t_optuna = time.time()
    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t_optuna
    stage_done("Optuna search", t_optuna)

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(complete_trials) == 0:
        raise RuntimeError("All trials were pruned or failed; no valid best_params are available")

    print(f"\n{'=' * 55}")
    print(f"  Search completed  elapsed: {elapsed / 60:.1f} min")
    print(f"{'=' * 55}")
    print(f"  Best validation accuracy: {study.best_value:.4f}")
    print("  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k:<20}: {v}")

    result_summary = {
        "best_val_acc": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": int(args.trials),
        "search_epochs": int(args.epochs),
        "elapsed_min": round(elapsed / 60, 2),
        "n_total_sets": int(n_sets),
        "holdout_set_id": None if holdout_sid is None else to_py_scalar(holdout_sid),
    }

    with open("cnn_tune_results.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    print("\nSearch results saved: cnn_tune_results.json")

    # 5) Final training
    print(f"\n{'=' * 55}")
    print(f"  Final training with best parameters ({args.final_epochs} epochs)")
    print(f"{'=' * 55}")

    if n_sets >= 2:
        X_tr, y_tr = X_tune, y_tune
        X_te, y_te = X_test, y_test

        print(f"Final training set: {len(y_tr)} windows")
        print(f"Final test set: set {holdout_sid}, total {len(y_te)} windows")

        extra_meta = {
            "outer_holdout_set_id": to_py_scalar(holdout_sid),
            "tuning_set_ids": [to_py_scalar(s) for s in np.unique(g_tune)],
        }
        train_groups_for_final = g_tune

    else:
        X_tr, y_tr, X_te, y_te = split_time_order(X_tune, y_tune, ratio=0.8)

        print(f"Final split: first {len(y_tr)} for training, last {len(y_te)} for testing")

        extra_meta = {
            "outer_holdout_set_id": None,
            "tuning_set_ids": [to_py_scalar(s) for s in np.unique(g_tune)],
        }
        train_groups_for_final = None

    final_train(
        X_tr, y_tr,
        X_te, y_te,
        best_params=study.best_params,
        n_epochs=args.final_epochs,
        save_path="model_cnn_tuned.pt",
        extra_meta=extra_meta,
        train_groups=train_groups_for_final,
        final_val_ratio=0.85,
    )

    # 6) Top 5 trials
    print(f"\n{'=' * 55}")
    print("  Top 5 trial results")
    print(f"{'=' * 55}")

    top5 = sorted(
        complete_trials,
        key=lambda t: t.value,
        reverse=True,
    )[:5]

    for i, t in enumerate(top5, start=1):
        print(
            f"  #{i}  "
            f"val_acc={t.value:.4f}  "
            f"layers={t.params.get('n_layers')}  "
            f"filter_base={t.params.get('filter_base')}  "
            f"kernel={t.params.get('kernel')}  "
            f"lr={t.params.get('lr', 0):.2e}"
        )

    stage_done("Total script runtime", total_t0)



