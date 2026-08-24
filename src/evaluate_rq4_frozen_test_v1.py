"""Frozen RQ4 F1/F2 test evaluation (Stage 24C).

Why this file exists
--------------------
First authorised test evaluation of frozen F1 and F2 against F0 (RQ3 A2).
Reuses A2 test predictions; no F0 rerun. No training or threshold changes.

How to run
----------
    source .venv/bin/activate
    PYTHONPATH=src python src/evaluate_rq4_frozen_test_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, select_device, stop_if
from rq3_augmentations_v1 import IMAGENET_MEAN, IMAGENET_STD
from rq4_frequency_cnn_v1 import FrequencyOnlyCNNV1
from rq4_frequency_transform_v1 import FrequencyTransformV1, NORM_PATH
from rq4_rgb_frequency_fusion_v1 import RGBFrequencyFusionV1

BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_KNOWN = 456
EXPECTED_UNSEEN = 1712
EXPECTED_SOURCES = 2168
EXPECTED_ROWS = 10840
EXPECTED_METRICS_ROWS = 30

CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
STRONG_SCORE_CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2"]
SPLITS = ["known_test", "unseen_test"]
KNOWN_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]
UNSEEN_GENERATORS = ["Midjourney", "VQDM", "Wukong"]
REGIMES = ["F0", "F1", "F2"]

SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
ROBUST_MANIFEST = PROJECT_ROOT / "metadata" / "robustness_v1_manifest.csv"
SCREEN_MANIFEST = PROJECT_ROOT / "metadata" / "screenshot_v1_manifest.csv"

F0_PRED = PROJECT_ROOT / "results" / "rq3_A2_test_predictions_v1.csv"
F0_FROZEN = PROJECT_ROOT / "results" / "rq3_A2_frozen_config_v1.json"
F1_FROZEN = PROJECT_ROOT / "results" / "rq4_F1_frozen_config_v1.json"
F2_FROZEN = PROJECT_ROOT / "results" / "rq4_F2_frozen_config_v1.json"
F1_CKPT = PROJECT_ROOT / "models" / "rq4_F1_frequency_only_selected_v1.pt"
F2_CKPT = PROJECT_ROOT / "models" / "rq4_F2_rgb_frequency_fusion_selected_v1.pt"
GATE_24B = PROJECT_ROOT / "results" / "rq4_24b_gate_v1.json"

F1_PRED_OUT = PROJECT_ROOT / "results" / "rq4_F1_test_predictions_v1.csv"
F2_PRED_OUT = PROJECT_ROOT / "results" / "rq4_F2_test_predictions_v1.csv"
METRICS_CSV = PROJECT_ROOT / "results" / "rq4_test_metrics_v1.csv"
GEN_RECALL_CSV = PROJECT_ROOT / "results" / "rq4_generator_recall_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "rq4_test_evaluation_report_v1.txt"
PAPER_TABLE = PROJECT_ROOT / "paper" / "tables" / "rq4_frequency_fusion_comparison.csv"

FIG_AUC = PROJECT_ROOT / "figures" / "rq4_unseen_auc_by_condition_v1.png"
FIG_DELTA = PROJECT_ROOT / "figures" / "rq4_unseen_delta_auc_v1.png"
FIG_ROBUST = PROJECT_ROOT / "figures" / "rq4_strong_robust_auc_v1.png"
FIG_GAP = PROJECT_ROOT / "figures" / "rq4_known_unseen_original_auc_v1.png"
FIG_THR = PROJECT_ROOT / "figures" / "rq4_threshold_behaviour_v1.png"


class PathRGBDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} size {rgb.size}")
        return self.transform(rgb), torch.tensor(float(row["label"]), dtype=torch.float32), index


class PathFusionDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, freq_transform: FrequencyTransformV1):
        self.rows = rows.reset_index(drop=True)
        self.freq_transform = freq_transform
        self.rgb_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} size {rgb.size}")
        return self.rgb_tensor(rgb), self.freq_transform(rgb), torch.tensor(float(row["label"]), dtype=torch.float32), index


class PathFreqDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, freq_transform: FrequencyTransformV1):
        self.rows = rows.reset_index(drop=True)
        self.freq_transform = freq_transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} size {rgb.size}")
        return self.freq_transform(rgb), torch.tensor(float(row["label"]), dtype=torch.float32), index


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_condition_frames() -> dict[str, pd.DataFrame]:
    meta = pd.read_csv(SPLIT_META_PATH)
    robust = pd.read_csv(ROBUST_MANIFEST)
    screen = pd.read_csv(SCREEN_MANIFEST)

    frames: dict[str, pd.DataFrame] = {}
    original = meta[meta["split"].isin(SPLITS)].copy()
    original = original.rename(columns={"image_id": "source_image_id", "processed_path": "path"})
    original = original[["source_image_id", "path", "split", "label", "generator"]]
    original = original.sort_values(["split", "source_image_id"]).reset_index(drop=True)
    stop_if(len(original) != EXPECTED_SOURCES, f"original sources {len(original)}")
    frames["original"] = original

    for condition in ["jpeg_q50", "resize_112", "blur_sigma2"]:
        sub = robust[robust["condition"] == condition].copy()
        sub = sub.rename(columns={"output_path": "path", "true_label": "label"})
        sub = sub[["source_image_id", "path", "split", "label", "generator"]]
        sub = sub.sort_values(["split", "source_image_id"]).reset_index(drop=True)
        stop_if(len(sub) != EXPECTED_SOURCES, f"{condition} count {len(sub)}")
        stop_if(
            not original["source_image_id"].astype(str).equals(sub["source_image_id"].astype(str)),
            f"{condition} source alignment",
        )
        frames[condition] = sub

    sub = screen[screen["condition"] == "screenshot_strong"].copy()
    if "true_label" in sub.columns:
        sub["label"] = sub["true_label"].astype(int)
    sub = sub.rename(columns={"output_path": "path"})
    sub = sub[["source_image_id", "path", "split", "label", "generator"]].copy()
    sub = sub.sort_values(["split", "source_image_id"]).reset_index(drop=True)
    stop_if(len(sub) != EXPECTED_SOURCES, f"screenshot count {len(sub)}")
    frames["screenshot_strong"] = sub
    return frames


def load_f0_predictions() -> pd.DataFrame:
    df = pd.read_csv(F0_PRED)
    stop_if(len(df) != EXPECTED_ROWS, f"F0/A2 rows {len(df)}")
    df = df.copy()
    df["regime"] = "F0"
    return df[["regime", "source_image_id", "split", "generator", "label", "condition", "logit", "probability"]]


@torch.no_grad()
def infer_f1(frames: dict[str, pd.DataFrame], device: torch.device, freq: FrequencyTransformV1) -> pd.DataFrame:
    ckpt = torch.load(F1_CKPT, map_location=device, weights_only=False)
    model = FrequencyOnlyCNNV1().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    records = []
    for condition in CONDITIONS:
        rows = frames[condition]
        loader = DataLoader(PathFreqDataset(rows, freq), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        logits_arr = np.empty(len(rows), dtype=float)
        probs_arr = np.empty(len(rows), dtype=float)
        for images, _, indices in tqdm(loader, desc=f"F1/{condition}", leave=False):
            batch_logits = model(images.to(device)).detach().cpu().numpy().reshape(-1)
            batch_probs = 1.0 / (1.0 + np.exp(-batch_logits))
            for i, idx in enumerate(indices.numpy()):
                logits_arr[int(idx)] = float(batch_logits[i])
                probs_arr[int(idx)] = float(batch_probs[i])
        for i, row in rows.iterrows():
            records.append(
                {
                    "regime": "F1",
                    "source_image_id": row["source_image_id"],
                    "split": row["split"],
                    "generator": row["generator"],
                    "label": int(row["label"]),
                    "condition": condition,
                    "logit": float(logits_arr[i]),
                    "probability": float(probs_arr[i]),
                }
            )
    pred = pd.DataFrame(records)
    stop_if(len(pred) != EXPECTED_ROWS, f"F1 rows {len(pred)}")
    pred.to_csv(F1_PRED_OUT, index=False)
    return pred


@torch.no_grad()
def infer_f2(frames: dict[str, pd.DataFrame], device: torch.device, freq: FrequencyTransformV1) -> pd.DataFrame:
    ckpt = torch.load(F2_CKPT, map_location=device, weights_only=False)
    model = RGBFrequencyFusionV1().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    records = []
    for condition in CONDITIONS:
        rows = frames[condition]
        loader = DataLoader(PathFusionDataset(rows, freq), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        logits_arr = np.empty(len(rows), dtype=float)
        probs_arr = np.empty(len(rows), dtype=float)
        for x_rgb, x_freq, _, indices in tqdm(loader, desc=f"F2/{condition}", leave=False):
            batch_logits = model(x_rgb.to(device), x_freq.to(device)).detach().cpu().numpy().reshape(-1)
            batch_probs = 1.0 / (1.0 + np.exp(-batch_logits))
            for i, idx in enumerate(indices.numpy()):
                logits_arr[int(idx)] = float(batch_logits[i])
                probs_arr[int(idx)] = float(batch_probs[i])
        for i, row in rows.iterrows():
            records.append(
                {
                    "regime": "F2",
                    "source_image_id": row["source_image_id"],
                    "split": row["split"],
                    "generator": row["generator"],
                    "label": int(row["label"]),
                    "condition": condition,
                    "logit": float(logits_arr[i]),
                    "probability": float(probs_arr[i]),
                }
            )
    pred = pd.DataFrame(records)
    stop_if(len(pred) != EXPECTED_ROWS, f"F2 rows {len(pred)}")
    pred.to_csv(F2_PRED_OUT, index=False)
    return pred


def assert_alignment(f0: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    for condition in CONDITIONS:
        a = f0[f0["condition"] == condition].sort_values(["split", "source_image_id"]).reset_index(drop=True)
        b = other[other["condition"] == condition].sort_values(["split", "source_image_id"]).reset_index(drop=True)
        stop_if(len(a) != len(b), f"{name}/{condition} length")
        stop_if(not a["source_image_id"].astype(str).equals(b["source_image_id"].astype(str)), f"{name} id align")
        stop_if(not np.array_equal(a["label"].to_numpy(), b["label"].to_numpy()), f"{name} label align")


def compute_metrics(all_preds: dict[str, pd.DataFrame], thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for regime_key, pred in all_preds.items():
        thr = thresholds[regime_key]
        for split in SPLITS:
            for condition in CONDITIONS:
                sub = pred[(pred["split"] == split) & (pred["condition"] == condition)]
                y_true = sub["label"].to_numpy(dtype=int)
                y_prob = sub["probability"].to_numpy(dtype=float)
                thr_m = threshold_metrics(y_true, y_prob, thr)
                rows.append(
                    {
                        "regime": regime_key,
                        "split": split,
                        "condition": condition,
                        "n": int(len(sub)),
                        "threshold": thr,
                        "roc_auc": float(roc_auc_score(y_true, y_prob)),
                        "average_precision": float(average_precision_score(y_true, y_prob)),
                        **thr_m,
                    }
                )
    metrics = pd.DataFrame(rows)
    out_rows = []
    for _, row in metrics.iterrows():
        base = metrics[
            (metrics["regime"] == row["regime"])
            & (metrics["split"] == row["split"])
            & (metrics["condition"] == "original")
        ].iloc[0]
        known = metrics[
            (metrics["regime"] == row["regime"])
            & (metrics["split"] == "known_test")
            & (metrics["condition"] == row["condition"])
        ].iloc[0]
        if row["condition"] == "original":
            delta_auc = 0.0
            delta_ap = 0.0
        else:
            delta_auc = float(row["roc_auc"] - base["roc_auc"])
            delta_ap = float(row["average_precision"] - base["average_precision"])
        gap = float(row["roc_auc"] - known["roc_auc"]) if row["split"] == "unseen_test" else float("nan")
        out_rows.append({**row.to_dict(), "delta_auc": delta_auc, "delta_ap": delta_ap, "generalisation_gap_auc": gap})
    out = pd.DataFrame(out_rows)
    stop_if(len(out) != EXPECTED_METRICS_ROWS, f"metrics rows {len(out)}")
    return out


def compute_generator_recall(all_preds: dict[str, pd.DataFrame], thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for regime_key, pred in all_preds.items():
        thr = thresholds[regime_key]
        for split in SPLITS:
            generators = KNOWN_GENERATORS if split == "known_test" else UNSEEN_GENERATORS
            for condition in CONDITIONS:
                for generator in generators:
                    sub = pred[
                        (pred["split"] == split)
                        & (pred["condition"] == condition)
                        & (pred["generator"] == generator)
                        & (pred["label"] == 1)
                    ]
                    n = int(len(sub))
                    if n == 0:
                        continue
                    detected = int((sub["probability"] >= thr).sum())
                    rows.append(
                        {
                            "regime": regime_key,
                            "split": split,
                            "condition": condition,
                            "generator": generator,
                            "recall": float(detected / n),
                            "sample_count": n,
                            "threshold": thr,
                        }
                    )
    return pd.DataFrame(rows)


def strong_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        for split in SPLITS:
            sub = metrics[
                (metrics["regime"] == regime)
                & (metrics["split"] == split)
                & (metrics["condition"].isin(STRONG_SCORE_CONDITIONS))
            ]
            rows.append(
                {
                    "regime": regime,
                    "split": split,
                    "strong_robust_test_auc": float(sub["roc_auc"].mean()),
                    "strong_robust_test_ap": float(sub["average_precision"].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_paper_table(metrics: pd.DataFrame, scores: pd.DataFrame, params: dict[str, int]) -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        sub = metrics[(metrics["regime"] == regime) & (metrics["split"] == "unseen_test")]
        score = scores[(scores["regime"] == regime) & (scores["split"] == "unseen_test")].iloc[0]
        auc = {r.condition: r.roc_auc for r in sub.itertuples()}
        ap = {r.condition: r.average_precision for r in sub.itertuples()}
        rows.append(
            {
                "regime": regime,
                "parameters": params[regime],
                "original_auc": auc["original"],
                "jpeg50_auc": auc["jpeg_q50"],
                "resize112_auc": auc["resize_112"],
                "blur2_auc": auc["blur_sigma2"],
                "screenshotStrong_auc": auc["screenshot_strong"],
                "StrongRobustTestAUC": score["strong_robust_test_auc"],
                "original_ap": ap["original"],
                "StrongRobustTestAP": score["strong_robust_test_ap"],
            }
        )
    return pd.DataFrame(rows)


def plot_figures(metrics: pd.DataFrame, scores: pd.DataFrame) -> None:
    cond_labels = ["Original", "JPEG50", "Resize112", "Blur2", "ScreenshotStrong"]
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONDITIONS))
    for i, regime in enumerate(REGIMES):
        vals = [
            metrics[
                (metrics["regime"] == regime) & (metrics["split"] == "unseen_test") & (metrics["condition"] == c)
            ].iloc[0]["roc_auc"]
            for c in CONDITIONS
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=regime)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("RQ4 unseen ROC-AUC by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_AUC, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    delta_conds = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
    delta_labels = ["JPEG50", "Resize112", "Blur2", "ScreenshotStrong"]
    x = np.arange(len(delta_conds))
    for i, regime in enumerate(REGIMES):
        vals = [
            metrics[
                (metrics["regime"] == regime) & (metrics["split"] == "unseen_test") & (metrics["condition"] == c)
            ].iloc[0]["delta_auc"]
            for c in delta_conds
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=regime)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(delta_labels)
    ax.set_ylabel("ΔAUC vs own original")
    ax.set_title("RQ4 unseen ΔAUC relative to each regime's original")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DELTA, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    known_vals = [
        scores[(scores.regime == r) & (scores.split == "known_test")].iloc[0].strong_robust_test_auc for r in REGIMES
    ]
    unseen_vals = [
        scores[(scores.regime == r) & (scores.split == "unseen_test")].iloc[0].strong_robust_test_auc for r in REGIMES
    ]
    x = np.arange(len(REGIMES))
    ax.bar(x - 0.18, known_vals, 0.35, label="Known")
    ax.bar(x + 0.18, unseen_vals, 0.35, label="Unseen")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_ylabel("StrongRobustTestAUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("RQ4 StrongRobustTestAUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROBUST, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    known_o = [
        metrics[(metrics.regime == r) & (metrics.split == "known_test") & (metrics.condition == "original")].iloc[0].roc_auc
        for r in REGIMES
    ]
    unseen_o = [
        metrics[(metrics.regime == r) & (metrics.split == "unseen_test") & (metrics.condition == "original")].iloc[0].roc_auc
        for r in REGIMES
    ]
    x = np.arange(len(REGIMES))
    ax.bar(x - 0.18, known_o, 0.35, label="Known")
    ax.bar(x + 0.18, unseen_o, 0.35, label="Unseen")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_ylabel("Original ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("RQ4 known vs unseen original AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_GAP, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, metric_name, title in zip(
        axes,
        ["recall", "specificity"],
        ["Unseen AI recall @ frozen threshold", "Unseen specificity @ frozen threshold"],
    ):
        for regime in REGIMES:
            vals = [
                metrics[
                    (metrics["regime"] == regime) & (metrics["split"] == "unseen_test") & (metrics["condition"] == c)
                ].iloc[0][metric_name]
                for c in CONDITIONS
            ]
            ax.plot(cond_labels, vals, marker="o", label=regime)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_THR, dpi=150)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    thresholds: dict[str, float],
    params: dict[str, int],
) -> None:
    def get(regime, split, cond, col):
        return float(
            metrics[
                (metrics["regime"] == regime) & (metrics["split"] == split) & (metrics["condition"] == cond)
            ].iloc[0][col]
        )

    lines = [
        "RQ4 Stage 24C — Frozen Test Evaluation Report",
        "=" * 60,
        "",
        "1. Hypothesis/context",
        "   Does explicit frequency-domain information provide complementary forensic",
        "   evidence beyond RGB/spatial information for AI-generated image detection?",
        "",
        "2. Sequential-design caveat",
        "   RQ4 follows RQ1–RQ3 on the same Tiny-GenImage pilot benchmark and is partly",
        "   motivated by RQ2 blur/downsampling degradation. It is NOT independent confirmation.",
        "",
        "3. Frozen models",
        f"   F0 RGB A2: params={params['F0']} thr={thresholds['F0']}",
        f"   F1 Frequency-only: params={params['F1']} thr={thresholds['F1']}",
        f"   F2 RGB+Frequency Fusion (PRIMARY): params={params['F2']} thr={thresholds['F2']}",
        "",
        "4–8. Unseen ROC-AUC by condition",
    ]
    for cond in CONDITIONS:
        lines.append(
            f"   {cond}: F0={get('F0','unseen_test',cond,'roc_auc'):.4f} "
            f"F1={get('F1','unseen_test',cond,'roc_auc'):.4f} "
            f"F2={get('F2','unseen_test',cond,'roc_auc'):.4f}"
        )
    lines.append("")
    lines.append("9. StrongRobustTestAUC (unseen)")
    for r in REGIMES:
        s = scores[(scores.regime == r) & (scores.split == "unseen_test")].iloc[0]
        lines.append(f"   {r}: AUC={s.strong_robust_test_auc:.4f} AP={s.strong_robust_test_ap:.4f}")
    lines.append("")
    lines.append("10. F1 frequency-only interpretation: see absolute AUC/AP vs F0.")
    lines.append("11. F2 vs F0 (primary):")
    for cond in CONDITIONS + ["STRONG"]:
        if cond == "STRONG":
            f0 = scores[(scores.regime == "F0") & (scores.split == "unseen_test")].iloc[0].strong_robust_test_auc
            f2 = scores[(scores.regime == "F2") & (scores.split == "unseen_test")].iloc[0].strong_robust_test_auc
            lines.append(f"   StrongRobustTestAUC diff F2-F0 = {f2 - f0:+.4f}")
        else:
            d = get("F2", "unseen_test", cond, "roc_auc") - get("F0", "unseen_test", cond, "roc_auc")
            lines.append(f"   {cond} AUC F2-F0 = {d:+.4f}")
    lines.append("")
    lines.append("12. F2 vs F1:")
    for cond in ["original", "blur_sigma2"]:
        d = get("F2", "unseen_test", cond, "roc_auc") - get("F1", "unseen_test", cond, "roc_auc")
        lines.append(f"   {cond} AUC F2-F1 = {d:+.4f}")
    lines.append("")
    lines.append("13. Original generalisation gaps (unseen-known AUC):")
    for r in REGIMES:
        gap = get(r, "unseen_test", "original", "roc_auc") - get(r, "known_test", "original", "roc_auc")
        lines.append(f"   {r}: {gap:+.4f}")
    lines.append("")
    lines.append("14. Generator observations: see rq4_generator_recall_v1.csv (VQDM diagnostic).")
    lines.append("15. Threshold behaviour: distinguish AUC/AP discrimination from recall/specificity;")
    lines.append("    high recall with collapsed specificity is NOT a robustness gain.")
    lines.append("16. Limitations: one FFT magnitude representation; sequential pilot design;")
    lines.append("    external confirmation pending.")
    lines.append("")
    lines.append("17. Integrity")
    lines.append("    Training after freeze: NO")
    lines.append("    Threshold changes: NO")
    lines.append("    Checkpoint changes: NO")
    lines.append("    Representation changes: NO")
    lines.append("    Test-derived model selection: NO")
    lines.append("    Primary RQ4 intervention changed: NO")
    lines.append("    F0 rerun: NO")
    lines.append("    F1/F2 first frozen test evaluation: YES")
    lines.append("")
    lines.append("Do not claim statistical significance in this stage.")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("=== Stage 24C — Frozen RQ4 test evaluation ===")
    stop_if(not GATE_24B.exists(), "24B gate missing — complete freeze first")
    with open(GATE_24B) as f:
        gate = json.load(f)
    stop_if(gate.get("test_accessed") is not False and gate.get("test_accessed") is not False, "gate anomaly")
    for key in [
        "F0_frozen",
        "F1_checkpoint_frozen",
        "F1_threshold_frozen",
        "F2_checkpoint_frozen",
        "F2_threshold_frozen",
        "F2_marked_primary",
    ]:
        stop_if(not gate.get(key), f"24B gate failed: {key}")

    with open(F0_FROZEN) as f:
        f0_cfg = json.load(f)
    with open(F1_FROZEN) as f:
        f1_cfg = json.load(f)
    with open(F2_FROZEN) as f:
        f2_cfg = json.load(f)

    thresholds = {
        "F0": float(f0_cfg["threshold"]),
        "F1": float(f1_cfg["threshold"]),
        "F2": float(f2_cfg["threshold"]),
    }
    params = {
        "F0": int(f0_cfg["total_parameters"]),
        "F1": int(f1_cfg["total_parameters"]),
        "F2": int(f2_cfg["total_parameters"]),
    }

    device = select_device()
    freq = FrequencyTransformV1.from_json(NORM_PATH)
    frames = load_condition_frames()

    f0 = load_f0_predictions()
    f1 = infer_f1(frames, device, freq)
    f2 = infer_f2(frames, device, freq)
    assert_alignment(f0, f1, "F1")
    assert_alignment(f0, f2, "F2")

    all_preds = {"F0": f0, "F1": f1, "F2": f2}
    metrics = compute_metrics(all_preds, thresholds)
    metrics.to_csv(METRICS_CSV, index=False)
    gen_recall = compute_generator_recall(all_preds, thresholds)
    gen_recall.to_csv(GEN_RECALL_CSV, index=False)
    scores = strong_scores(metrics)
    paper = make_paper_table(metrics, scores, params)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(PAPER_TABLE, index=False)
    plot_figures(metrics, scores)
    write_report(metrics, scores, thresholds, params)

    # mark gate that test was accessed (after freeze; OK)
    gate["test_accessed"] = True
    gate["stage_24c_complete"] = True
    with open(GATE_24B, "w") as f:
        json.dump(gate, f, indent=2)
        f.write("\n")

    print("Saved predictions, metrics, figures, report.")
    print("Stage 24C COMPLETE.")


if __name__ == "__main__":
    main()
