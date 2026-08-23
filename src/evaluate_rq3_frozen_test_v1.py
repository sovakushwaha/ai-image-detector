"""Frozen RQ3 transformation-aware test evaluation (Stage 23D).

Why this file exists
--------------------
First authorised test evaluation of frozen A1/A2/A3 against A0 under original
and strong robustness conditions. No training, threshold changes, or primary-
candidate changes. Reuses authoritative A0 predictions from RQ1/RQ2.

How to run
----------
    source .venv/bin/activate
    python src/evaluate_rq3_frozen_test_v1.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from mobilenet_v3_small_binary_v1 import DEFAULT_WEIGHTS, MobileNetV3SmallBinaryV1

BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_KNOWN = 456
EXPECTED_UNSEEN = 1712
EXPECTED_SOURCES = 2168
EXPECTED_ROWS_PER_MODEL = 10840
EXPECTED_METRICS_ROWS = 40

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
STRONG_SCORE_CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2"]
SPLITS = ["known_test", "unseen_test"]
KNOWN_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]
UNSEEN_GENERATORS = ["Midjourney", "VQDM", "Wukong"]

SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
ROBUST_MANIFEST = PROJECT_ROOT / "metadata" / "robustness_v1_manifest.csv"
SCREEN_MANIFEST = PROJECT_ROOT / "metadata" / "screenshot_v1_manifest.csv"

A0_FROZEN = PROJECT_ROOT / "results" / "mobilenet_v3_small_frozen_config_v1.json"
A0_KNOWN = PROJECT_ROOT / "results" / "mobilenet_v3_small_known_test_predictions_v1.csv"
A0_UNSEEN = PROJECT_ROOT / "results" / "mobilenet_v3_small_unseen_test_predictions_v1.csv"
A0_RQ2 = PROJECT_ROOT / "results" / "rq2_mobilenet_predictions_v1.csv"
A0_SCREEN = PROJECT_ROOT / "results" / "rq2_screenshot_mobilenet_predictions_v1.csv"

METRICS_CSV = PROJECT_ROOT / "results" / "rq3_test_metrics_v1.csv"
GEN_RECALL_CSV = PROJECT_ROOT / "results" / "rq3_generator_recall_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "rq3_test_evaluation_report_v1.txt"
PAPER_TABLE = PROJECT_ROOT / "paper" / "tables" / "rq3_transformation_aware_comparison.csv"

FIG_AUC = PROJECT_ROOT / "figures" / "rq3_unseen_auc_by_condition_v1.png"
FIG_DELTA = PROJECT_ROOT / "figures" / "rq3_unseen_delta_auc_v1.png"
FIG_ROBUST = PROJECT_ROOT / "figures" / "rq3_robust_test_auc_v1.png"
FIG_THR = PROJECT_ROOT / "figures" / "rq3_pretrained_threshold_behaviour_v1.png"


@dataclass(frozen=True)
class RegimeSpec:
    key: str
    label: str
    regime_id: str
    frozen_config: Path
    checkpoint: Path | None
    predictions_out: Path | None
    primary: bool
    reuse_a0: bool


REGIMES = [
    RegimeSpec(
        key="A0",
        label="Clean",
        regime_id="mobilenet_clean_v1",
        frozen_config=A0_FROZEN,
        checkpoint=None,
        predictions_out=None,
        primary=False,
        reuse_a0=True,
    ),
    RegimeSpec(
        key="A1",
        label="Blur",
        regime_id="mobilenet_blur_aug_v1",
        frozen_config=PROJECT_ROOT / "results" / "rq3_A1_frozen_config_v1.json",
        checkpoint=PROJECT_ROOT / "models" / "mobilenet_blur_aug_selected_v1.pt",
        predictions_out=PROJECT_ROOT / "results" / "rq3_A1_test_predictions_v1.csv",
        primary=False,
        reuse_a0=False,
    ),
    RegimeSpec(
        key="A2",
        label="Resize+JPEG",
        regime_id="mobilenet_resize_jpeg_aug_v1",
        frozen_config=PROJECT_ROOT / "results" / "rq3_A2_frozen_config_v1.json",
        checkpoint=PROJECT_ROOT / "models" / "mobilenet_resize_jpeg_aug_selected_v1.pt",
        predictions_out=PROJECT_ROOT / "results" / "rq3_A2_test_predictions_v1.csv",
        primary=True,
        reuse_a0=False,
    ),
    RegimeSpec(
        key="A3",
        label="Combined",
        regime_id="mobilenet_combined_aug_v1",
        frozen_config=PROJECT_ROOT / "results" / "rq3_A3_frozen_config_v1.json",
        checkpoint=PROJECT_ROOT / "models" / "mobilenet_combined_aug_selected_v1.pt",
        predictions_out=PROJECT_ROOT / "results" / "rq3_A3_test_predictions_v1.csv",
        primary=False,
        reuse_a0=False,
    ),
]


class PathDataset(Dataset):
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


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )


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
    stop_if((original["split"] == "known_test").sum() != EXPECTED_KNOWN, "known original count")
    stop_if((original["split"] == "unseen_test").sum() != EXPECTED_UNSEEN, "unseen original count")
    frames["original"] = original

    for condition in ["jpeg_q50", "resize_112", "blur_sigma2"]:
        sub = robust[robust["condition"] == condition].copy()
        sub = sub.rename(columns={"output_path": "path", "true_label": "label"})
        sub = sub[["source_image_id", "path", "split", "label", "generator"]]
        sub = sub.sort_values(["split", "source_image_id"]).reset_index(drop=True)
        stop_if(len(sub) != EXPECTED_SOURCES, f"{condition} count {len(sub)}")
        stop_if(
            not original["source_image_id"].astype(str).equals(sub["source_image_id"].astype(str)),
            f"{condition} source id alignment failed",
        )
        stop_if(not np.array_equal(original["label"].to_numpy(), sub["label"].to_numpy()), f"{condition} label mismatch")
        stop_if(
            not original["generator"].astype(str).equals(sub["generator"].astype(str)),
            f"{condition} generator mismatch",
        )
        stop_if(not original["split"].astype(str).equals(sub["split"].astype(str)), f"{condition} split mismatch")
        frames[condition] = sub

    sub = screen[screen["condition"] == "screenshot_strong"].copy()
    # screenshot manifest may contain both label and true_label; prefer true_label
    if "true_label" in sub.columns:
        sub["label"] = sub["true_label"].astype(int)
    else:
        sub["label"] = sub["label"].astype(int)
    sub = sub.rename(columns={"output_path": "path"})
    sub = sub[["source_image_id", "path", "split", "label", "generator"]].copy()
    sub = sub.sort_values(["split", "source_image_id"]).reset_index(drop=True)
    stop_if(len(sub) != EXPECTED_SOURCES, f"screenshot_strong count {len(sub)}")
    stop_if(
        not original["source_image_id"].astype(str).equals(sub["source_image_id"].astype(str)),
        "screenshot_strong source id alignment failed",
    )
    frames["screenshot_strong"] = sub
    return frames


def load_a0_predictions() -> pd.DataFrame:
    known = pd.read_csv(A0_KNOWN)
    unseen = pd.read_csv(A0_UNSEEN)
    original = pd.concat([known, unseen], ignore_index=True)
    original = original.rename(
        columns={
            "image_id": "source_image_id",
            "true_label": "label",
            "raw_logit": "logit",
            "ai_probability": "probability",
        }
    )
    original["condition"] = "original"
    original["regime"] = "A0"
    original = original[["regime", "source_image_id", "split", "generator", "label", "condition", "logit", "probability"]]

    rq2 = pd.read_csv(A0_RQ2)
    rq2 = rq2[rq2["condition"].isin(["jpeg_q50", "resize_112", "blur_sigma2"])].copy()
    rq2["regime"] = "A0"
    rq2["logit"] = np.nan  # historical RQ2 file lacks logits; probability is authoritative
    rq2 = rq2[["regime", "source_image_id", "split", "generator", "label", "condition", "logit", "probability"]]

    screen = pd.read_csv(A0_SCREEN)
    screen = screen[screen["condition"] == "screenshot_strong"].copy()
    screen["regime"] = "A0"
    if "raw_logit" in screen.columns:
        screen = screen.rename(columns={"raw_logit": "logit"})
    else:
        screen["logit"] = np.nan
    screen = screen[["regime", "source_image_id", "split", "generator", "label", "condition", "logit", "probability"]]

    out = pd.concat([original, rq2, screen], ignore_index=True)
    stop_if(len(out) != EXPECTED_ROWS_PER_MODEL, f"A0 prediction rows {len(out)}")
    for condition in CONDITIONS:
        sub = out[out["condition"] == condition]
        stop_if(len(sub) != EXPECTED_SOURCES, f"A0 {condition} rows {len(sub)}")
        stop_if((sub["split"] == "known_test").sum() != EXPECTED_KNOWN, f"A0 {condition} known")
        stop_if((sub["split"] == "unseen_test").sum() != EXPECTED_UNSEEN, f"A0 {condition} unseen")
    return out


@torch.no_grad()
def run_regime_inference(
    regime: RegimeSpec,
    frames: dict[str, pd.DataFrame],
    device: torch.device,
) -> pd.DataFrame:
    stop_if(regime.checkpoint is None or not regime.checkpoint.exists(), f"missing {regime.checkpoint}")
    ckpt = torch.load(regime.checkpoint, map_location=device, weights_only=False)
    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = build_transform()

    records: list[dict] = []
    for condition in CONDITIONS:
        rows = frames[condition]
        loader = DataLoader(PathDataset(rows, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        probs = np.empty(len(rows), dtype=float)
        logits = np.empty(len(rows), dtype=float)
        for images, _, indices in tqdm(loader, desc=f"{regime.key}/{condition}", leave=False):
            batch_logits = model(images.to(device)).detach().cpu().numpy().reshape(-1)
            batch_probs = 1.0 / (1.0 + np.exp(-batch_logits))
            for i, idx in enumerate(indices.numpy()):
                logits[int(idx)] = float(batch_logits[i])
                probs[int(idx)] = float(batch_probs[i])
        for i, row in rows.iterrows():
            records.append(
                {
                    "regime": regime.key,
                    "source_image_id": row["source_image_id"],
                    "split": row["split"],
                    "generator": row["generator"],
                    "label": int(row["label"]),
                    "condition": condition,
                    "logit": float(logits[i]),
                    "probability": float(probs[i]),
                }
            )
    pred = pd.DataFrame(records)
    stop_if(len(pred) != EXPECTED_ROWS_PER_MODEL, f"{regime.key} rows {len(pred)}")
    known_rows = ((pred["split"] == "known_test")).sum()
    unseen_rows = ((pred["split"] == "unseen_test")).sum()
    stop_if(known_rows != EXPECTED_KNOWN * 5, f"{regime.key} known rows {known_rows}")
    stop_if(unseen_rows != EXPECTED_UNSEEN * 5, f"{regime.key} unseen rows {unseen_rows}")
    if regime.predictions_out is not None:
        regime.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(regime.predictions_out, index=False)
    return pred


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
    # add deltas vs original within regime×split
    out_rows = []
    for _, row in metrics.iterrows():
        base = metrics[
            (metrics["regime"] == row["regime"])
            & (metrics["split"] == row["split"])
            & (metrics["condition"] == "original")
        ].iloc[0]
        if row["condition"] == "original":
            deltas = {
                "delta_auc": 0.0,
                "delta_ap": 0.0,
                "delta_balanced_accuracy": 0.0,
                "delta_recall": 0.0,
                "delta_specificity": 0.0,
                "delta_f1": 0.0,
            }
        else:
            deltas = {
                "delta_auc": float(row["roc_auc"] - base["roc_auc"]),
                "delta_ap": float(row["average_precision"] - base["average_precision"]),
                "delta_balanced_accuracy": float(row["balanced_accuracy"] - base["balanced_accuracy"]),
                "delta_recall": float(row["recall"] - base["recall"]),
                "delta_specificity": float(row["specificity"] - base["specificity"]),
                "delta_f1": float(row["f1"] - base["f1"]),
            }
        out_rows.append({**row.to_dict(), **deltas})
    out = pd.DataFrame(out_rows)
    stop_if(len(out) != EXPECTED_METRICS_ROWS, f"metrics rows {len(out)}")
    return out


def compute_generator_recall(all_preds: dict[str, pd.DataFrame], thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for regime_key, pred in all_preds.items():
        thr = thresholds[regime_key]
        for split in SPLITS:
            generators = KNOWN_GENERATORS if split == "known_test" else UNSEEN_GENERATORS
            # also include known gens if present in unseen? No — known gens only in known_test
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


def strong_robust_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["A0", "A1", "A2", "A3"]:
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


def make_paper_table(metrics: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["A0", "A1", "A2", "A3"]:
        sub = metrics[(metrics["regime"] == regime) & (metrics["split"] == "unseen_test")]
        score = scores[(scores["regime"] == regime) & (scores["split"] == "unseen_test")].iloc[0]
        auc_map = {r.condition: r.roc_auc for r in sub.itertuples()}
        rows.append(
            {
                "regime": regime,
                "original_auc": auc_map["original"],
                "jpeg50_auc": auc_map["jpeg_q50"],
                "resize112_auc": auc_map["resize_112"],
                "blur2_auc": auc_map["blur_sigma2"],
                "screenshot_strong_auc": auc_map["screenshot_strong"],
                "strong_robust_test_auc": score["strong_robust_test_auc"],
            }
        )
    return pd.DataFrame(rows)


def plot_figures(metrics: pd.DataFrame, scores: pd.DataFrame) -> None:
    FIGURES = PROJECT_ROOT / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)
    regimes = ["A0", "A1", "A2", "A3"]
    cond_labels = ["Original", "JPEG50", "Resize112", "Blur2", "ScreenshotStrong"]
    conditions = CONDITIONS

    # Unseen AUC by condition
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    width = 0.18
    for i, regime in enumerate(regimes):
        vals = [
            metrics[
                (metrics["regime"] == regime)
                & (metrics["split"] == "unseen_test")
                & (metrics["condition"] == c)
            ].iloc[0]["roc_auc"]
            for c in conditions
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=regime)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("RQ3 unseen ROC-AUC by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_AUC, dpi=150)
    plt.close(fig)

    # Unseen delta AUC
    fig, ax = plt.subplots(figsize=(10, 5))
    delta_conds = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
    delta_labels = ["JPEG50", "Resize112", "Blur2", "ScreenshotStrong"]
    x = np.arange(len(delta_conds))
    for i, regime in enumerate(regimes):
        vals = [
            metrics[
                (metrics["regime"] == regime)
                & (metrics["split"] == "unseen_test")
                & (metrics["condition"] == c)
            ].iloc[0]["delta_auc"]
            for c in delta_conds
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=regime)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(delta_labels)
    ax.set_ylabel("ΔAUC vs own original")
    ax.set_title("RQ3 unseen ΔAUC relative to each regime's original")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DELTA, dpi=150)
    plt.close(fig)

    # Strong robust test AUC
    fig, ax = plt.subplots(figsize=(7, 4))
    known_vals = [scores[(scores.regime == r) & (scores.split == "known_test")].iloc[0].strong_robust_test_auc for r in regimes]
    unseen_vals = [scores[(scores.regime == r) & (scores.split == "unseen_test")].iloc[0].strong_robust_test_auc for r in regimes]
    x = np.arange(len(regimes))
    ax.bar(x - 0.18, known_vals, 0.35, label="Known")
    ax.bar(x + 0.18, unseen_vals, 0.35, label="Unseen")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("StrongRobustTestAUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("RQ3 StrongRobustTestAUC (orig+jpeg50+resize112+blur2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROBUST, dpi=150)
    plt.close(fig)

    # Threshold behaviour: recall / specificity unseen
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, metric_name, title in zip(
        axes,
        ["recall", "specificity"],
        ["Unseen AI recall @ frozen threshold", "Unseen specificity @ frozen threshold"],
    ):
        for i, regime in enumerate(regimes):
            vals = [
                metrics[
                    (metrics["regime"] == regime)
                    & (metrics["split"] == "unseen_test")
                    & (metrics["condition"] == c)
                ].iloc[0][metric_name]
                for c in conditions
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


def fmt(v: float) -> str:
    return f"{v:.6f}"


def write_report(
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    gen_recall: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    def get(regime, split, condition, col):
        return float(
            metrics[
                (metrics["regime"] == regime)
                & (metrics["split"] == split)
                & (metrics["condition"] == condition)
            ].iloc[0][col]
        )

    lines = [
        "RQ3 Frozen Test Evaluation Report — Stage 23D",
        "=============================================",
        "",
        "1. SEQUENTIAL-DESIGN CAVEAT",
        "RQ3 augmentation choices were motivated by RQ2 findings on the same pilot",
        "benchmark. This evaluation is a sequential follow-up experiment, NOT fully",
        "independent confirmatory evidence on the same transformed test samples.",
        "Stronger confirmation requires external/new-generator evaluation.",
        "A2 was selected as the primary candidate from validation BEFORE this test",
        "and remains primary regardless of test outcomes.",
        "",
        "2. FROZEN MODELS / THRESHOLDS",
    ]
    for regime in REGIMES:
        lines.append(f"- {regime.key} ({regime.regime_id}): threshold={thresholds[regime.key]:.12f}")
    lines.extend(["", "3. ORIGINAL PERFORMANCE"])
    for split in SPLITS:
        lines.append(f"[{split}]")
        for regime in ["A0", "A1", "A2", "A3"]:
            lines.append(
                f"  {regime}: AUC={get(regime, split, 'original', 'roc_auc'):.6f} "
                f"AP={get(regime, split, 'original', 'average_precision'):.6f} "
                f"BalAcc={get(regime, split, 'original', 'balanced_accuracy'):.6f}"
            )
    for section, condition in [
        ("4. JPEG50", "jpeg_q50"),
        ("5. RESIZE112", "resize_112"),
        ("6. BLUR2", "blur_sigma2"),
        ("7. SCREENSHOT STRONG", "screenshot_strong"),
    ]:
        lines.extend(["", section])
        for split in SPLITS:
            lines.append(f"[{split}]")
            for regime in ["A0", "A1", "A2", "A3"]:
                lines.append(
                    f"  {regime}: AUC={get(regime, split, condition, 'roc_auc'):.6f} "
                    f"ΔAUC={get(regime, split, condition, 'delta_auc'):+.6f} "
                    f"AP={get(regime, split, condition, 'average_precision'):.6f} "
                    f"Rec={get(regime, split, condition, 'recall'):.6f} "
                    f"Spec={get(regime, split, condition, 'specificity'):.6f}"
                )

    lines.extend(["", "8. STRONG ROBUST TEST SCORE"])
    for _, row in scores.iterrows():
        lines.append(
            f"{row.regime} {row.split}: StrongRobustTestAUC={row.strong_robust_test_auc:.6f} "
            f"AP={row.strong_robust_test_ap:.6f}"
        )

    lines.extend(["", "9. PRIMARY A2 VS A0"])
    for split in SPLITS:
        lines.append(f"[{split}]")
        for condition in CONDITIONS:
            d = get("A2", split, condition, "roc_auc") - get("A0", split, condition, "roc_auc")
            dap = get("A2", split, condition, "average_precision") - get("A0", split, condition, "average_precision")
            lines.append(f"  {condition}: ΔAUC(A2−A0)={d:+.6f} ΔAP={dap:+.6f}")
        s2 = scores[(scores.regime == "A2") & (scores.split == split)].iloc[0]
        s0 = scores[(scores.regime == "A0") & (scores.split == split)].iloc[0]
        lines.append(
            f"  StrongRobustTestAUC difference: {s2.strong_robust_test_auc - s0.strong_robust_test_auc:+.6f}"
        )

    lines.extend(
        [
            "",
            "10. A1/A2/A3 ABLATION COMPARISON",
            "A1 (blur-aware): inspect Blur2 transfer vs A0.",
            f"  Unseen Blur2 ΔAUC(A1−A0)="
            f"{get('A1','unseen_test','blur_sigma2','roc_auc')-get('A0','unseen_test','blur_sigma2','roc_auc'):+.6f}",
            "A2 (resize+JPEG): inspect JPEG50/Resize112 vs A0.",
            f"  Unseen JPEG50 ΔAUC(A2−A0)="
            f"{get('A2','unseen_test','jpeg_q50','roc_auc')-get('A0','unseen_test','jpeg_q50','roc_auc'):+.6f}",
            f"  Unseen Resize112 ΔAUC(A2−A0)="
            f"{get('A2','unseen_test','resize_112','roc_auc')-get('A0','unseen_test','resize_112','roc_auc'):+.6f}",
            "A3 (combined): compare StrongRobustTestAUC vs A2.",
            f"  Unseen StrongRobust A3−A2="
            f"{scores[(scores.regime=='A3')&(scores.split=='unseen_test')].iloc[0].strong_robust_test_auc - scores[(scores.regime=='A2')&(scores.split=='unseen_test')].iloc[0].strong_robust_test_auc:+.6f}",
            "",
            "11. CROSS-TRANSFORMATION EFFECTS",
            "ScreenshotStrong was not explicitly included as a training augmentation.",
            f"Unseen ScreenshotStrong ΔAUC(A1−A0)="
            f"{get('A1','unseen_test','screenshot_strong','roc_auc')-get('A0','unseen_test','screenshot_strong','roc_auc'):+.6f}",
            f"Unseen ScreenshotStrong ΔAUC(A2−A0)="
            f"{get('A2','unseen_test','screenshot_strong','roc_auc')-get('A0','unseen_test','screenshot_strong','roc_auc'):+.6f}",
            f"Unseen ScreenshotStrong ΔAUC(A3−A0)="
            f"{get('A3','unseen_test','screenshot_strong','roc_auc')-get('A0','unseen_test','screenshot_strong','roc_auc'):+.6f}",
            f"A2 Blur2 (no blur augmentation): ΔAUC(A2−A0)="
            f"{get('A2','unseen_test','blur_sigma2','roc_auc')-get('A0','unseen_test','blur_sigma2','roc_auc'):+.6f}",
            "",
            "12. GENERATOR-SPECIFIC OBSERVATIONS",
            "Unseen AI recall at frozen thresholds:",
        ]
    )
    for regime in ["A0", "A1", "A2", "A3"]:
        for generator in UNSEEN_GENERATORS:
            vals = []
            for condition in CONDITIONS:
                g = gen_recall[
                    (gen_recall.regime == regime)
                    & (gen_recall.split == "unseen_test")
                    & (gen_recall.condition == condition)
                    & (gen_recall.generator == generator)
                ]
                if len(g):
                    vals.append(f"{condition}={g.iloc[0].recall:.3f}(n={int(g.iloc[0].sample_count)})")
            lines.append(f"  {regime} {generator}: " + "; ".join(vals))

    lines.extend(
        [
            "",
            "13. THRESHOLD BEHAVIOUR",
            "Distinguish discrimination (AUC/AP) from operating-point (recall/specificity).",
            "High AI recall with collapsed specificity is NOT described as robustness gain.",
        ]
    )
    for regime in ["A0", "A1", "A2", "A3"]:
        lines.append(f"{regime} unseen recall/spec:")
        for condition in CONDITIONS:
            lines.append(
                f"  {condition}: recall={get(regime,'unseen_test',condition,'recall'):.3f} "
                f"spec={get(regime,'unseen_test',condition,'specificity'):.3f} "
                f"FPR={get(regime,'unseen_test',condition,'fpr'):.3f}"
            )

    lines.extend(
        [
            "",
            "14. LIMITATIONS",
            "- Sequential follow-up to RQ2 on the same pilot benchmark.",
            "- No statistical significance testing in Stage 23D.",
            "- Screenshot condition is a digital approximation, not physical recapture.",
            "- Clean-performance trade-offs may accompany robustness gains.",
            "",
            "15. SCIENTIFIC INTEGRITY",
            "Model training after Stage 23C: NO",
            "Checkpoint changes: NO",
            "Threshold changes: NO",
            "Test-derived threshold selection: NO",
            "Primary candidate changed after test: NO",
            "Transformation-specific thresholds: NO",
            "Generator-specific thresholds: NO",
            "Samples removed after results: NO",
            "A0 rerun unnecessarily: NO",
            "A1/A2/A3 first frozen test evaluation: YES",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("STAGE 23D — FROZEN RQ3 TRANSFORMATION-AWARE TEST EVALUATION")
    device = select_device()
    print(f"Device: {device}")

    frames = load_condition_frames()
    thresholds = {}
    for regime in REGIMES:
        cfg = json.loads(regime.frozen_config.read_text(encoding="utf-8"))
        thresholds[regime.key] = float(cfg["threshold"])
        if regime.key == "A2":
            stop_if(not cfg.get("primary_rq3_candidate", False), "A2 not marked primary")

    print("A0 thresholds (reused historical predictions):", f"{thresholds['A0']:.12f}")
    all_preds: dict[str, pd.DataFrame] = {}
    all_preds["A0"] = load_a0_predictions()
    print(f"Loaded A0 historical predictions: {len(all_preds['A0'])} rows")

    for regime in REGIMES:
        if regime.reuse_a0:
            continue
        print(f"Running inference for {regime.key}...")
        all_preds[regime.key] = run_regime_inference(regime, frames, device)

    metrics = compute_metrics(all_preds, thresholds)
    scores = strong_robust_scores(metrics)
    gen_recall = compute_generator_recall(all_preds, thresholds)
    paper_table = make_paper_table(metrics, scores)

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_CSV, index=False)
    gen_recall.to_csv(GEN_RECALL_CSV, index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    paper_table.to_csv(PAPER_TABLE, index=False)
    plot_figures(metrics, scores)
    write_report(metrics, scores, gen_recall, thresholds)

    def auc(regime, split, condition):
        return float(
            metrics[
                (metrics.regime == regime) & (metrics.split == split) & (metrics.condition == condition)
            ].iloc[0].roc_auc
        )

    print("\nSTAGE 23D — RQ3 FROZEN TEST EVALUATION COMPLETE")
    print("\nUNSEEN ROC-AUC")
    print(f"{'Regime':<12} {'Original':>9} {'JPEG50':>9} {'Resize112':>10} {'Blur2':>9} {'ScreenshotStrong':>16}")
    for regime, label in [("A0", "A0 Clean"), ("A1", "A1 Blur"), ("A2", "A2 ResizeJ"), ("A3", "A3 Combined")]:
        print(
            f"{label:<12} "
            f"{auc(regime,'unseen_test','original'):9.6f} "
            f"{auc(regime,'unseen_test','jpeg_q50'):9.6f} "
            f"{auc(regime,'unseen_test','resize_112'):10.6f} "
            f"{auc(regime,'unseen_test','blur_sigma2'):9.6f} "
            f"{auc(regime,'unseen_test','screenshot_strong'):16.6f}"
        )

    print("\nUNSEEN STRONG ROBUST TEST AUC")
    for regime in ["A0", "A1", "A2", "A3"]:
        v = scores[(scores.regime == regime) & (scores.split == "unseen_test")].iloc[0].strong_robust_test_auc
        print(f"{regime}: {v:.6f}")

    print("\nPRIMARY A2 VS A0")
    for condition, name in [
        ("original", "Original"),
        ("jpeg_q50", "JPEG50"),
        ("resize_112", "Resize112"),
        ("blur_sigma2", "Blur2"),
        ("screenshot_strong", "ScreenshotStrong"),
    ]:
        d = auc("A2", "unseen_test", condition) - auc("A0", "unseen_test", condition)
        print(f"{name} ΔAUC: {d:+.6f}")
    s2 = scores[(scores.regime == "A2") & (scores.split == "unseen_test")].iloc[0].strong_robust_test_auc
    s0 = scores[(scores.regime == "A0") & (scores.split == "unseen_test")].iloc[0].strong_robust_test_auc
    print(f"StrongRobustTestAUC difference: {s2 - s0:+.6f}")

    print("\nKNOWN STRONG ROBUST TEST AUC")
    for regime in ["A0", "A1", "A2", "A3"]:
        v = scores[(scores.regime == regime) & (scores.split == "known_test")].iloc[0].strong_robust_test_auc
        print(f"{regime}: {v:.6f}")

    print("\nPRIMARY CANDIDATE:\nA2 Resize+JPEG")
    print("\nPrimary candidate changed after test:\nNO")
    print("\nModel retraining:\nNO")
    print("\nThreshold changes:\nNO")
    print("\nRQ3 TEST EVALUATION:\nCOMPLETE")
    print("\nSTATISTICAL PAIRED ANALYSIS:\nPENDING")
    print("\nSTOP.")


if __name__ == "__main__":
    main()
