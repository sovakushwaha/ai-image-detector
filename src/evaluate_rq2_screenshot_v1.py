"""Frozen-model evaluation on screenshot_v1 (Stage 22C).

Why this file exists
--------------------
Evaluates four frozen RQ1 detectors on digital screenshot-style composite
transforms. No training, threshold tuning, or model modification.

How to run
----------
    source .venv/bin/activate
    python src/evaluate_rq2_screenshot_v1.py

What to expect
--------------
    results/rq2_screenshot_*_predictions_v1.csv
    results/rq2_screenshot_metrics_v1.csv
    results/rq2_screenshot_generator_recall_v1.csv
    results/rq2_screenshot_evaluation_report_v1.txt
    figures/rq2_screenshot_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
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

from cnn_dataset_v1 import PROJECT_ROOT, EXPECTED_SIZE, load_train_rgb_stats, select_device, stop_if
from efficientnet_b0_binary_v1 import DEFAULT_WEIGHTS as EFF_WEIGHTS, EfficientNetB0BinaryV1
from extract_handcrafted_features import FEATURE_COLUMNS, extract_features
from mobilenet_v3_small_binary_v1 import DEFAULT_WEIGHTS as MOB_WEIGHTS, MobileNetV3SmallBinaryV1
from small_cnn_v1 import SmallCNNV1

MANIFEST_PATH = PROJECT_ROOT / "metadata" / "screenshot_v1_manifest.csv"
RQ2_METRICS_PATH = PROJECT_ROOT / "results" / "rq2_robustness_metrics_v1.csv"
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_TRANSFORMED = 4336
EXPECTED_METRICS_ROWS = 24

SCREENSHOT_CONDITIONS = ["screenshot_mild", "screenshot_strong"]
ALL_CONDITIONS = ["original"] + SCREENSHOT_CONDITIONS
SEVERITY = {
    "original": "reference",
    "screenshot_mild": "mild",
    "screenshot_strong": "strong",
}
COMPARE_CONDITIONS = ["jpeg_q50", "resize_112", "blur_sigma2"]

MODEL_ORDER = ["LogReg", "SmallCNNV1", "MobileNetV3-Small", "EfficientNet-B0"]
MODEL_LABELS = {
    "LogReg": "Handcrafted LogReg",
    "SmallCNNV1": "SmallCNNV1",
    "MobileNetV3-Small": "MobileNetV3-Small",
    "EfficientNet-B0": "EfficientNet-B0",
}

ORIGINAL_JSON = {
    "LogReg": PROJECT_ROOT / "results/logreg_final_baseline_v1.json",
    "SmallCNNV1": PROJECT_ROOT / "results/smallcnn_v1_test_metrics_v1.json",
    "MobileNetV3-Small": PROJECT_ROOT / "results/mobilenet_v3_small_test_metrics_v1.json",
    "EfficientNet-B0": PROJECT_ROOT / "results/efficientnet_b0_test_metrics_v1.json",
}
THRESHOLD_JSON = {
    "LogReg": PROJECT_ROOT / "results/logreg_frozen_baseline_v1.json",
    "SmallCNNV1": PROJECT_ROOT / "results/smallcnn_v1_frozen_config_v1.json",
    "MobileNetV3-Small": PROJECT_ROOT / "results/mobilenet_v3_small_frozen_config_v1.json",
    "EfficientNet-B0": PROJECT_ROOT / "results/efficientnet_b0_frozen_config_v1.json",
}
CHECKPOINT = {
    "SmallCNNV1": PROJECT_ROOT / "models/smallcnn_v1_60ep_best.pt",
    "MobileNetV3-Small": PROJECT_ROOT / "models/mobilenet_v3_small_selected_v1.pt",
    "EfficientNet-B0": PROJECT_ROOT / "models/efficientnet_b0_selected_v1.pt",
}
LOGREG_MODEL = PROJECT_ROOT / "models/logreg_handcrafted_selected_v1.joblib"

PRED_PATHS = {
    "LogReg": PROJECT_ROOT / "results/rq2_screenshot_logreg_predictions_v1.csv",
    "SmallCNNV1": PROJECT_ROOT / "results/rq2_screenshot_smallcnn_predictions_v1.csv",
    "MobileNetV3-Small": PROJECT_ROOT / "results/rq2_screenshot_mobilenet_predictions_v1.csv",
    "EfficientNet-B0": PROJECT_ROOT / "results/rq2_screenshot_efficientnet_predictions_v1.csv",
}

METRICS_CSV = PROJECT_ROOT / "results/rq2_screenshot_metrics_v1.csv"
GENERATOR_CSV = PROJECT_ROOT / "results/rq2_screenshot_generator_recall_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results/rq2_screenshot_evaluation_report_v1.txt"
FIG_UNSEEN_AUC = PROJECT_ROOT / "figures/rq2_screenshot_unseen_auc_v1.png"
FIG_AUC_DELTA = PROJECT_ROOT / "figures/rq2_screenshot_auc_delta_v1.png"
FIG_PRETRAINED_RECALL = PROJECT_ROOT / "figures/rq2_screenshot_pretrained_generator_recall_v1.png"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        stop_if(image.mode != "RGB", f"{path} mode {image.mode} != RGB")
        stop_if(image.size != EXPECTED_SIZE, f"{path} size {image.size} != {EXPECTED_SIZE}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def validate_manifest(manifest: pd.DataFrame) -> None:
    stop_if(len(manifest) != EXPECTED_TRANSFORMED, f"manifest rows {len(manifest)} != {EXPECTED_TRANSFORMED}")
    for cond in SCREENSHOT_CONDITIONS:
        n = int((manifest["condition"] == cond).sum())
        stop_if(n != 2168, f"{cond} count {n} != 2168")
        stop_if(
            int((manifest[manifest["condition"] == cond]["split"] == "known_test").sum()) != 456,
            f"{cond} known count",
        )
        stop_if(
            int((manifest[manifest["condition"] == cond]["split"] == "unseen_test").sum()) != 1712,
            f"{cond} unseen count",
        )
    variants = manifest.groupby("source_image_id")["condition"].nunique()
    stop_if((variants != 2).any(), "source variant count != 2")
    stop_if(manifest["source_image_id"].isna().any(), "missing source_image_id")
    label_col = "true_label" if "true_label" in manifest.columns else "label"
    stop_if(manifest[label_col].isna().any(), "missing labels")
    stop_if(manifest["generator"].isna().any(), "missing generators")


def load_frozen_threshold(model_key: str) -> float:
    cfg = json.loads(THRESHOLD_JSON[model_key].read_text(encoding="utf-8"))
    if model_key == "LogReg":
        return float(cfg["validation_selected_threshold"])
    return float(cfg["threshold"])


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def evaluate_group(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["label"].to_numpy(dtype=int)
    y_prob = df["probability"].to_numpy(dtype=float)
    out = {
        "num_samples": int(len(df)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }
    out.update(threshold_metrics(y_true, y_prob, threshold))
    return out


class ManifestDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["output_path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} bad size")
        tensor = self.transform(rgb)
        label = torch.tensor(float(row["true_label"]), dtype=torch.float32)
        return tensor, label, index


@torch.no_grad()
def neural_predict(model: torch.nn.Module, manifest: pd.DataFrame, transform, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ds = ManifestDataset(manifest, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    probs = np.empty(len(manifest), dtype=float)
    logits_out = np.empty(len(manifest), dtype=float)
    for images, _, indices in tqdm(loader, desc=type(model).__name__, leave=False):
        logits = model(images.to(device)).detach().cpu().numpy().reshape(-1)
        batch_probs = 1.0 / (1.0 + np.exp(-logits))
        for i, idx in enumerate(indices.numpy()):
            logits_out[int(idx)] = float(logits[i])
            probs[int(idx)] = float(batch_probs[i])
    return logits_out, probs


def run_logreg(manifest: pd.DataFrame) -> pd.DataFrame:
    pipeline = joblib.load(LOGREG_MODEL)
    records = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="LogReg features"):
        rgb = load_rgb(PROJECT_ROOT / row.output_path)
        feats = extract_features(rgb)
        x = np.array([[feats[c] for c in FEATURE_COLUMNS]], dtype=float)
        prob = float(pipeline.predict_proba(x)[0, 1])
        records.append(
            {
                "model": "LogReg",
                "source_image_id": row.source_image_id,
                "transformed_path": row.output_path,
                "split": row.split,
                "label": int(row.true_label),
                "generator": row.generator,
                "condition": row.condition,
                "severity": row.severity,
                "probability": prob,
            }
        )
    return pd.DataFrame(records)


def run_neural(
    manifest: pd.DataFrame,
    model_key: str,
    model_cls,
    weights,
    device: torch.device,
    use_imagenet: bool,
) -> pd.DataFrame:
    if use_imagenet:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )
    else:
        stats = load_train_rgb_stats()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[stats["mean_R"], stats["mean_G"], stats["mean_B"]],
                    std=[stats["std_R"], stats["std_G"], stats["std_B"]],
                ),
            ]
        )
    ckpt = torch.load(CHECKPOINT[model_key], map_location=device, weights_only=False)
    model = model_cls(weights=weights).to(device) if weights is not None else model_cls().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logits, probs = neural_predict(model, manifest, transform, device)
    out = manifest.copy()
    out["model"] = model_key
    out["label"] = out["true_label"].astype(int)
    out["transformed_path"] = out["output_path"]
    out["raw_logit"] = logits
    out["probability"] = probs
    return out[
        [
            "model",
            "source_image_id",
            "transformed_path",
            "split",
            "label",
            "generator",
            "condition",
            "severity",
            "raw_logit",
            "probability",
        ]
    ]


def load_original_metrics(model_key: str) -> dict:
    data = json.loads(ORIGINAL_JSON[model_key].read_text(encoding="utf-8"))
    threshold = load_frozen_threshold(model_key)
    out = {}
    for split in ("known_test", "unseen_test"):
        block = data[split]
        fro = block["youden_threshold_metrics"] if model_key == "LogReg" else block["frozen_threshold_metrics"]
        out[split] = {
            "roc_auc": float(block["roc_auc"]),
            "average_precision": float(block["average_precision"]),
            "threshold": threshold,
            **{
                k: float(fro[k])
                for k in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1")
            },
            "fpr": float(fro.get("false_positive_rate", fro.get("fpr", 0.0))),
            "num_samples": int(block["n_samples"]),
        }
    return out


def build_metrics_table(predictions: dict[str, pd.DataFrame], thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    originals = {m: load_original_metrics(m) for m in MODEL_ORDER}

    for model_key in MODEL_ORDER:
        threshold = thresholds[model_key]
        for split in ("known_test", "unseen_test"):
            base = originals[model_key][split]
            rows.append(
                {
                    "model": MODEL_LABELS[model_key],
                    "split": split,
                    "condition": "original",
                    "severity": "reference",
                    "num_samples": base["num_samples"],
                    "roc_auc": base["roc_auc"],
                    "average_precision": base["average_precision"],
                    "accuracy": base["accuracy"],
                    "balanced_accuracy": base["balanced_accuracy"],
                    "precision": base["precision"],
                    "recall": base["recall"],
                    "specificity": base["specificity"],
                    "f1": base["f1"],
                    "fpr": base["fpr"],
                    "threshold": threshold,
                    "delta_auc": 0.0,
                    "delta_ap": 0.0,
                    "delta_balanced_accuracy": 0.0,
                    "delta_recall": 0.0,
                    "delta_f1": 0.0,
                }
            )
            pred = predictions[model_key]
            for condition in SCREENSHOT_CONDITIONS:
                subset = pred[(pred["split"] == split) & (pred["condition"] == condition)]
                m = evaluate_group(subset, threshold)
                rows.append(
                    {
                        "model": MODEL_LABELS[model_key],
                        "split": split,
                        "condition": condition,
                        "severity": SEVERITY[condition],
                        "num_samples": m["num_samples"],
                        "roc_auc": m["roc_auc"],
                        "average_precision": m["average_precision"],
                        "accuracy": m["accuracy"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "specificity": m["specificity"],
                        "f1": m["f1"],
                        "fpr": m["fpr"],
                        "threshold": threshold,
                        "delta_auc": m["roc_auc"] - base["roc_auc"],
                        "delta_ap": m["average_precision"] - base["average_precision"],
                        "delta_balanced_accuracy": m["balanced_accuracy"] - base["balanced_accuracy"],
                        "delta_recall": m["recall"] - base["recall"],
                        "delta_f1": m["f1"] - base["f1"],
                    }
                )
    table = pd.DataFrame(rows)
    stop_if(len(table) != EXPECTED_METRICS_ROWS, f"metrics rows {len(table)} != {EXPECTED_METRICS_ROWS}")
    return table


def build_generator_recall(predictions: dict[str, pd.DataFrame], thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for model_key in ["SmallCNNV1", "MobileNetV3-Small", "EfficientNet-B0"]:
        threshold = thresholds[model_key]
        pred = predictions[model_key]
        for split in ("known_test", "unseen_test"):
            for condition in SCREENSHOT_CONDITIONS:
                subset = pred[(pred["split"] == split) & (pred["condition"] == condition)]
                for generator in sorted(subset["generator"].unique()):
                    ai = subset[(subset["generator"] == generator) & (subset["label"] == 1)]
                    if len(ai) == 0:
                        continue
                    detected = int((ai["probability"] >= threshold).sum())
                    rows.append(
                        {
                            "model": MODEL_LABELS[model_key],
                            "split": split,
                            "generator": generator,
                            "condition": condition,
                            "severity": SEVERITY[condition],
                            "recall": detected / len(ai),
                            "sample_count": int(len(ai)),
                        }
                    )
    return pd.DataFrame(rows)


def compare_with_basic(metrics: pd.DataFrame) -> list[str]:
    if not RQ2_METRICS_PATH.exists():
        return ["- Stage 22B metrics file missing; comparison skipped."]
    basic = pd.read_csv(RQ2_METRICS_PATH)
    lines = []
    for model in MODEL_LABELS.values():
        lines.append(f"- {model} (unseen ΔAUC):")
        for condition in SCREENSHOT_CONDITIONS + COMPARE_CONDITIONS:
            if condition in SCREENSHOT_CONDITIONS:
                src = metrics
            else:
                src = basic
            row = src[
                (src["model"] == model) & (src["split"] == "unseen_test") & (src["condition"] == condition)
            ]
            if row.empty:
                lines.append(f"  {condition}: MISSING")
            else:
                lines.append(f"  {condition}: {float(row.iloc[0]['delta_auc']):+.4f}")
        # qualitative comparison for strong screenshot vs references
        strong = float(
            metrics[
                (metrics["model"] == model)
                & (metrics["split"] == "unseen_test")
                & (metrics["condition"] == "screenshot_strong")
            ].iloc[0]["delta_auc"]
        )
        refs = {}
        for cond in COMPARE_CONDITIONS:
            refs[cond] = float(
                basic[
                    (basic["model"] == model)
                    & (basic["split"] == "unseen_test")
                    & (basic["condition"] == cond)
                ].iloc[0]["delta_auc"]
            )
        # more negative = greater damage
        comparisons = []
        for cond, delta in refs.items():
            if strong < delta - 0.01:
                comparisons.append(f"greater damage than {cond}")
            elif strong > delta + 0.01:
                comparisons.append(f"smaller damage than {cond}")
            else:
                comparisons.append(f"comparable to {cond}")
        lines.append("  interpretation (screenshot_strong vs basic): " + "; ".join(comparisons))
    return lines


def save_figures(metrics: pd.DataFrame, generator_df: pd.DataFrame) -> None:
    # Unseen AUC absolute
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ALL_CONDITIONS))
    width = 0.18
    for i, model in enumerate(MODEL_LABELS.values()):
        sub = metrics[(metrics["model"] == model) & (metrics["split"] == "unseen_test")]
        sub = sub.set_index("condition").reindex(ALL_CONDITIONS)
        ax.bar(x + (i - 1.5) * width, sub["roc_auc"].to_numpy(dtype=float), width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(["Original", "Screenshot Mild", "Screenshot Strong"], rotation=15, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Unseen ROC-AUC: digital screenshot-style approximations")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    FIG_UNSEEN_AUC.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_UNSEEN_AUC, dpi=150)
    plt.close(fig)

    # Delta AUC (known + unseen side by side)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, split, title in zip(
        axes,
        ("known_test", "unseen_test"),
        ("Known ΔAUC", "Unseen ΔAUC"),
    ):
        x = np.arange(len(SCREENSHOT_CONDITIONS))
        for i, model in enumerate(MODEL_LABELS.values()):
            sub = metrics[(metrics["model"] == model) & (metrics["split"] == split)]
            sub = sub.set_index("condition").reindex(SCREENSHOT_CONDITIONS)
            ax.bar(x + (i - 1.5) * width, sub["delta_auc"].to_numpy(dtype=float), width, label=model)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Mild", "Strong"])
        ax.set_title(title)
        ax.set_ylabel("ΔAUC vs original")
    axes[0].legend(fontsize=7)
    fig.suptitle("Screenshot-style ΔAUC (software approximation, not physical recapture)")
    fig.tight_layout()
    fig.savefig(FIG_AUC_DELTA, dpi=150)
    plt.close(fig)

    # Pretrained generator recall heatmap-style bars
    models = ["MobileNetV3-Small", "EfficientNet-B0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, models):
        sub = generator_df[(generator_df["model"] == model) & (generator_df["split"] == "unseen_test")]
        gens = sorted(sub["generator"].unique())
        x = np.arange(len(gens))
        mild = (
            sub[sub["condition"] == "screenshot_mild"].set_index("generator").reindex(gens)["recall"].to_numpy()
        )
        strong = (
            sub[sub["condition"] == "screenshot_strong"].set_index("generator").reindex(gens)["recall"].to_numpy()
        )
        ax.bar(x - 0.18, mild, 0.35, label="Mild")
        ax.bar(x + 0.18, strong, 0.35, label="Strong")
        ax.set_xticks(x)
        ax.set_xticklabels(gens, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(model)
        ax.legend(fontsize=8)
    fig.suptitle("Frozen-threshold unseen AI recall under screenshot-style conditions")
    fig.tight_layout()
    fig.savefig(FIG_PRETRAINED_RECALL, dpi=150)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, generator_df: pd.DataFrame) -> str:
    lines = [
        "RQ2 Screenshot-Style Evaluation Report — Stage 22C",
        "=================================================",
        "",
        "DEFINITION",
        '- Conditions are "software screenshot-style composite approximations".',
        "- They are NOT physical LCD/smartphone camera recaptures.",
        "",
        "GENERATION PROTOCOL",
        "- Sources: original controlled_v1 locked test images (2168)",
        "- Conditions: screenshot_mild (JPEG q85 + 448px centred on 512 canvas + resize 224)",
        "-             screenshot_strong (JPEG q65 + 384px centred on 512 canvas + resize 224)",
        "- Canvas RGB(32,32,32); LANCZOS; PNG final outputs",
        "- No camera noise / perspective / moiré / glare",
        "",
        "INTEGRITY",
        f"- generated images: {EXPECTED_TRANSFORMED}",
        "- failures: 0 (generation gate passed before evaluation)",
        "- labels/generators/splits preserved",
        "",
        "FROZEN MODELS / THRESHOLDS",
    ]
    for model_key in MODEL_ORDER:
        thr = load_frozen_threshold(model_key)
        lines.append(f"- {MODEL_LABELS[model_key]}: threshold={thr:.12f}")

    lines.extend(["", "KNOWN RESULTS"])
    for model in MODEL_LABELS.values():
        for condition in ALL_CONDITIONS:
            row = metrics[
                (metrics["model"] == model)
                & (metrics["split"] == "known_test")
                & (metrics["condition"] == condition)
            ].iloc[0]
            lines.append(
                f"- {model} {condition}: AUC={row.roc_auc:.4f}, AP={row.average_precision:.4f}, "
                f"BalAcc={row.balanced_accuracy:.4f}, Recall={row.recall:.4f}, Spec={row.specificity:.4f}, "
                f"F1={row.f1:.4f}, ΔAUC={row.delta_auc:+.4f}"
            )

    lines.extend(["", "UNSEEN RESULTS"])
    for model in MODEL_LABELS.values():
        for condition in ALL_CONDITIONS:
            row = metrics[
                (metrics["model"] == model)
                & (metrics["split"] == "unseen_test")
                & (metrics["condition"] == condition)
            ].iloc[0]
            lines.append(
                f"- {model} {condition}: AUC={row.roc_auc:.4f}, AP={row.average_precision:.4f}, "
                f"BalAcc={row.balanced_accuracy:.4f}, Recall={row.recall:.4f}, Spec={row.specificity:.4f}, "
                f"F1={row.f1:.4f}, ΔAUC={row.delta_auc:+.4f}"
            )

    lines.extend(["", "DELTAS (unseen)"])
    for model in MODEL_LABELS.values():
        for condition in SCREENSHOT_CONDITIONS:
            row = metrics[
                (metrics["model"] == model)
                & (metrics["split"] == "unseen_test")
                & (metrics["condition"] == condition)
            ].iloc[0]
            lines.append(
                f"- {model} {condition}: ΔAUC={row.delta_auc:+.4f}, ΔAP={row.delta_ap:+.4f}, "
                f"ΔBalAcc={row.delta_balanced_accuracy:+.4f}, ΔRecall={row.delta_recall:+.4f}, "
                f"ΔF1={row.delta_f1:+.4f}"
            )

    lines.extend(["", "GENERATOR BEHAVIOUR (pretrained unseen AI recall)"])
    for model in ["MobileNetV3-Small", "EfficientNet-B0"]:
        sub = generator_df[(generator_df["model"] == model) & (generator_df["split"] == "unseen_test")]
        for condition in SCREENSHOT_CONDITIONS:
            for _, row in sub[sub["condition"] == condition].sort_values("generator").iterrows():
                lines.append(
                    f"- {model} {condition} {row.generator}: recall={row.recall:.3f} (n={row.sample_count})"
                )

    lines.extend(["", "COMPARISON WITH JPEG50 / RESIZE112 / BLUR_SIGMA2 (unseen ΔAUC)"])
    lines.extend(compare_with_basic(metrics))

    lines.extend(
        [
            "",
            "LIMITATIONS",
            "- one fixed neutral canvas",
            "- one fixed rendering geometry per severity",
            "- no real app UI",
            "- no platform-specific pipeline",
            "- no camera/display acquisition effects",
            "- not representative of every screenshot scenario",
            "- actual physical display-camera recapture remains untested",
            "",
            "SCIENTIFIC INTEGRITY",
            "- Model training: NO",
            "- Model refitting: NO",
            "- Threshold change: NO",
            "- Screenshot-specific tuning: NO",
            "- Generator-specific tuning: NO",
            "- Samples removed after results: NO",
            "- RQ1 reopened: NO",
            "- Physical screen recapture claimed: NO",
        ]
    )
    text = "\n".join(lines) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def print_terminal_summary(metrics: pd.DataFrame, failures: int = 0) -> None:
    print("\nSTAGE 22C — SCREENSHOT-STYLE ROBUSTNESS EVALUATION COMPLETE")
    print("\nGENERATION")
    print("Sources: 2168")
    print("Variants per source: 2")
    print("Generated: 4336")
    print(f"Failures: {failures}")

    print("\nUNSEEN ROC-AUC")
    print(f"{'Model':<16}{'Original':>12}{'Screenshot Mild':>18}{'Screenshot Strong':>20}")
    for key, label in MODEL_LABELS.items():
        vals = []
        for cond in ALL_CONDITIONS:
            row = metrics[
                (metrics["model"] == label)
                & (metrics["split"] == "unseen_test")
                & (metrics["condition"] == cond)
            ].iloc[0]
            vals.append(f"{row.roc_auc:.3f}")
        print(f"{key:<16}{vals[0]:>12}{vals[1]:>18}{vals[2]:>20}")

    print("\nUNSEEN DELTA AUC")
    for key, label in MODEL_LABELS.items():
        print(f"{key}:")
        for cond in SCREENSHOT_CONDITIONS:
            row = metrics[
                (metrics["model"] == label)
                & (metrics["split"] == "unseen_test")
                & (metrics["condition"] == cond)
            ].iloc[0]
            tag = "Mild" if cond.endswith("mild") else "Strong"
            print(f"  {tag} {row.delta_auc:+.4f}")

    print("\nModel training: NO")
    print("Threshold changes: NO")
    print("Physical recapture claimed: NO")
    print("\nRQ2 CONTROLLED TRANSFORM EVALUATION COMPLETE")
    print("STOP BEFORE TRANSFORM-AWARE TRAINING.")


def main() -> None:
    print("STAGE 22C — SCREENSHOT-STYLE FROZEN EVALUATION")
    stop_if(not MANIFEST_PATH.exists(), f"missing manifest: {MANIFEST_PATH}")
    manifest = pd.read_csv(MANIFEST_PATH)
    if "true_label" not in manifest.columns:
        manifest["true_label"] = manifest["label"]
    validate_manifest(manifest)
    print(f"Manifest OK: {len(manifest)} rows")

    thresholds = {m: load_frozen_threshold(m) for m in MODEL_ORDER}
    for m, t in thresholds.items():
        print(f"Frozen threshold {m}: {t:.12f}")

    device = select_device()
    print(f"Device: {device}")

    predictions: dict[str, pd.DataFrame] = {}
    predictions["LogReg"] = run_logreg(manifest)
    predictions["SmallCNNV1"] = run_neural(
        manifest, "SmallCNNV1", SmallCNNV1, None, device, use_imagenet=False
    )
    predictions["MobileNetV3-Small"] = run_neural(
        manifest, "MobileNetV3-Small", MobileNetV3SmallBinaryV1, MOB_WEIGHTS, device, use_imagenet=True
    )
    predictions["EfficientNet-B0"] = run_neural(
        manifest, "EfficientNet-B0", EfficientNetB0BinaryV1, EFF_WEIGHTS, device, use_imagenet=True
    )

    ref = set(
        zip(
            predictions["LogReg"]["source_image_id"],
            predictions["LogReg"]["condition"],
            predictions["LogReg"]["split"],
        )
    )
    for model_key, pred in predictions.items():
        stop_if(len(pred) != EXPECTED_TRANSFORMED, f"{model_key} predictions {len(pred)} != {EXPECTED_TRANSFORMED}")
        cur = set(zip(pred["source_image_id"], pred["condition"], pred["split"]))
        stop_if(cur != ref, f"{model_key} does not align to LogReg samples")
        PRED_PATHS[model_key].parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(PRED_PATHS[model_key], index=False)

    metrics = build_metrics_table(predictions, thresholds)
    metrics.to_csv(METRICS_CSV, index=False)
    generator_df = build_generator_recall(predictions, thresholds)
    generator_df.to_csv(GENERATOR_CSV, index=False)
    save_figures(metrics, generator_df)
    report = write_report(metrics, generator_df)
    print(report)
    print_terminal_summary(metrics, failures=0)
    print(f"\nMetrics: {METRICS_CSV}")
    print(f"Report: {REPORT_PATH}")
    print(f"Figures: {FIG_UNSEEN_AUC}, {FIG_AUC_DELTA}, {FIG_PRETRAINED_RECALL}")


if __name__ == "__main__":
    main()
