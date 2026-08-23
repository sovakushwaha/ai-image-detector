"""Frozen-model robustness evaluation on robustness_v1 (Stage 22B).

Why this file exists
--------------------
Evaluates the four frozen RQ1 detectors on controlled transformed test images.
No training, threshold tuning, or model modification.

How to run
----------
    source .venv/bin/activate
    python src/evaluate_rq2_robustness_v1.py

What to expect
--------------
    results/rq2_*_predictions_v1.csv
    results/rq2_robustness_metrics_v1.csv
    results/rq2_generator_recall_v1.csv
    results/rq2_robustness_evaluation_report_v1.txt
    figures/rq2_*.png
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

MANIFEST_PATH = PROJECT_ROOT / "metadata" / "robustness_v1_manifest.csv"
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_TRANSFORMED = 17344
EXPECTED_METRICS_ROWS = 72

TRANSFORM_CONDITIONS = [
    "jpeg_q75",
    "jpeg_q50",
    "crop_90",
    "crop_75",
    "resize_160",
    "resize_112",
    "blur_sigma1",
    "blur_sigma2",
]
ALL_CONDITIONS = ["original"] + TRANSFORM_CONDITIONS
SEVERITY = {
    "original": "reference",
    "jpeg_q75": "mild",
    "jpeg_q50": "strong",
    "crop_90": "mild",
    "crop_75": "strong",
    "resize_160": "mild",
    "resize_112": "strong",
    "blur_sigma1": "mild",
    "blur_sigma2": "strong",
}
TRANSFORM_FAMILIES = {
    "jpeg": ("jpeg_q75", "jpeg_q50"),
    "crop": ("crop_90", "crop_75"),
    "resize": ("resize_160", "resize_112"),
    "blur": ("blur_sigma1", "blur_sigma2"),
}

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
    m: PROJECT_ROOT / "results" / f"rq2_{m.lower().replace('-', '').replace('v3small', 'mobilenet').replace('efficientnetb0', 'efficientnet').replace('smallcnnv1', 'smallcnn').replace('logreg', 'logreg')}_predictions_v1.csv"
    for m in MODEL_ORDER
}
# Fix explicit paths
PRED_PATHS = {
    "LogReg": PROJECT_ROOT / "results/rq2_logreg_predictions_v1.csv",
    "SmallCNNV1": PROJECT_ROOT / "results/rq2_smallcnn_predictions_v1.csv",
    "MobileNetV3-Small": PROJECT_ROOT / "results/rq2_mobilenet_predictions_v1.csv",
    "EfficientNet-B0": PROJECT_ROOT / "results/rq2_efficientnet_predictions_v1.csv",
}

METRICS_CSV = PROJECT_ROOT / "results/rq2_robustness_metrics_v1.csv"
GENERATOR_CSV = PROJECT_ROOT / "results/rq2_generator_recall_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results/rq2_robustness_evaluation_report_v1.txt"
FIG_UNSEEN_AUC = PROJECT_ROOT / "figures/rq2_unseen_auc_by_condition_v1.png"
FIG_UNSEEN_DELTA = PROJECT_ROOT / "figures/rq2_unseen_auc_delta_v1.png"
FIG_KNOWN_DELTA = PROJECT_ROOT / "figures/rq2_known_auc_delta_v1.png"
FIG_PRETRAINED_RECALL = PROJECT_ROOT / "figures/rq2_pretrained_unseen_recall_v1.png"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_robustness_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        stop_if(image.mode != "RGB", f"{path} mode {image.mode} != RGB")
        stop_if(image.size != EXPECTED_SIZE, f"{path} size {image.size} != {EXPECTED_SIZE}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def validate_manifest(manifest: pd.DataFrame) -> None:
    stop_if(len(manifest) != EXPECTED_TRANSFORMED, f"manifest rows {len(manifest)} != {EXPECTED_TRANSFORMED}")
    for cond in TRANSFORM_CONDITIONS:
        n = int((manifest["condition"] == cond).sum())
        stop_if(n != 2168, f"{cond} count {n} != 2168")
        stop_if(int((manifest[manifest["condition"] == cond]["split"] == "known_test").sum()) != 456, f"{cond} known count")
        stop_if(int((manifest[manifest["condition"] == cond]["split"] == "unseen_test").sum()) != 1712, f"{cond} unseen count")
    variants = manifest.groupby("source_image_id")["condition"].nunique()
    stop_if((variants != 8).any(), "source variant count != 8")
    stop_if(manifest["source_image_id"].isna().any(), "missing source_image_id")


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
def neural_predict(model: torch.nn.Module, manifest: pd.DataFrame, transform, device: torch.device) -> np.ndarray:
    model.eval()
    ds = ManifestDataset(manifest, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    probs = np.empty(len(manifest), dtype=float)
    for images, _, indices in tqdm(loader, desc=type(model).__name__, leave=False):
        logits = model(images.to(device)).cpu().numpy()
        batch_probs = 1.0 / (1.0 + np.exp(-logits))
        for i, idx in enumerate(indices.numpy()):
            probs[int(idx)] = float(batch_probs[i])
    return probs


def run_logreg(manifest: pd.DataFrame, threshold: float) -> pd.DataFrame:
    pipeline = joblib.load(LOGREG_MODEL)
    records = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="LogReg features"):
        rgb = load_robustness_rgb(PROJECT_ROOT / row.output_path)
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


def run_smallcnn(manifest: pd.DataFrame, device: torch.device) -> pd.DataFrame:
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
    ckpt = torch.load(CHECKPOINT["SmallCNNV1"], map_location=device, weights_only=False)
    model = SmallCNNV1().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    probs = neural_predict(model, manifest, transform, device)
    out = manifest.copy()
    out["model"] = "SmallCNNV1"
    out["label"] = out["true_label"].astype(int)
    out["transformed_path"] = out["output_path"]
    out["probability"] = probs
    return out[
        ["model", "source_image_id", "transformed_path", "split", "label", "generator", "condition", "severity", "probability"]
    ]


def run_imagenet_model(
    manifest: pd.DataFrame,
    model_key: str,
    model_cls,
    weights,
    device: torch.device,
) -> pd.DataFrame:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    ckpt = torch.load(CHECKPOINT[model_key], map_location=device, weights_only=False)
    model = model_cls(weights=weights).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    probs = neural_predict(model, manifest, transform, device)
    out = manifest.copy()
    out["model"] = model_key
    out["label"] = out["true_label"].astype(int)
    out["transformed_path"] = out["output_path"]
    out["probability"] = probs
    return out[
        ["model", "source_image_id", "transformed_path", "split", "label", "generator", "condition", "severity", "probability"]
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
            **{k: float(fro[k]) for k in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "false_positive_rate")},
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
            for condition in TRANSFORM_CONDITIONS:
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
            for condition in ALL_CONDITIONS:
                if condition == "original":
                    continue
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


def summarize_extremes(metrics: pd.DataFrame) -> dict:
    summary = {}
    for model in metrics["model"].unique():
        summary[model] = {}
        for split in ("known_test", "unseen_test"):
            sub = metrics[(metrics["model"] == model) & (metrics["split"] == split) & (metrics["condition"] != "original")]
            summary[model][split] = {
                "largest_auc_drop": sub.loc[sub["delta_auc"].idxmin(), ["condition", "delta_auc"]].to_dict(),
                "largest_ap_drop": sub.loc[sub["delta_ap"].idxmin(), ["condition", "delta_ap"]].to_dict(),
                "largest_recall_drop": sub.loc[sub["delta_recall"].idxmin(), ["condition", "delta_recall"]].to_dict(),
                "most_robust_auc": sub.loc[sub["delta_auc"].idxmax(), ["condition", "delta_auc"]].to_dict(),
                "most_damaging_auc": sub.loc[sub["delta_auc"].idxmin(), ["condition", "delta_auc"]].to_dict(),
            }
    return summary


def save_auc_figure(metrics: pd.DataFrame, split: str, path: Path, value_col: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ALL_CONDITIONS))
    width = 0.18
    for i, model in enumerate(MODEL_LABELS.values()):
        sub = metrics[(metrics["model"] == model) & (metrics["split"] == split)]
        sub = sub.set_index("condition").reindex(ALL_CONDITIONS)
        vals = sub[value_col].to_numpy(dtype=float)
        ax.bar(x + (i - 1.5) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_CONDITIONS, rotation=35, ha="right")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(title)
    ax.legend(fontsize=8)
    if value_col.startswith("delta"):
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pretrained_recall_figure(generator_df: pd.DataFrame) -> None:
    models = ["MobileNetV3-Small", "EfficientNet-B0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, models):
        sub = generator_df[(generator_df["model"] == model) & (generator_df["split"] == "unseen_test")]
        pivot = sub.groupby("condition")["recall"].mean().reindex(TRANSFORM_CONDITIONS)
        ax.bar(pivot.index, pivot.values, color="#DD8452")
        ax.set_title(f"{model} unseen mean AI recall")
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
    fig.suptitle("Frozen-threshold unseen AI recall by transformation (mean across generators)")
    fig.tight_layout()
    fig.savefig(FIG_PRETRAINED_RECALL, dpi=150)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, generator_df: pd.DataFrame, extremes: dict) -> str:
    lines = [
        "RQ2 Robustness Evaluation Report — Stage 22B",
        "===========================================",
        "",
        "PROTOCOL",
        "- frozen models: LogReg, SmallCNNV1, MobileNetV3-Small, EfficientNet-B0",
        "- robustness dataset: robustness_v1 (17,344 transformed images)",
        "- original reference: existing frozen controlled_v1 test evaluations",
        "- thresholds: each model's own validation-selected Youden threshold",
        "- no retraining, no threshold tuning on transformed data",
        "",
        "ORIGINAL REFERENCE PERFORMANCE",
    ]
    for model in MODEL_LABELS.values():
        for split in ("known_test", "unseen_test"):
            row = metrics[(metrics["model"] == model) & (metrics["split"] == split) & (metrics["condition"] == "original")].iloc[0]
            lines.append(f"- {model} {split}: AUC={row.roc_auc:.4f}, AP={row.average_precision:.4f}")

    for family, (mild, strong) in TRANSFORM_FAMILIES.items():
        lines.extend(["", f"{family.upper()} RESULTS (unseen AUC deltas)"])
        for model in MODEL_LABELS.values():
            m = metrics[(metrics["model"] == model) & (metrics["split"] == "unseen_test") & (metrics["condition"] == mild)].iloc[0]
            s = metrics[(metrics["model"] == model) & (metrics["split"] == "unseen_test") & (metrics["condition"] == strong)].iloc[0]
            lines.append(
                f"- {model}: {mild} delta={m.delta_auc:+.4f}, {strong} delta={s.delta_auc:+.4f}"
            )

    lines.extend(["", "LARGEST UNSEEN AUC DROP BY MODEL"])
    for model, splits in extremes.items():
        drop = splits["unseen_test"]["largest_auc_drop"]
        lines.append(f"- {model}: {drop['condition']} ({drop['delta_auc']:+.4f})")

    lines.extend(["", "VQDM UNSEEN RECALL (MobileNet / EfficientNet, frozen threshold)"])
    for model in ["MobileNetV3-Small", "EfficientNet-B0"]:
        sub = generator_df[(generator_df["model"] == model) & (generator_df["split"] == "unseen_test") & (generator_df["generator"] == "VQDM")]
        for _, row in sub.iterrows():
            lines.append(f"- {model} {row.condition}: recall={row.recall:.3f}")

    lines.extend(
        [
            "",
            "LIMITATIONS",
            "- Controlled transformations; not a full social-media pipeline.",
            "- Screenshot/re-digitisation excluded from Stage 22B.",
            "",
            "SCIENTIFIC INTEGRITY",
            "- Model training performed: NO",
            "- Model refitting performed: NO",
            "- Threshold changes: NO",
            "- Transformation-specific threshold tuning: NO",
            "- Generator-specific threshold tuning: NO",
            "- Test-time augmentation: NO",
            "- Samples excluded after seeing results: NO",
            "- RQ1 model development reopened: NO",
            "- Models evaluated on transformed locked test samples: YES",
        ]
    )
    return "\n".join(lines)


def print_terminal_summary(metrics: pd.DataFrame, extremes: dict) -> None:
    print("STAGE 22B — FROZEN-MODEL ROBUSTNESS EVALUATION COMPLETE")
    print("")
    print("UNSEEN ROC-AUC")
    header = f"{'Condition':<14}" + "".join(f"{m:>12}" for m in ["LogReg", "SmallCNN", "MobileNet", "EfficientNet"])
    print(header)
    name_map = {
        "Handcrafted LogReg": "LogReg",
        "SmallCNNV1": "SmallCNN",
        "MobileNetV3-Small": "MobileNet",
        "EfficientNet-B0": "EfficientNet",
    }
    for condition in ALL_CONDITIONS:
        parts = [f"{condition:<14}"]
        for model in MODEL_LABELS.values():
            val = metrics[(metrics["model"] == model) & (metrics["split"] == "unseen_test") & (metrics["condition"] == condition)].iloc[0].roc_auc
            parts.append(f"{val:12.3f}")
        print("".join(parts))
    print("")
    print("LARGEST UNSEEN AUC DROP")
    for model, splits in extremes.items():
        drop = splits["unseen_test"]["largest_auc_drop"]
        print(f"{name_map.get(model, model)}: {drop['condition']} ({drop['delta_auc']:+.4f})")
    print("")
    print("FROZEN THRESHOLDS CHANGED: NO")
    print("MODEL TRAINING: NO")
    print("TEST-TIME TUNING: NO")
    print("")
    print("RQ2 BASIC TRANSFORM EVALUATION: COMPLETE")
    print("SCREENSHOT EVALUATION: PENDING")
    print("STOP BEFORE SCREENSHOT SIMULATION OR TRANSFORMATION-AWARE TRAINING.")


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    validate_manifest(manifest)
    device = select_device()
    thresholds = {m: load_frozen_threshold(m) for m in MODEL_ORDER}

    logreg_pred = run_logreg(manifest, thresholds["LogReg"])
    smallcnn_pred = run_smallcnn(manifest, device)
    mobilenet_pred = run_imagenet_model(manifest, "MobileNetV3-Small", MobileNetV3SmallBinaryV1, MOB_WEIGHTS, device)
    efficientnet_pred = run_imagenet_model(manifest, "EfficientNet-B0", EfficientNetB0BinaryV1, EFF_WEIGHTS, device)

    predictions = {
        "LogReg": logreg_pred,
        "SmallCNNV1": smallcnn_pred,
        "MobileNetV3-Small": mobilenet_pred,
        "EfficientNet-B0": efficientnet_pred,
    }

    for model_key, pred in predictions.items():
        stop_if(len(pred) != EXPECTED_TRANSFORMED, f"{model_key} predictions {len(pred)} != {EXPECTED_TRANSFORMED}")
        path = PRED_PATHS[model_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(path, index=False)

    # alignment check
    ref = predictions["LogReg"][["source_image_id", "condition", "split"]].reset_index(drop=True)
    for model_key, pred in predictions.items():
        cur = pred[["source_image_id", "condition", "split"]].reset_index(drop=True)
        stop_if(not ref.equals(cur), f"{model_key} sample alignment failed")

    metrics = build_metrics_table(predictions, thresholds)
    generator_df = build_generator_recall(predictions, thresholds)
    extremes = summarize_extremes(metrics)

    metrics.to_csv(METRICS_CSV, index=False)
    generator_df.to_csv(GENERATOR_CSV, index=False)

    save_auc_figure(metrics, "unseen_test", FIG_UNSEEN_AUC, "roc_auc", "Unseen ROC-AUC by condition")
    save_auc_figure(metrics, "unseen_test", FIG_UNSEEN_DELTA, "delta_auc", "Unseen Delta AUC vs original")
    save_auc_figure(metrics, "known_test", FIG_KNOWN_DELTA, "delta_auc", "Known Delta AUC vs original")
    save_pretrained_recall_figure(generator_df)

    REPORT_PATH.write_text(write_report(metrics, generator_df, extremes), encoding="utf-8")
    print_terminal_summary(metrics, extremes)
    print(f"\nWrote {METRICS_CSV}")
    print(f"Wrote {GENERATOR_CSV}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
