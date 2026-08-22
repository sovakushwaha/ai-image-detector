"""Frozen SmallCNNV1 known vs unseen test evaluation (Stage 16).

Why this file exists
--------------------
SmallCNNV1 Baseline V1 is fully frozen. This script evaluates known_test
and unseen_test once using the epoch-60 checkpoint and validation-selected
Youden threshold. No training, tuning, or threshold changes.

How to run
----------
    source .venv/bin/activate
    python src/evaluate_smallcnn_frozen_test_v1.py

What to expect
--------------
    results/smallcnn_v1_known_test_predictions_v1.csv
    results/smallcnn_v1_unseen_test_predictions_v1.csv
    results/smallcnn_v1_test_evaluation_v1.txt
    results/smallcnn_v1_test_metrics_v1.json
    figures/smallcnn_v1_known_unseen_roc.png
    figures/smallcnn_v1_known_unseen_pr.png
    figures/smallcnn_v1_generator_recall.png
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from cnn_dataset_v1 import (
    PROJECT_ROOT,
    EXPECTED_SIZE,
    SPLIT_META_PATH,
    build_transforms,
    load_train_rgb_stats,
    select_device,
    stop_if,
)
from small_cnn_v1 import SmallCNNV1, count_parameters

# --- frozen configuration ---
DEFAULT_THRESHOLD = 0.50
FROZEN_YOUDEN_THRESHOLD = 0.5406003902614535
SELECTED_EPOCH = 60
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
TOTAL_PARAMS = 23873

KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]
UNSEEN_AI_GENERATORS = ["Midjourney", "VQDM", "Wukong"]
ALL_AI_GENERATORS = KNOWN_AI_GENERATORS + UNSEEN_AI_GENERATORS

TEST_SPLITS = {"known_test", "unseen_test"}
EXPECTED_COUNTS = {
    "known_test": {"total": 456, "real": 228, "ai": 228},
    "unseen_test": {"total": 1712, "real": 856, "ai": 856},
}
KNOWN_AI_COUNTS = {"ADM": 57, "BigGAN": 57, "GLIDE": 57, "SD15": 57}
UNSEEN_AI_COUNTS = {"Midjourney": 286, "VQDM": 285, "Wukong": 285}

# Historical Classical Baseline V1 reference (do not rerun)
LOGREG_REF = {
    "known_roc_auc": 0.725,
    "known_ap": 0.723,
    "known_bal_acc_youden": 0.686,
    "known_f1_youden": 0.683,
    "known_recall_youden": 0.675,
    "unseen_roc_auc": 0.645,
    "unseen_ap": 0.614,
    "unseen_bal_acc_youden": 0.580,
    "unseen_f1_youden": 0.503,
    "unseen_recall_youden": 0.425,
    "auc_gap": -0.079,
    "ap_gap": -0.108,
    "bal_acc_gap": -0.106,
    "f1_gap": -0.180,
    "recall_gap": -0.250,
}

FROZEN_CONFIG_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_frozen_config_v1.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "smallcnn_v1_60ep_best.pt"
KNOWN_PRED_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_known_test_predictions_v1.csv"
UNSEEN_PRED_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_unseen_test_predictions_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_test_evaluation_v1.txt"
METRICS_JSON_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_test_metrics_v1.json"
ROC_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_known_unseen_roc.png"
PR_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_known_unseen_pr.png"
GEN_RECALL_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_generator_recall.png"


class TestControlledV1Dataset(Dataset):
    """Load locked test splits only (Stage 16)."""

    def __init__(self, split: str, transform, meta_path: Path = SPLIT_META_PATH):
        stop_if(split not in TEST_SPLITS, f"refusing non-test split '{split}'")
        table = pd.read_csv(meta_path)
        self.rows = table[table["split"] == split].copy().reset_index(drop=True)
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["processed_path"]
        with Image.open(path) as image:
            image.load()
            stop_if(image.format != "JPEG", f"{path} is {image.format}, expected JPEG")
            stop_if(image.mode != "RGB", f"{path} is mode {image.mode}, expected RGB")
            stop_if(image.size != EXPECTED_SIZE, f"{path} is {image.size}, expected {EXPECTED_SIZE}")
            rgb = image.convert("RGB")
        image_tensor = self.transform(rgb)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        image_id = str(row["image_id"])
        return image_tensor, label, image_id


def load_frozen_config() -> dict:
    stop_if(not FROZEN_CONFIG_PATH.exists(), f"missing frozen config: {FROZEN_CONFIG_PATH}")
    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    stop_if(config["model"] != "SmallCNNV1", "model mismatch")
    stop_if(int(config["selected_epoch"]) != SELECTED_EPOCH, "epoch mismatch")
    stop_if(config["checkpoint"] != "models/smallcnn_v1_60ep_best.pt", "checkpoint path mismatch")
    stop_if(config["threshold_method"] != "validation_youden_j", "threshold method mismatch")
    stop_if(abs(float(config["threshold"]) - FROZEN_YOUDEN_THRESHOLD) > 1e-6, "threshold mismatch")
    return config


def validate_test_counts(rows: pd.DataFrame, split: str) -> None:
    expected = EXPECTED_COUNTS[split]
    stop_if(len(rows) != expected["total"], f"{split} total {len(rows)} != {expected['total']}")
    stop_if(int((rows["label"] == 0).sum()) != expected["real"], f"{split} real count wrong")
    stop_if(int((rows["label"] == 1).sum()) != expected["ai"], f"{split} ai count wrong")
    if split == "known_test":
        for gen, count in KNOWN_AI_COUNTS.items():
            n = int(((rows["generator"] == gen) & (rows["label"] == 1)).sum())
            stop_if(n != count, f"known_test {gen} AI count {n} != {count}")
    if split == "unseen_test":
        for gen, count in UNSEEN_AI_COUNTS.items():
            n = int(((rows["generator"] == gen) & (rows["label"] == 1)).sum())
            stop_if(n != count, f"unseen_test {gen} AI count {n} != {count}")


def confusion_parts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_parts(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(fpr),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    meta_rows: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []
    all_ids: list[str] = []

    for images, labels, image_ids in loader:
        images = images.to(device)
        logits = model(images).cpu().numpy()
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.numpy().astype(int).tolist())
        all_ids.extend(list(image_ids))

    meta = meta_rows.set_index("image_id")
    records = []
    for image_id, label, logit in zip(all_ids, all_labels, all_logits):
        row = meta.loc[image_id]
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        records.append(
            {
                "split": split,
                "image_id": image_id,
                "processed_path": row["processed_path"],
                "true_label": int(label),
                "generator": row["generator"],
                "raw_logit": float(logit),
                "ai_probability": prob,
            }
        )
    return pd.DataFrame(records)


def evaluate_predictions(
    predictions: pd.DataFrame,
    default_threshold: float,
    frozen_threshold: float,
) -> dict:
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_prob = predictions["ai_probability"].to_numpy(dtype=float)
    return {
        "n_samples": int(len(predictions)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "default_threshold_metrics": threshold_metrics(y_true, y_prob, default_threshold),
        "frozen_threshold_metrics": threshold_metrics(y_true, y_prob, frozen_threshold),
    }


def generator_recall_table(
    predictions: pd.DataFrame,
    generators: list[str],
    test_condition: str,
    default_threshold: float,
    frozen_threshold: float,
) -> pd.DataFrame:
    rows = []
    for generator in generators:
        group = predictions[(predictions["generator"] == generator) & (predictions["true_label"] == 1)]
        n_ai = int(len(group))
        det_050 = int((group["ai_probability"] >= default_threshold).sum())
        det_frozen = int((group["ai_probability"] >= frozen_threshold).sum())
        rows.append(
            {
                "test_condition": test_condition,
                "generator": generator,
                "ai_samples": n_ai,
                "detected_at_050": det_050,
                "recall_at_050": float(det_050 / n_ai) if n_ai else float("nan"),
                "detected_at_frozen_youden": det_frozen,
                "recall_at_frozen_youden": float(det_frozen / n_ai) if n_ai else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def save_roc_figure(y_k, p_k, y_u, p_u) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for y_true, y_prob, label in [(y_k, p_k, "known_test"), (y_u, p_u, "unseen_test")]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("SmallCNNV1 known vs unseen test ROC")
    ax.legend()
    fig.tight_layout()
    ROC_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROC_FIG_PATH, dpi=150)
    plt.close(fig)


def save_pr_figure(y_k, p_k, y_u, p_u) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for y_true, y_prob, label in [(y_k, p_k, "known_test"), (y_u, p_u, "unseen_test")]:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("SmallCNNV1 known vs unseen test precision-recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PR_FIG_PATH, dpi=150)
    plt.close(fig)


def save_generator_recall_figure(generator_recall: pd.DataFrame, frozen_threshold: float) -> None:
    subset = generator_recall.copy()
    subset["group"] = subset["test_condition"].map(
        {"known_test": "Known", "unseen_test": "Unseen"}
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subset))
    colors = ["#4C72B0" if g == "Known" else "#DD8452" for g in subset["group"]]
    ax.bar(x, subset["recall_at_frozen_youden"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["generator"], rotation=30, ha="right")
    ax.set_ylabel("AI recall")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"SmallCNNV1 AI recall by generator (frozen Youden threshold = {frozen_threshold:.4f})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="Known"), Patch(color="#DD8452", label="Unseen")])
    fig.tight_layout()
    fig.savefig(GEN_RECALL_FIG_PATH, dpi=150)
    plt.close(fig)


def write_report(
    frozen_config: dict,
    known_eval: dict,
    unseen_eval: dict,
    generator_recall: pd.DataFrame,
    gaps: dict,
    comparison: pd.DataFrame,
) -> str:
    k_def = known_eval["default_threshold_metrics"]
    k_fro = known_eval["frozen_threshold_metrics"]
    u_def = unseen_eval["default_threshold_metrics"]
    u_fro = unseen_eval["frozen_threshold_metrics"]

    lines = [
        "SmallCNNV1 Frozen Test Evaluation — Stage 16",
        "============================================",
        "",
        "FROZEN MODEL CONFIGURATION",
        "- model: SmallCNNV1",
        f"- selected epoch: {SELECTED_EPOCH}",
        f"- checkpoint: models/smallcnn_v1_60ep_best.pt",
        f"- parameter count: {TOTAL_PARAMS}",
        f"- seed: {RANDOM_SEED}",
        f"- normalization mean: {frozen_config['normalization_mean']}",
        f"- normalization std: {frozen_config['normalization_std']}",
        f"- frozen threshold: {FROZEN_YOUDEN_THRESHOLD}",
        f"- threshold method: validation_youden_j",
        "",
        "KNOWN TEST",
        f"- ROC-AUC: {known_eval['roc_auc']:.6f}",
        f"- Average Precision: {known_eval['average_precision']:.6f}",
        "Threshold 0.50 (diagnostic):",
        f"  accuracy={k_def['accuracy']:.6f}, balanced_acc={k_def['balanced_accuracy']:.6f}, "
        f"precision={k_def['precision']:.6f}, recall={k_def['recall']:.6f}, "
        f"specificity={k_def['specificity']:.6f}, f1={k_def['f1']:.6f}, fpr={k_def['false_positive_rate']:.6f}",
        f"  TN={k_def['TN']} FP={k_def['FP']} FN={k_def['FN']} TP={k_def['TP']}",
        "Frozen Youden threshold:",
        f"  accuracy={k_fro['accuracy']:.6f}, balanced_acc={k_fro['balanced_accuracy']:.6f}, "
        f"precision={k_fro['precision']:.6f}, recall={k_fro['recall']:.6f}, "
        f"specificity={k_fro['specificity']:.6f}, f1={k_fro['f1']:.6f}, fpr={k_fro['false_positive_rate']:.6f}",
        f"  TN={k_fro['TN']} FP={k_fro['FP']} FN={k_fro['FN']} TP={k_fro['TP']}",
        "",
        "UNSEEN TEST",
        f"- ROC-AUC: {unseen_eval['roc_auc']:.6f}",
        f"- Average Precision: {unseen_eval['average_precision']:.6f}",
        "Threshold 0.50 (diagnostic):",
        f"  accuracy={u_def['accuracy']:.6f}, balanced_acc={u_def['balanced_accuracy']:.6f}, "
        f"precision={u_def['precision']:.6f}, recall={u_def['recall']:.6f}, "
        f"specificity={u_def['specificity']:.6f}, f1={u_def['f1']:.6f}, fpr={u_def['false_positive_rate']:.6f}",
        f"  TN={u_def['TN']} FP={u_def['FP']} FN={u_def['FN']} TP={u_def['TP']}",
        "Frozen Youden threshold:",
        f"  accuracy={u_fro['accuracy']:.6f}, balanced_acc={u_fro['balanced_accuracy']:.6f}, "
        f"precision={u_fro['precision']:.6f}, recall={u_fro['recall']:.6f}, "
        f"specificity={u_fro['specificity']:.6f}, f1={u_fro['f1']:.6f}, fpr={u_fro['false_positive_rate']:.6f}",
        f"  TN={u_fro['TN']} FP={u_fro['FP']} FN={u_fro['FN']} TP={u_fro['TP']}",
        "",
        "GENERALISATION GAPS (unseen - known)",
        f"- ROC-AUC gap: {gaps['roc_auc']:+.6f}",
        f"- AP gap: {gaps['average_precision']:+.6f}",
        f"- Balanced Accuracy gap (frozen threshold): {gaps['balanced_accuracy']:+.6f}",
        f"- F1 gap (frozen threshold): {gaps['f1']:+.6f}",
        f"- AI Recall gap (frozen threshold): {gaps['recall']:+.6f}",
        "",
        "GENERATOR-SPECIFIC AI RECALL",
        generator_recall.to_string(index=False),
        "",
        "CLASSICAL BASELINE COMPARISON (historical reference only)",
        comparison.to_string(index=False),
        "",
        "SAFETY / SCIENTIFIC INTEGRITY",
        "- Model refitted after test access: NO",
        "- Model weights changed: NO",
        "- Threshold changed: NO",
        "- Threshold selected using test data: NO",
        "- Hyperparameters changed: NO",
        "- known_test used for evaluation: YES",
        "- unseen_test used for evaluation: YES",
        "- Classical Baseline V1 modified: NO",
    ]
    return "\n".join(lines)


def main() -> None:
    device = select_device()
    frozen_config = load_frozen_config()
    frozen_threshold = float(frozen_config["threshold"])

    norm_stats = load_train_rgb_stats(PROJECT_ROOT / "results" / "cnn_train_normalization_v1.json")
    transform = build_transforms(norm_stats)

    stop_if(not CHECKPOINT_PATH.exists(), f"missing checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    stop_if(int(ckpt.get("epoch", -1)) != SELECTED_EPOCH, "checkpoint epoch mismatch")

    model = SmallCNNV1().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    total_params, _ = count_parameters(model)
    stop_if(total_params != TOTAL_PARAMS, f"parameter count {total_params} != {TOTAL_PARAMS}")
    model.eval()

    known_ds = TestControlledV1Dataset("known_test", transform=transform)
    unseen_ds = TestControlledV1Dataset("unseen_test", transform=transform)
    validate_test_counts(known_ds.rows, "known_test")
    validate_test_counts(unseen_ds.rows, "unseen_test")

    known_loader = DataLoader(known_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    unseen_loader = DataLoader(unseen_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    known_pred = run_inference(model, known_loader, device, known_ds.rows, "known_test")
    unseen_pred = run_inference(model, unseen_loader, device, unseen_ds.rows, "unseen_test")

    known_pred["predicted_label_default_050"] = (known_pred["ai_probability"] >= DEFAULT_THRESHOLD).astype(int)
    known_pred["predicted_label_frozen_youden"] = (known_pred["ai_probability"] >= frozen_threshold).astype(int)
    unseen_pred["predicted_label_default_050"] = (unseen_pred["ai_probability"] >= DEFAULT_THRESHOLD).astype(int)
    unseen_pred["predicted_label_frozen_youden"] = (unseen_pred["ai_probability"] >= frozen_threshold).astype(int)

    KNOWN_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    known_pred.to_csv(KNOWN_PRED_PATH, index=False)
    unseen_pred.to_csv(UNSEEN_PRED_PATH, index=False)

    known_eval = evaluate_predictions(known_pred, DEFAULT_THRESHOLD, frozen_threshold)
    unseen_eval = evaluate_predictions(unseen_pred, DEFAULT_THRESHOLD, frozen_threshold)

    gaps = {
        "roc_auc": unseen_eval["roc_auc"] - known_eval["roc_auc"],
        "average_precision": unseen_eval["average_precision"] - known_eval["average_precision"],
        "balanced_accuracy": (
            unseen_eval["frozen_threshold_metrics"]["balanced_accuracy"]
            - known_eval["frozen_threshold_metrics"]["balanced_accuracy"]
        ),
        "f1": unseen_eval["frozen_threshold_metrics"]["f1"] - known_eval["frozen_threshold_metrics"]["f1"],
        "recall": unseen_eval["frozen_threshold_metrics"]["recall"] - known_eval["frozen_threshold_metrics"]["recall"],
    }

    generator_recall = pd.concat(
        [
            generator_recall_table(
                known_pred, KNOWN_AI_GENERATORS, "known_test", DEFAULT_THRESHOLD, frozen_threshold
            ),
            generator_recall_table(
                unseen_pred, UNSEEN_AI_GENERATORS, "unseen_test", DEFAULT_THRESHOLD, frozen_threshold
            ),
        ],
        ignore_index=True,
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "LogReg V1",
                "known_auc": LOGREG_REF["known_roc_auc"],
                "unseen_auc": LOGREG_REF["unseen_roc_auc"],
                "auc_gap": LOGREG_REF["auc_gap"],
                "known_ap": LOGREG_REF["known_ap"],
                "unseen_ap": LOGREG_REF["unseen_ap"],
                "ap_gap": LOGREG_REF["ap_gap"],
            },
            {
                "model": "SmallCNNV1",
                "known_auc": known_eval["roc_auc"],
                "unseen_auc": unseen_eval["roc_auc"],
                "auc_gap": gaps["roc_auc"],
                "known_ap": known_eval["average_precision"],
                "unseen_ap": unseen_eval["average_precision"],
                "ap_gap": gaps["average_precision"],
            },
        ]
    )

    y_k = known_pred["true_label"].to_numpy()
    p_k = known_pred["ai_probability"].to_numpy()
    y_u = unseen_pred["true_label"].to_numpy()
    p_u = unseen_pred["ai_probability"].to_numpy()
    save_roc_figure(y_k, p_k, y_u, p_u)
    save_pr_figure(y_k, p_k, y_u, p_u)
    save_generator_recall_figure(generator_recall, frozen_threshold)

    metrics_json = {
        "model": "SmallCNNV1",
        "checkpoint": "models/smallcnn_v1_60ep_best.pt",
        "selected_epoch": SELECTED_EPOCH,
        "frozen_threshold": frozen_threshold,
        "known_test": known_eval,
        "unseen_test": unseen_eval,
        "gaps_unseen_minus_known": gaps,
        "generator_recall": generator_recall.to_dict(orient="records"),
        "classical_baseline_comparison": comparison.to_dict(orient="records"),
        "model_refitted_after_test": False,
        "threshold_changed": False,
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    report = write_report(frozen_config, known_eval, unseen_eval, generator_recall, gaps, comparison)
    REPORT_PATH.write_text(report, encoding="utf-8")

    k_fro = known_eval["frozen_threshold_metrics"]
    u_fro = unseen_eval["frozen_threshold_metrics"]

    print("STAGE 16 — SMALLCNNV1 FROZEN TEST EVALUATION")
    print("")
    print("KNOWN TEST")
    print(f"ROC-AUC: {known_eval['roc_auc']:.6f}")
    print(f"AP: {known_eval['average_precision']:.6f}")
    print(f"Frozen-threshold Balanced Accuracy: {k_fro['balanced_accuracy']:.6f}")
    print(f"Frozen-threshold AI Recall: {k_fro['recall']:.6f}")
    print(f"Frozen-threshold F1: {k_fro['f1']:.6f}")
    print("")
    print("UNSEEN TEST")
    print(f"ROC-AUC: {unseen_eval['roc_auc']:.6f}")
    print(f"AP: {unseen_eval['average_precision']:.6f}")
    print(f"Frozen-threshold Balanced Accuracy: {u_fro['balanced_accuracy']:.6f}")
    print(f"Frozen-threshold AI Recall: {u_fro['recall']:.6f}")
    print(f"Frozen-threshold F1: {u_fro['f1']:.6f}")
    print("")
    print("GENERALISATION GAP")
    print(f"ROC-AUC: {gaps['roc_auc']:+.6f}")
    print(f"AP: {gaps['average_precision']:+.6f}")
    print(f"Balanced Accuracy: {gaps['balanced_accuracy']:+.6f}")
    print(f"AI Recall: {gaps['recall']:+.6f}")
    print(f"F1: {gaps['f1']:+.6f}")
    print("")
    print("CLASSICAL BASELINE")
    print(f"Known ROC-AUC: {LOGREG_REF['known_roc_auc']:.3f}")
    print(f"Unseen ROC-AUC: {LOGREG_REF['unseen_roc_auc']:.3f}")
    print(f"Gap: {LOGREG_REF['auc_gap']:+.3f}")
    print("")
    print("SMALLCNNV1")
    print(f"Known ROC-AUC: {known_eval['roc_auc']:.3f}")
    print(f"Unseen ROC-AUC: {unseen_eval['roc_auc']:.3f}")
    print(f"Gap: {gaps['roc_auc']:+.3f}")
    print("")
    print("Model refitted: NO")
    print("Threshold changed: NO")
    print("Test-time tuning: NO")
    print("")
    print("STAGE 16 COMPLETE")


if __name__ == "__main__":
    main()
