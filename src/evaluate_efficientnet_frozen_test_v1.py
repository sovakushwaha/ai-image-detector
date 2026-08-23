"""Frozen EfficientNet-B0 known vs unseen test evaluation (Stage 20 Part B).

Why this file exists
--------------------
EfficientNet-B0 Transfer Baseline V1 is fully frozen after Part A threshold
selection. This script evaluates known_test and unseen_test once using the
selected Phase-2 checkpoint and validation-selected Youden threshold. No
training, tuning, or threshold changes.

How to run
----------
    source .venv/bin/activate
    python src/evaluate_efficientnet_frozen_test_v1.py

Prerequisite
------------
    python src/select_efficientnet_threshold_v1.py

What to expect
--------------
    results/efficientnet_b0_known_test_predictions_v1.csv
    results/efficientnet_b0_unseen_test_predictions_v1.csv
    results/efficientnet_b0_test_evaluation_v1.txt
    results/efficientnet_b0_test_metrics_v1.json
    figures/efficientnet_b0_known_unseen_roc_v1.png
    figures/efficientnet_b0_known_unseen_pr_v1.png
    figures/efficientnet_b0_generator_recall_v1.png
    figures/rq1_final_model_comparison_v1.png
    paper/tables/rq1_final_model_comparison.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib.patches import Patch
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
from torchvision import transforms

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, SPLIT_META_PATH, select_device, stop_if
from efficientnet_b0_binary_v1 import (
    DEFAULT_WEIGHTS,
    EfficientNetB0BinaryV1,
    count_parameters,
)

# --- frozen configuration ---
DEFAULT_THRESHOLD = 0.50
SELECTED_PHASE = 2
SELECTED_EPOCH = 16
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
TOTAL_PARAMS = 4008829

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]
UNSEEN_AI_GENERATORS = ["Midjourney", "VQDM", "Wukong"]

TEST_SPLITS = {"known_test", "unseen_test"}
EXPECTED_COUNTS = {
    "known_test": {"total": 456, "real": 228, "ai": 228},
    "unseen_test": {"total": 1712, "real": 856, "ai": 856},
}
KNOWN_AI_COUNTS = {"ADM": 57, "BigGAN": 57, "GLIDE": 57, "SD15": 57}
UNSEEN_AI_COUNTS = {"Midjourney": 286, "VQDM": 285, "Wukong": 285}

# Historical frozen references (do not rerun)
LOGREG_REF = {
    "model": "Handcrafted LogReg",
    "parameters": "N/A (classical)",
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
}

SMALLCNN_REF = {
    "model": "SmallCNNV1",
    "parameters": 23873,
    "known_roc_auc": 0.866,
    "known_ap": 0.867,
    "known_bal_acc_youden": 0.787,
    "known_f1_youden": 0.780,
    "known_recall_youden": 0.754,
    "unseen_roc_auc": 0.722,
    "unseen_ap": 0.656,
    "unseen_bal_acc_youden": 0.589,
    "unseen_f1_youden": 0.444,
    "unseen_recall_youden": 0.328,
    "auc_gap": -0.145,
    "ap_gap": -0.212,
}

MOBILENET_REF = {
    "model": "MobileNetV3-Small",
    "parameters": 1518881,
    "known_roc_auc": 0.951928,
    "known_ap": 0.957360,
    "known_bal_acc_youden": 0.883772,
    "known_f1_youden": 0.881432,
    "known_recall_youden": 0.864035,
    "unseen_roc_auc": 0.858989,
    "unseen_ap": 0.870927,
    "unseen_bal_acc_youden": 0.747079,
    "unseen_f1_youden": 0.701996,
    "unseen_recall_youden": 0.595794,
    "auc_gap": -0.092939,
    "ap_gap": -0.086433,
}

FROZEN_CONFIG_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_frozen_config_v1.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_selected_v1.pt"
KNOWN_PRED_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_known_test_predictions_v1.csv"
UNSEEN_PRED_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_unseen_test_predictions_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_test_evaluation_v1.txt"
METRICS_JSON_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_test_metrics_v1.json"
ROC_FIG_PATH = PROJECT_ROOT / "figures" / "efficientnet_b0_known_unseen_roc_v1.png"
PR_FIG_PATH = PROJECT_ROOT / "figures" / "efficientnet_b0_known_unseen_pr_v1.png"
GEN_RECALL_FIG_PATH = PROJECT_ROOT / "figures" / "efficientnet_b0_generator_recall_v1.png"
RQ1_COMP_FIG_PATH = PROJECT_ROOT / "figures" / "rq1_final_model_comparison_v1.png"
RQ1_TABLE_CSV = PROJECT_ROOT / "paper" / "tables" / "rq1_final_model_comparison.csv"
RQ1_CROSS_CSV = PROJECT_ROOT / "paper" / "tables" / "rq1_cross_generator_comparison.csv"


class TestControlledV1Dataset(Dataset):
    """Load locked test splits only (Stage 20 Part B)."""

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


def build_imagenet_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def assert_part_a_safety() -> dict:
    stop_if(not FROZEN_CONFIG_PATH.exists(), f"missing frozen config: {FROZEN_CONFIG_PATH}")
    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    stop_if(config["model"] != "EfficientNet-B0", "model mismatch")
    stop_if(int(config["selected_phase"]) != SELECTED_PHASE, "selected phase mismatch")
    stop_if(int(config["selected_epoch"]) != SELECTED_EPOCH, "selected epoch mismatch")
    stop_if(config["checkpoint"] != "models/efficientnet_b0_selected_v1.pt", "checkpoint path mismatch")
    stop_if(config["threshold_method"] != "validation_youden_j", "threshold method mismatch")
    stop_if(not config.get("model_weights_frozen", False), "model not frozen")
    stop_if(not config.get("threshold_selected_validation_only", False), "threshold not validation-only")
    stop_if(config.get("known_test_accessed", True), "known_test accessed during Part A")
    stop_if(config.get("unseen_test_accessed", True), "unseen_test accessed during Part A")
    stop_if(config["normalization_mean"] != IMAGENET_MEAN, "normalization mean mismatch")
    stop_if(config["normalization_std"] != IMAGENET_STD, "normalization std mismatch")
    print("Part A safety assertion:")
    print("Model frozen: YES")
    print("Threshold frozen: YES")
    print("Threshold selected using validation only: YES")
    print("known_test accessed so far: NO")
    print("unseen_test accessed so far: NO")
    print("")
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
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
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
    ax.set_title("EfficientNet-B0 known vs unseen test ROC")
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
    ax.set_title("EfficientNet-B0 known vs unseen test precision-recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PR_FIG_PATH, dpi=150)
    plt.close(fig)


def save_generator_recall_figure(generator_recall: pd.DataFrame, frozen_threshold: float) -> None:
    subset = generator_recall.copy()
    subset["group"] = subset["test_condition"].map({"known_test": "Known", "unseen_test": "Unseen"})
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(subset))
    colors = ["#4C72B0" if g == "Known" else "#DD8452" for g in subset["group"]]
    ax.bar(x, subset["recall_at_frozen_youden"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["generator"], rotation=30, ha="right")
    ax.set_ylabel("AI recall")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"EfficientNet-B0 AI recall by generator (frozen Youden threshold = {frozen_threshold:.4f})"
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(handles=[Patch(color="#4C72B0", label="Known"), Patch(color="#DD8452", label="Unseen")])
    fig.tight_layout()
    fig.savefig(GEN_RECALL_FIG_PATH, dpi=150)
    plt.close(fig)


def save_rq1_comparison_figure(final_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    models = final_table["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, final_table["Known ROC-AUC"], width, label="Known ROC-AUC", color="#4C72B0")
    ax.bar(x + width / 2, final_table["Unseen ROC-AUC"], width, label="Unseen ROC-AUC", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("RQ1 final model comparison: known vs unseen ROC-AUC (pilot protocol)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RQ1_COMP_FIG_PATH, dpi=150)
    plt.close(fig)


def build_final_comparison_table(
    known_eval: dict,
    unseen_eval: dict,
    gaps: dict,
) -> pd.DataFrame:
    k_fro = known_eval["frozen_threshold_metrics"]
    u_fro = unseen_eval["frozen_threshold_metrics"]
    eff_row = {
        "Model": "EfficientNet-B0",
        "Parameters": TOTAL_PARAMS,
        "Known ROC-AUC": known_eval["roc_auc"],
        "Unseen ROC-AUC": unseen_eval["roc_auc"],
        "AUC Gap": gaps["roc_auc"],
        "Known AP": known_eval["average_precision"],
        "Unseen AP": unseen_eval["average_precision"],
        "AP Gap": gaps["average_precision"],
        "Known Balanced Accuracy": k_fro["balanced_accuracy"],
        "Unseen Balanced Accuracy": u_fro["balanced_accuracy"],
        "Known AI Recall": k_fro["recall"],
        "Unseen AI Recall": u_fro["recall"],
        "Known F1": k_fro["f1"],
        "Unseen F1": u_fro["f1"],
    }
    rows = [
        {
            "Model": LOGREG_REF["model"],
            "Parameters": LOGREG_REF["parameters"],
            "Known ROC-AUC": LOGREG_REF["known_roc_auc"],
            "Unseen ROC-AUC": LOGREG_REF["unseen_roc_auc"],
            "AUC Gap": LOGREG_REF["auc_gap"],
            "Known AP": LOGREG_REF["known_ap"],
            "Unseen AP": LOGREG_REF["unseen_ap"],
            "AP Gap": LOGREG_REF["ap_gap"],
            "Known Balanced Accuracy": LOGREG_REF["known_bal_acc_youden"],
            "Unseen Balanced Accuracy": LOGREG_REF["unseen_bal_acc_youden"],
            "Known AI Recall": LOGREG_REF["known_recall_youden"],
            "Unseen AI Recall": LOGREG_REF["unseen_recall_youden"],
            "Known F1": LOGREG_REF["known_f1_youden"],
            "Unseen F1": LOGREG_REF["unseen_f1_youden"],
        },
        {
            "Model": SMALLCNN_REF["model"],
            "Parameters": SMALLCNN_REF["parameters"],
            "Known ROC-AUC": SMALLCNN_REF["known_roc_auc"],
            "Unseen ROC-AUC": SMALLCNN_REF["unseen_roc_auc"],
            "AUC Gap": SMALLCNN_REF["auc_gap"],
            "Known AP": SMALLCNN_REF["known_ap"],
            "Unseen AP": SMALLCNN_REF["unseen_ap"],
            "AP Gap": SMALLCNN_REF["ap_gap"],
            "Known Balanced Accuracy": SMALLCNN_REF["known_bal_acc_youden"],
            "Unseen Balanced Accuracy": SMALLCNN_REF["unseen_bal_acc_youden"],
            "Known AI Recall": SMALLCNN_REF["known_recall_youden"],
            "Unseen AI Recall": SMALLCNN_REF["unseen_recall_youden"],
            "Known F1": SMALLCNN_REF["known_f1_youden"],
            "Unseen F1": SMALLCNN_REF["unseen_f1_youden"],
        },
        {
            "Model": MOBILENET_REF["model"],
            "Parameters": MOBILENET_REF["parameters"],
            "Known ROC-AUC": MOBILENET_REF["known_roc_auc"],
            "Unseen ROC-AUC": MOBILENET_REF["unseen_roc_auc"],
            "AUC Gap": MOBILENET_REF["auc_gap"],
            "Known AP": MOBILENET_REF["known_ap"],
            "Unseen AP": MOBILENET_REF["unseen_ap"],
            "AP Gap": MOBILENET_REF["ap_gap"],
            "Known Balanced Accuracy": MOBILENET_REF["known_bal_acc_youden"],
            "Unseen Balanced Accuracy": MOBILENET_REF["unseen_bal_acc_youden"],
            "Known AI Recall": MOBILENET_REF["known_recall_youden"],
            "Unseen AI Recall": MOBILENET_REF["unseen_recall_youden"],
            "Known F1": MOBILENET_REF["known_f1_youden"],
            "Unseen F1": MOBILENET_REF["unseen_f1_youden"],
        },
        eff_row,
    ]
    return pd.DataFrame(rows)


def mobilenet_vs_efficientnet(known_eval: dict, unseen_eval: dict, gaps: dict) -> dict:
    return {
        "known_auc_diff": known_eval["roc_auc"] - MOBILENET_REF["known_roc_auc"],
        "unseen_auc_diff": unseen_eval["roc_auc"] - MOBILENET_REF["unseen_roc_auc"],
        "known_ap_diff": known_eval["average_precision"] - MOBILENET_REF["known_ap"],
        "unseen_ap_diff": unseen_eval["average_precision"] - MOBILENET_REF["unseen_ap"],
        "auc_gap_diff": gaps["roc_auc"] - MOBILENET_REF["auc_gap"],
        "ap_gap_diff": gaps["average_precision"] - MOBILENET_REF["ap_gap"],
    }


def rq1_observations(
    known_eval: dict,
    unseen_eval: dict,
    gaps: dict,
    mobilenet_diff: dict,
) -> list[str]:
    obs = []
    if known_eval["roc_auc"] > MOBILENET_REF["known_roc_auc"]:
        obs.append("EfficientNet exceeded MobileNet on known ROC-AUC.")
    elif known_eval["roc_auc"] < MOBILENET_REF["known_roc_auc"]:
        obs.append("EfficientNet fell below MobileNet on known ROC-AUC.")
    else:
        obs.append("EfficientNet matched MobileNet on known ROC-AUC.")

    if unseen_eval["roc_auc"] > MOBILENET_REF["unseen_roc_auc"]:
        obs.append("EfficientNet exceeded MobileNet on unseen ROC-AUC.")
    elif unseen_eval["roc_auc"] < MOBILENET_REF["unseen_roc_auc"]:
        obs.append("EfficientNet fell below MobileNet on unseen ROC-AUC.")
    else:
        obs.append("EfficientNet matched MobileNet on unseen ROC-AUC.")

    eff_gap_mag = abs(gaps["roc_auc"])
    mob_gap_mag = abs(MOBILENET_REF["auc_gap"])
    if eff_gap_mag < mob_gap_mag:
        obs.append("EfficientNet known→unseen AUC gap was smaller in magnitude than MobileNet.")
    elif eff_gap_mag > mob_gap_mag:
        obs.append("EfficientNet known→unseen AUC gap was larger in magnitude than MobileNet.")
    else:
        obs.append("EfficientNet and MobileNet had equal known→unseen AUC gap magnitude.")

    obs.append(
        f"Known AUC difference (EfficientNet - MobileNet): {mobilenet_diff['known_auc_diff']:+.6f}"
    )
    obs.append(
        f"Unseen AUC difference (EfficientNet - MobileNet): {mobilenet_diff['unseen_auc_diff']:+.6f}"
    )
    return obs


def write_report(
    frozen_config: dict,
    frozen_threshold: float,
    known_eval: dict,
    unseen_eval: dict,
    generator_recall: pd.DataFrame,
    gaps: dict,
    final_table: pd.DataFrame,
    mobilenet_diff: dict,
    observations: list[str],
) -> str:
    k_def = known_eval["default_threshold_metrics"]
    k_fro = known_eval["frozen_threshold_metrics"]
    u_def = unseen_eval["default_threshold_metrics"]
    u_fro = unseen_eval["frozen_threshold_metrics"]

    lines = [
        "EfficientNet-B0 Frozen Test Evaluation — Stage 20 Part B",
        "=======================================================",
        "",
        "FROZEN CONFIGURATION",
        "- model: EfficientNet-B0",
        f"- pretrained weights: {frozen_config['pretrained_weights']}",
        f"- selected phase: {SELECTED_PHASE}",
        f"- selected epoch: {SELECTED_EPOCH}",
        "- checkpoint: models/efficientnet_b0_selected_v1.pt",
        f"- parameter count: {TOTAL_PARAMS}",
        f"- seed: {RANDOM_SEED}",
        f"- normalization mean: {IMAGENET_MEAN}",
        f"- normalization std: {IMAGENET_STD}",
        f"- validation ROC-AUC: {frozen_config['validation_roc_auc']:.8f}",
        f"- validation AP: {frozen_config['validation_ap']:.8f}",
        f"- frozen threshold: {frozen_threshold:.12f}",
        f"- threshold method: validation_youden_j",
        "",
        "KNOWN TEST",
        f"- ROC-AUC: {known_eval['roc_auc']:.8f}",
        f"- Average Precision: {known_eval['average_precision']:.8f}",
        "Threshold 0.50 (diagnostic):",
        f"  accuracy={k_def['accuracy']:.8f}, balanced_acc={k_def['balanced_accuracy']:.8f}, "
        f"precision={k_def['precision']:.8f}, recall={k_def['recall']:.8f}, "
        f"specificity={k_def['specificity']:.8f}, f1={k_def['f1']:.8f}, fpr={k_def['false_positive_rate']:.8f}",
        f"  TN={k_def['TN']} FP={k_def['FP']} FN={k_def['FN']} TP={k_def['TP']}",
        "Frozen Youden threshold:",
        f"  accuracy={k_fro['accuracy']:.8f}, balanced_acc={k_fro['balanced_accuracy']:.8f}, "
        f"precision={k_fro['precision']:.8f}, recall={k_fro['recall']:.8f}, "
        f"specificity={k_fro['specificity']:.8f}, f1={k_fro['f1']:.8f}, fpr={k_fro['false_positive_rate']:.8f}",
        f"  TN={k_fro['TN']} FP={k_fro['FP']} FN={k_fro['FN']} TP={k_fro['TP']}",
        "",
        "UNSEEN TEST",
        f"- ROC-AUC: {unseen_eval['roc_auc']:.8f}",
        f"- Average Precision: {unseen_eval['average_precision']:.8f}",
        "Threshold 0.50 (diagnostic):",
        f"  accuracy={u_def['accuracy']:.8f}, balanced_acc={u_def['balanced_accuracy']:.8f}, "
        f"precision={u_def['precision']:.8f}, recall={u_def['recall']:.8f}, "
        f"specificity={u_def['specificity']:.8f}, f1={u_def['f1']:.8f}, fpr={u_def['false_positive_rate']:.8f}",
        f"  TN={u_def['TN']} FP={u_def['FP']} FN={u_def['FN']} TP={u_def['TP']}",
        "Frozen Youden threshold:",
        f"  accuracy={u_fro['accuracy']:.8f}, balanced_acc={u_fro['balanced_accuracy']:.8f}, "
        f"precision={u_fro['precision']:.8f}, recall={u_fro['recall']:.8f}, "
        f"specificity={u_fro['specificity']:.8f}, f1={u_fro['f1']:.8f}, fpr={u_fro['false_positive_rate']:.8f}",
        f"  TN={u_fro['TN']} FP={u_fro['FP']} FN={u_fro['FN']} TP={u_fro['TP']}",
        "",
        "GENERALISATION GAP (unseen - known)",
        f"- ROC-AUC gap: {gaps['roc_auc']:+.8f}",
        f"- AP gap: {gaps['average_precision']:+.8f}",
        f"- Balanced Accuracy gap (frozen threshold): {gaps['balanced_accuracy']:+.8f}",
        f"- F1 gap (frozen threshold): {gaps['f1']:+.8f}",
        f"- AI Recall gap (frozen threshold): {gaps['recall']:+.8f}",
        "",
        "GENERATOR-SPECIFIC AI RECALL",
        generator_recall.to_string(index=False),
        "",
        "FINAL FOUR-MODEL RQ1 COMPARISON",
        final_table.to_string(index=False),
        "",
        "MOBILENET VS EFFICIENTNET DIFFERENCES",
        f"- Known AUC difference: {mobilenet_diff['known_auc_diff']:+.6f}",
        f"- Unseen AUC difference: {mobilenet_diff['unseen_auc_diff']:+.6f}",
        f"- Known AP difference: {mobilenet_diff['known_ap_diff']:+.6f}",
        f"- Unseen AP difference: {mobilenet_diff['unseen_ap_diff']:+.6f}",
        f"- AUC gap difference (EfficientNet gap - MobileNet gap): {mobilenet_diff['auc_gap_diff']:+.6f}",
        f"- AP gap difference (EfficientNet gap - MobileNet gap): {mobilenet_diff['ap_gap_diff']:+.6f}",
        "",
        "RQ1 OBJECTIVE OBSERVATIONS",
        *observations,
        "",
        "SCIENTIFIC INTEGRITY",
        "- EfficientNet retrained after test access: NO",
        "- Threshold changed after test access: NO",
        "- Test-derived threshold selection: NO",
        "- Hyperparameters changed after test access: NO",
        "- known_test evaluated: YES",
        "- unseen_test evaluated: YES",
        "- Previous frozen models rerun: NO",
        "- Previous frozen models modified: NO",
        "- Additional architecture automatically started: NO",
    ]
    return "\n".join(lines)


def main() -> None:
    device = select_device()
    frozen_config = assert_part_a_safety()
    frozen_threshold = float(frozen_config["threshold"])
    transform = build_imagenet_transforms()

    stop_if(not CHECKPOINT_PATH.exists(), f"missing checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    stop_if(int(ckpt.get("phase", -1)) != SELECTED_PHASE, "checkpoint phase mismatch")
    stop_if(int(ckpt.get("epoch", -1)) != SELECTED_EPOCH, "checkpoint epoch mismatch")

    model = EfficientNetB0BinaryV1(weights=DEFAULT_WEIGHTS).to(device)
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

    final_table = build_final_comparison_table(known_eval, unseen_eval, gaps)
    mobilenet_diff = mobilenet_vs_efficientnet(known_eval, unseen_eval, gaps)
    observations = rq1_observations(known_eval, unseen_eval, gaps, mobilenet_diff)

    y_k = known_pred["true_label"].to_numpy()
    p_k = known_pred["ai_probability"].to_numpy()
    y_u = unseen_pred["true_label"].to_numpy()
    p_u = unseen_pred["ai_probability"].to_numpy()
    save_roc_figure(y_k, p_k, y_u, p_u)
    save_pr_figure(y_k, p_k, y_u, p_u)
    save_generator_recall_figure(generator_recall, frozen_threshold)
    save_rq1_comparison_figure(final_table)

    RQ1_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_table.to_csv(RQ1_TABLE_CSV, index=False)

    cross_table = final_table[
        [
            "Model",
            "Known ROC-AUC",
            "Unseen ROC-AUC",
            "AUC Gap",
            "Known AP",
            "Unseen AP",
            "AP Gap",
        ]
    ].rename(
        columns={
            "Model": "Model",
            "Known ROC-AUC": "Known ROC-AUC",
            "Unseen ROC-AUC": "Unseen ROC-AUC",
            "AUC Gap": "AUC Gap",
            "Known AP": "Known AP",
            "Unseen AP": "Unseen AP",
            "AP Gap": "AP Gap",
        }
    )
    cross_table.to_csv(RQ1_CROSS_CSV, index=False)

    k_fro = known_eval["frozen_threshold_metrics"]
    u_fro = unseen_eval["frozen_threshold_metrics"]

    metrics_json = {
        "model": "EfficientNet-B0",
        "checkpoint": "models/efficientnet_b0_selected_v1.pt",
        "selected_phase": SELECTED_PHASE,
        "selected_epoch": SELECTED_EPOCH,
        "frozen_threshold": frozen_threshold,
        "validation_roc_auc": frozen_config["validation_roc_auc"],
        "validation_ap": frozen_config["validation_ap"],
        "known_test": known_eval,
        "unseen_test": unseen_eval,
        "gaps_unseen_minus_known": gaps,
        "generator_recall": generator_recall.to_dict(orient="records"),
        "four_model_comparison": final_table.to_dict(orient="records"),
        "mobilenet_vs_efficientnet": mobilenet_diff,
        "rq1_observations": observations,
        "logreg_reference": LOGREG_REF,
        "smallcnn_reference": SMALLCNN_REF,
        "mobilenet_reference": MOBILENET_REF,
        "efficientnet_retrained_after_test": False,
        "threshold_changed_after_test": False,
        "test_derived_threshold_selection": False,
        "hyperparameters_changed_after_test": False,
        "known_test_evaluated": True,
        "unseen_test_evaluated": True,
        "previous_frozen_models_rerun": False,
        "previous_frozen_models_modified": False,
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    report = write_report(
        frozen_config,
        frozen_threshold,
        known_eval,
        unseen_eval,
        generator_recall,
        gaps,
        final_table,
        mobilenet_diff,
        observations,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("STAGE 20 — EFFICIENTNET FREEZE + TEST + RQ1 CLOSURE")
    print("")
    print("VALIDATION FREEZE")
    print(f"ROC-AUC: {frozen_config['validation_roc_auc']:.6f}")
    print(f"AP: {frozen_config['validation_ap']:.6f}")
    print(f"Youden threshold: {frozen_threshold:.12f}")
    print(f"Youden J: {frozen_config['validation_youden_j']:.6f}")
    print("")
    print("KNOWN TEST")
    print(f"ROC-AUC: {known_eval['roc_auc']:.6f}")
    print(f"AP: {known_eval['average_precision']:.6f}")
    print(f"Balanced Accuracy: {k_fro['balanced_accuracy']:.6f}")
    print(f"AI Recall: {k_fro['recall']:.6f}")
    print(f"F1: {k_fro['f1']:.6f}")
    print("")
    print("UNSEEN TEST")
    print(f"ROC-AUC: {unseen_eval['roc_auc']:.6f}")
    print(f"AP: {unseen_eval['average_precision']:.6f}")
    print(f"Balanced Accuracy: {u_fro['balanced_accuracy']:.6f}")
    print(f"AI Recall: {u_fro['recall']:.6f}")
    print(f"F1: {u_fro['f1']:.6f}")
    print("")
    print("GENERALISATION GAP")
    print(f"ROC-AUC: {gaps['roc_auc']:+.6f}")
    print(f"AP: {gaps['average_precision']:+.6f}")
    print(f"Balanced Accuracy: {gaps['balanced_accuracy']:+.6f}")
    print(f"AI Recall: {gaps['recall']:+.6f}")
    print(f"F1: {gaps['f1']:+.6f}")
    print("")
    print("FINAL RQ1 TABLE")
    print("")
    print("LogReg:")
    print(f"Known AUC {LOGREG_REF['known_roc_auc']:.3f}")
    print(f"Unseen AUC {LOGREG_REF['unseen_roc_auc']:.3f}")
    print("")
    print("SmallCNNV1:")
    print(f"Known AUC {SMALLCNN_REF['known_roc_auc']:.3f}")
    print(f"Unseen AUC {SMALLCNN_REF['unseen_roc_auc']:.3f}")
    print("")
    print("MobileNetV3-Small:")
    print(f"Known AUC {MOBILENET_REF['known_roc_auc']:.3f}")
    print(f"Unseen AUC {MOBILENET_REF['unseen_roc_auc']:.3f}")
    print("")
    print("EfficientNet-B0:")
    print(f"Known AUC {known_eval['roc_auc']:.3f}")
    print(f"Unseen AUC {unseen_eval['roc_auc']:.3f}")
    print("")
    print("RQ1 BASELINE MODEL DEVELOPMENT: CLOSED")
    print("")
    print("Model retrained after test access: NO")
    print("Threshold changed after test access: NO")
    print("Test-time tuning: NO")
    print("")
    print("STOP.")
    print(f"\nWrote {KNOWN_PRED_PATH}")
    print(f"Wrote {UNSEEN_PRED_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {METRICS_JSON_PATH}")
    print(f"Wrote {RQ1_TABLE_CSV}")
    print(f"Wrote {RQ1_CROSS_CSV}")


if __name__ == "__main__":
    main()
