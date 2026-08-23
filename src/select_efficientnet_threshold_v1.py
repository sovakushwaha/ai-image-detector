"""Validation-only threshold selection for frozen EfficientNet-B0 (Stage 20 Part A).

Why this file exists
--------------------
The selected Phase-2 EfficientNet checkpoint is frozen. This script runs
validation inference only, verifies checkpoint reproduction, selects a
Youden J threshold, and records both default (0.50) and selected operating
points. Test sets remain locked.

How to run
----------
    source .venv/bin/activate
    python src/select_efficientnet_threshold_v1.py

What to expect
--------------
    results/efficientnet_b0_validation_predictions_v1.csv
    results/efficientnet_b0_threshold_selection_v1.txt
    results/efficientnet_b0_frozen_config_v1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn_dataset_v1 import ControlledV1Dataset, load_split_metadata, select_device
from efficientnet_b0_binary_v1 import (
    DEFAULT_WEIGHTS,
    EfficientNetB0BinaryV1,
    count_parameters,
)

# --- named constants ---
DEFAULT_THRESHOLD = 0.50
YOUDEN_J_TIE_TOLERANCE = 1e-12
SELECTED_PHASE = 2
SELECTED_EPOCH = 16
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_VAL = 456

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REPRESENTATION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"
THRESHOLD_METHOD = "validation_youden_j"
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]

# Stage 19 reference values for reproduction check
STAGE19_REF_AUC = 0.954909
STAGE19_REF_AP = 0.960120
AUC_TOLERANCE = 0.005
AP_TOLERANCE = 0.005

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_selected_v1.pt"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_validation_predictions_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_threshold_selection_v1.txt"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "results" / "efficientnet_b0_frozen_config_v1.json"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def build_imagenet_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def confusion_parts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_parts(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(false_positive_rate),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def select_youden_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Select threshold maximising Youden J with tie-breaking.

    Tie-break order:
    1. maximum Youden J
    2. threshold closest to 0.50
    3. lower threshold
    """
    youden_j = tpr - fpr
    finite_mask = np.isfinite(thresholds)
    fpr = fpr[finite_mask]
    tpr = tpr[finite_mask]
    thresholds = thresholds[finite_mask]
    youden_j = youden_j[finite_mask]

    max_j = float(np.max(youden_j))
    tied = np.isclose(youden_j, max_j, atol=YOUDEN_J_TIE_TOLERANCE)
    candidates = pd.DataFrame(
        {
            "threshold": thresholds[tied],
            "fpr": fpr[tied],
            "tpr": tpr[tied],
            "youden_j": youden_j[tied],
            "distance_to_050": np.abs(thresholds[tied] - DEFAULT_THRESHOLD),
        }
    )
    candidates = candidates.sort_values(
        ["youden_j", "distance_to_050", "threshold"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    selected_threshold = float(candidates.loc[0, "threshold"])
    selected_j = float(candidates.loc[0, "youden_j"])
    return selected_threshold, selected_j


def generator_recall_rows(
    predictions: pd.DataFrame,
    threshold: float,
    label_suffix: str,
) -> list[dict]:
    rows = []
    pred_col = "predicted_label_youden" if label_suffix == "youden" else "predicted_label_default_050"
    for generator in KNOWN_AI_GENERATORS:
        group = predictions[(predictions["generator"] == generator) & (predictions["true_label"] == 1)]
        n_ai = int(len(group))
        detected = int((group[pred_col] == 1).sum())
        recall = float(detected / n_ai) if n_ai else float("nan")
        rows.append(
            {
                "generator": generator,
                "threshold": threshold,
                "ai_samples": n_ai,
                "detected": detected,
                "ai_recall": recall,
            }
        )
    return rows


@torch.no_grad()
def run_validation_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    val_rows: pd.DataFrame,
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

    stop_if(len(all_ids) != EXPECTED_VAL, f"expected {EXPECTED_VAL} predictions, got {len(all_ids)}")

    meta = val_rows.set_index("image_id")
    records = []
    for image_id, label, logit in zip(all_ids, all_labels, all_logits):
        row = meta.loc[image_id]
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
        records.append(
            {
                "image_id": image_id,
                "processed_path": row["processed_path"],
                "true_label": int(label),
                "generator": row["generator"],
                "split": row["split"],
                "raw_logit": float(logit),
                "ai_probability": prob,
            }
        )
    return pd.DataFrame(records)


def write_report(
    device: torch.device,
    total_params: int,
    roc_auc: float,
    average_precision: float,
    default_metrics: dict,
    youden_threshold: float,
    youden_j: float,
    youden_metrics: dict,
    generator_recall: pd.DataFrame,
    checkpoint_path: Path,
) -> str:
    lines = [
        "EfficientNet-B0 Threshold Selection V1 — Stage 20 Part A",
        "=======================================================",
        "",
        "MODEL",
        "- model: EfficientNet-B0",
        f"- pretrained weights: {PRETRAINED_WEIGHTS}",
        f"- selected phase: {SELECTED_PHASE}",
        f"- selected epoch: {SELECTED_EPOCH}",
        f"- checkpoint path: {checkpoint_path.relative_to(PROJECT_ROOT)}",
        f"- total parameters: {total_params}",
        f"- seed: {RANDOM_SEED}",
        f"- device: {device}",
        "",
        "VALIDATION",
        f"- sample count: {EXPECTED_VAL}",
        f"- ROC-AUC: {roc_auc:.8f}",
        f"- Average Precision: {average_precision:.8f}",
        "",
        "DEFAULT THRESHOLD 0.50",
        f"- accuracy: {default_metrics['accuracy']:.8f}",
        f"- balanced accuracy: {default_metrics['balanced_accuracy']:.8f}",
        f"- precision: {default_metrics['precision']:.8f}",
        f"- recall (AI sensitivity): {default_metrics['recall']:.8f}",
        f"- specificity: {default_metrics['specificity']:.8f}",
        f"- F1: {default_metrics['f1']:.8f}",
        f"- false positive rate: {default_metrics['false_positive_rate']:.8f}",
        f"- confusion matrix: TN={default_metrics['TN']} FP={default_metrics['FP']} "
        f"FN={default_metrics['FN']} TP={default_metrics['TP']}",
        "",
        "YOUDEN THRESHOLD",
        "validation-selected balanced operating threshold using Youden J",
        f"- exact threshold: {youden_threshold:.12f}",
        f"- Youden J: {youden_j:.8f}",
        f"- accuracy: {youden_metrics['accuracy']:.8f}",
        f"- balanced accuracy: {youden_metrics['balanced_accuracy']:.8f}",
        f"- precision: {youden_metrics['precision']:.8f}",
        f"- recall: {youden_metrics['recall']:.8f}",
        f"- specificity: {youden_metrics['specificity']:.8f}",
        f"- F1: {youden_metrics['f1']:.8f}",
        f"- false positive rate: {youden_metrics['false_positive_rate']:.8f}",
        f"- confusion matrix: TN={youden_metrics['TN']} FP={youden_metrics['FP']} "
        f"FN={youden_metrics['FN']} TP={youden_metrics['TP']}",
        "",
        "GENERATOR RECALL (diagnostic only; one global threshold)",
        generator_recall.to_string(index=False),
        "",
        "INTERPRETATION",
        "The Youden threshold is a validation-selected balanced operating point.",
        "It is not a deployment threshold.",
        "",
        "FREEZE STATUS",
        "- model weights frozen: YES",
        "- selected phase frozen: YES",
        "- selected epoch frozen: YES",
        "- threshold selected using validation only: YES",
        "- training performed: NO",
        "- optimizer updates performed: NO",
        "- known_test accessed: NO",
        "- unseen_test accessed: NO",
    ]
    return "\n".join(lines)


def main() -> None:
    device = select_device()
    stop_if(not CHECKPOINT_PATH.exists(), f"missing checkpoint: {CHECKPOINT_PATH}")

    # --- Data safety ---
    val_meta = load_split_metadata("validation", SPLIT_META_PATH)
    stop_if(len(val_meta) != EXPECTED_VAL, f"validation count {len(val_meta)} != {EXPECTED_VAL}")
    known_test_rows_accessed = 0
    unseen_test_rows_accessed = 0
    stop_if(known_test_rows_accessed != 0, f"known_test rows accessed = {known_test_rows_accessed}")
    stop_if(unseen_test_rows_accessed != 0, f"unseen_test rows accessed = {unseen_test_rows_accessed}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    stop_if(int(ckpt.get("phase", -1)) != SELECTED_PHASE, f"checkpoint phase != {SELECTED_PHASE}")
    stop_if(int(ckpt.get("epoch", -1)) != SELECTED_EPOCH, f"checkpoint epoch != {SELECTED_EPOCH}")
    stop_if(int(ckpt.get("seed", -1)) != RANDOM_SEED, f"checkpoint seed != {RANDOM_SEED}")

    final_sel = ckpt.get("final_selection", {})
    stop_if(int(final_sel.get("selected_phase", -1)) != SELECTED_PHASE, "final_selection phase mismatch")
    stop_if(int(final_sel.get("selected_epoch", -1)) != SELECTED_EPOCH, "final_selection epoch mismatch")

    transform = build_imagenet_transforms()
    val_ds = ControlledV1Dataset("validation", transform=transform)
    stop_if(len(val_ds) != EXPECTED_VAL, f"validation dataset count {len(val_ds)} != {EXPECTED_VAL}")
    stop_if(
        val_ds.rows["split"].isin(["known_test", "unseen_test"]).any(),
        "test rows leaked into validation dataset",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = EfficientNetB0BinaryV1(weights=DEFAULT_WEIGHTS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total_params, _ = count_parameters(model)

    predictions = run_validation_inference(model, val_loader, device, val_ds.rows)
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_prob = predictions["ai_probability"].to_numpy(dtype=float)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    average_precision = float(average_precision_score(y_true, y_prob))

    stop_if(
        abs(roc_auc - STAGE19_REF_AUC) > AUC_TOLERANCE,
        f"ROC-AUC {roc_auc:.6f} differs materially from Stage 19 reference {STAGE19_REF_AUC}",
    )
    stop_if(
        abs(average_precision - STAGE19_REF_AP) > AP_TOLERANCE,
        f"AP {average_precision:.6f} differs materially from Stage 19 reference {STAGE19_REF_AP}",
    )

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    youden_threshold, youden_j = select_youden_threshold(fpr, tpr, roc_thresholds)

    default_metrics = threshold_metrics(y_true, y_prob, DEFAULT_THRESHOLD)
    youden_metrics = threshold_metrics(y_true, y_prob, youden_threshold)

    predictions["predicted_label_default_050"] = (y_prob >= DEFAULT_THRESHOLD).astype(int)
    predictions["predicted_label_youden"] = (y_prob >= youden_threshold).astype(int)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    gen_rows = []
    gen_rows.extend(generator_recall_rows(predictions, DEFAULT_THRESHOLD, "default"))
    gen_rows.extend(generator_recall_rows(predictions, youden_threshold, "youden"))
    generator_recall = pd.DataFrame(gen_rows)

    frozen_config = {
        "model": "EfficientNet-B0",
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "selected_phase": SELECTED_PHASE,
        "selected_epoch": SELECTED_EPOCH,
        "seed": RANDOM_SEED,
        "representation": REPRESENTATION,
        "split_protocol": SPLIT_PROTOCOL,
        "threshold_method": THRESHOLD_METHOD,
        "default_threshold": DEFAULT_THRESHOLD,
        "threshold": youden_threshold,
        "validation_youden_j": youden_j,
        "validation_roc_auc": roc_auc,
        "validation_ap": average_precision,
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "total_parameters": total_params,
        "default_threshold_metrics": default_metrics,
        "youden_threshold_metrics": youden_metrics,
        "model_weights_frozen": True,
        "selected_phase_frozen": True,
        "selected_epoch_frozen": True,
        "threshold_selected_validation_only": True,
        "known_test_accessed": False,
        "unseen_test_accessed": False,
        "training_performed": False,
        "optimizer_updates_performed": False,
    }
    FROZEN_CONFIG_PATH.write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")

    report = write_report(
        device,
        total_params,
        roc_auc,
        average_precision,
        default_metrics,
        youden_threshold,
        youden_j,
        youden_metrics,
        generator_recall,
        CHECKPOINT_PATH,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("STAGE 20 PART A — EFFICIENTNET-B0 THRESHOLD SELECTION COMPLETE")
    print(f"Validation ROC-AUC: {roc_auc:.6f}")
    print(f"Validation AP: {average_precision:.6f}")
    print("")
    print(f"Default threshold: {DEFAULT_THRESHOLD:.6f}")
    print(f"Youden threshold: {youden_threshold:.12f}")
    print(f"Youden J: {youden_j:.6f}")
    print("")
    print("At Youden threshold:")
    print(f"Accuracy: {youden_metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy: {youden_metrics['balanced_accuracy']:.6f}")
    print(f"Precision: {youden_metrics['precision']:.6f}")
    print(f"Recall: {youden_metrics['recall']:.6f}")
    print(f"Specificity: {youden_metrics['specificity']:.6f}")
    print(f"F1: {youden_metrics['f1']:.6f}")
    print(f"FPR: {youden_metrics['false_positive_rate']:.6f}")
    print("")
    print(f"Selected phase: {SELECTED_PHASE}")
    print(f"Selected epoch: {SELECTED_EPOCH}")
    print("")
    print("Model frozen: YES")
    print("Threshold frozen: YES")
    print("")
    print("known_test accessed: NO")
    print("unseen_test accessed: NO")
    print("")
    print("STOP BEFORE TEST EVALUATION")
    print(f"\nWrote {PREDICTIONS_PATH}")
    print(f"Wrote {FROZEN_CONFIG_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
