"""Clean-validation Youden threshold selection for RQ3 A1–A3 (Stage 23C).

Why this file exists
--------------------
RQ3 checkpoints were selected via RobustValAUC. Operating thresholds are
selected separately using Youden J on the original clean validation split
only — matching A0 / LogReg / SmallCNN / EfficientNet methodology.

No training. No test access. No transformation-specific thresholds.

How to run
----------
    source .venv/bin/activate
    python src/select_rq3_thresholds_v1.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

from cnn_dataset_v1 import ControlledV1Dataset, PROJECT_ROOT, load_split_metadata, select_device, stop_if
from mobilenet_v3_small_binary_v1 import (
    DEFAULT_WEIGHTS,
    MobileNetV3SmallBinaryV1,
    count_parameters,
)

DEFAULT_THRESHOLD = 0.50
YOUDEN_J_TIE_TOLERANCE = 1e-12
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_VAL = 456
AUC_TOLERANCE = 0.005
AP_TOLERANCE = 0.005

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THRESHOLD_METHOD = "clean_validation_youden_j"
KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]

SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "rq3_development_summary_v1.csv"
A0_FROZEN_PATH = PROJECT_ROOT / "results" / "mobilenet_v3_small_frozen_config_v1.json"
REPORT_PATH = PROJECT_ROOT / "results" / "rq3_threshold_selection_report_v1.txt"
FROZEN_SUMMARY_CSV = PROJECT_ROOT / "results" / "rq3_frozen_models_v1.csv"


@dataclass(frozen=True)
class RegimeSpec:
    key: str
    regime_id: str
    augmentation: str
    checkpoint: Path
    predictions_path: Path
    frozen_config_path: Path
    selected_phase: int
    selected_epoch: int
    primary: bool


REGIMES = [
    RegimeSpec(
        key="A1",
        regime_id="mobilenet_blur_aug_v1",
        augmentation="blur-aware",
        checkpoint=PROJECT_ROOT / "models" / "mobilenet_blur_aug_selected_v1.pt",
        predictions_path=PROJECT_ROOT / "results" / "rq3_A1_clean_validation_predictions_v1.csv",
        frozen_config_path=PROJECT_ROOT / "results" / "rq3_A1_frozen_config_v1.json",
        selected_phase=2,
        selected_epoch=9,
        primary=False,
    ),
    RegimeSpec(
        key="A2",
        regime_id="mobilenet_resize_jpeg_aug_v1",
        augmentation="resize+JPEG-aware",
        checkpoint=PROJECT_ROOT / "models" / "mobilenet_resize_jpeg_aug_selected_v1.pt",
        predictions_path=PROJECT_ROOT / "results" / "rq3_A2_clean_validation_predictions_v1.csv",
        frozen_config_path=PROJECT_ROOT / "results" / "rq3_A2_frozen_config_v1.json",
        selected_phase=2,
        selected_epoch=16,
        primary=True,
    ),
    RegimeSpec(
        key="A3",
        regime_id="mobilenet_combined_aug_v1",
        augmentation="combined blur+resize+JPEG-aware",
        checkpoint=PROJECT_ROOT / "models" / "mobilenet_combined_aug_selected_v1.pt",
        predictions_path=PROJECT_ROOT / "results" / "rq3_A3_clean_validation_predictions_v1.csv",
        frozen_config_path=PROJECT_ROOT / "results" / "rq3_A3_frozen_config_v1.json",
        selected_phase=2,
        selected_epoch=8,
        primary=False,
    ),
]


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
    return float(candidates.loc[0, "threshold"]), float(candidates.loc[0, "youden_j"])


def generator_recall_rows(predictions: pd.DataFrame, threshold: float, label_suffix: str) -> list[dict]:
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
                "threshold_label": label_suffix,
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
                "label": int(label),
                "generator": row["generator"],
                "split": row["split"],
                "logit": float(logit),
                "raw_logit": float(logit),
                "probability": prob,
                "ai_probability": prob,
            }
        )
    return pd.DataFrame(records)


def process_regime(
    regime: RegimeSpec,
    val_loader: DataLoader,
    val_rows: pd.DataFrame,
    device: torch.device,
    summary: pd.DataFrame,
) -> dict:
    stop_if(not regime.checkpoint.exists(), f"missing checkpoint: {regime.checkpoint}")
    ckpt = torch.load(regime.checkpoint, map_location=device, weights_only=False)

    stop_if(int(ckpt.get("phase", -1)) != regime.selected_phase, f"{regime.key}: phase mismatch")
    stop_if(int(ckpt.get("epoch", -1)) != regime.selected_epoch, f"{regime.key}: epoch mismatch")
    final_sel = ckpt.get("final_selection", {})
    stop_if(int(final_sel.get("selected_phase", -1)) != regime.selected_phase, f"{regime.key}: final phase mismatch")
    stop_if(int(final_sel.get("selected_epoch", -1)) != regime.selected_epoch, f"{regime.key}: final epoch mismatch")

    ref_auc = float(ckpt["condition_aucs"]["original"])
    ref_ap = float(ckpt["condition_aps"]["original"])
    robust_val_auc = float(ckpt["robust_val_auc"])
    summary_row = summary[summary["regime"] == regime.key].iloc[0]
    stop_if(
        abs(float(summary_row["original_auc"]) - ref_auc) > 1e-9,
        f"{regime.key}: summary original AUC mismatch vs checkpoint",
    )
    stop_if(
        abs(float(summary_row["robust_val_auc"]) - robust_val_auc) > 1e-9,
        f"{regime.key}: summary RobustValAUC mismatch vs checkpoint",
    )

    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    total_params, _ = count_parameters(model)

    predictions = run_validation_inference(model, val_loader, device, val_rows)
    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_prob = predictions["probability"].to_numpy(dtype=float)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    average_precision = float(average_precision_score(y_true, y_prob))

    stop_if(
        abs(roc_auc - ref_auc) > AUC_TOLERANCE,
        f"{regime.key}: clean AUC {roc_auc:.8f} differs materially from Stage-23B {ref_auc:.8f}",
    )
    stop_if(
        abs(average_precision - ref_ap) > AP_TOLERANCE,
        f"{regime.key}: clean AP {average_precision:.8f} differs materially from Stage-23B {ref_ap:.8f}",
    )

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    youden_threshold, youden_j = select_youden_threshold(fpr, tpr, roc_thresholds)
    default_metrics = threshold_metrics(y_true, y_prob, DEFAULT_THRESHOLD)
    youden_metrics = threshold_metrics(y_true, y_prob, youden_threshold)

    predictions["predicted_label_default_050"] = (y_prob >= DEFAULT_THRESHOLD).astype(int)
    predictions["predicted_label_youden"] = (y_prob >= youden_threshold).astype(int)
    regime.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(regime.predictions_path, index=False)

    gen_rows = []
    gen_rows.extend(generator_recall_rows(predictions, DEFAULT_THRESHOLD, "default"))
    gen_rows.extend(generator_recall_rows(predictions, youden_threshold, "youden"))
    generator_recall = pd.DataFrame(gen_rows)

    frozen_config = {
        "model": "MobileNetV3-Small",
        "regime": regime.regime_id,
        "checkpoint": str(regime.checkpoint.relative_to(PROJECT_ROOT)),
        "selected_phase": regime.selected_phase,
        "selected_epoch": regime.selected_epoch,
        "seed": RANDOM_SEED,
        "augmentation": regime.augmentation,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "threshold_method": THRESHOLD_METHOD,
        "default_threshold": DEFAULT_THRESHOLD,
        "threshold": youden_threshold,
        "validation_youden_j": youden_j,
        "clean_validation_auc": roc_auc,
        "clean_validation_ap": average_precision,
        "robust_val_auc": robust_val_auc,
        "stage23b_reference_clean_auc": ref_auc,
        "stage23b_reference_clean_ap": ref_ap,
        "checkpoint_reproduction_passed": True,
        "primary_rq3_candidate": regime.primary,
        "representation": "controlled_v1",
        "split_protocol": "generator_protocol_v1",
        "total_parameters": total_params,
        "default_threshold_metrics": default_metrics,
        "youden_threshold_metrics": youden_metrics,
        "generator_recall": gen_rows,
        "model_weights_frozen": True,
        "selected_phase_frozen": True,
        "selected_epoch_frozen": True,
        "threshold_selected_clean_validation_only": True,
        "transformation_specific_thresholds": False,
        "generator_specific_thresholds": False,
        "known_test_accessed": False,
        "unseen_test_accessed": False,
        "training_performed": False,
        "optimizer_updates_performed": False,
        "checkpoint_changed": False,
    }
    regime.frozen_config_path.write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")

    return {
        "regime": regime,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "ref_auc": ref_auc,
        "ref_ap": ref_ap,
        "robust_val_auc": robust_val_auc,
        "youden_threshold": youden_threshold,
        "youden_j": youden_j,
        "default_metrics": default_metrics,
        "youden_metrics": youden_metrics,
        "generator_recall": generator_recall,
        "total_params": total_params,
        "frozen_config": frozen_config,
    }


def write_report(a0_cfg: dict, results: list[dict]) -> None:
    lines = [
        "RQ3 Threshold Selection Report — Stage 23C",
        "==========================================",
        "",
        "METHODOLOGY",
        "-----------",
        "Checkpoint selection used RobustValAUC (mean AUC over original, JPEG50,",
        "Resize112, Blur2 validation conditions).",
        "Operating thresholds are selected separately using Youden J on the",
        "ORIGINAL CLEAN validation split only (456 images).",
        "No transformed-validation thresholds.",
        "No transformation-specific or generator-specific thresholds.",
        "No test data accessed.",
        "",
        "A0 CLEAN — historical frozen reference (unchanged)",
        f"- checkpoint: {a0_cfg['checkpoint']}",
        f"- frozen threshold: {a0_cfg['threshold']:.12f}",
        f"- validation Youden J: {a0_cfg['validation_youden_j']:.8f}",
        f"- clean validation AUC: {a0_cfg['validation_roc_auc']:.8f}",
        f"- clean validation AP: {a0_cfg['validation_ap']:.8f}",
        "- A0 threshold changed in Stage 23C: NO",
        "",
    ]
    for res in results:
        regime = res["regime"]
        lines.extend(
            [
                f"{regime.key} — {regime.regime_id}",
                f"- checkpoint: {regime.checkpoint.relative_to(PROJECT_ROOT)}",
                f"- selected phase/epoch: {regime.selected_phase}/{regime.selected_epoch}",
                f"- checkpoint reproduction: PASSED",
                f"- Stage-23B clean AUC/AP: {res['ref_auc']:.8f} / {res['ref_ap']:.8f}",
                f"- reproduced clean AUC/AP: {res['roc_auc']:.8f} / {res['average_precision']:.8f}",
                f"- RobustValAUC (Stage 23B): {res['robust_val_auc']:.8f}",
                f"- primary RQ3 candidate: {regime.primary}",
                "",
                "Default threshold 0.50:",
                f"- accuracy={res['default_metrics']['accuracy']:.8f}",
                f"- balanced accuracy={res['default_metrics']['balanced_accuracy']:.8f}",
                f"- precision={res['default_metrics']['precision']:.8f}",
                f"- recall={res['default_metrics']['recall']:.8f}",
                f"- specificity={res['default_metrics']['specificity']:.8f}",
                f"- F1={res['default_metrics']['f1']:.8f}",
                f"- FPR={res['default_metrics']['false_positive_rate']:.8f}",
                f"- TN={res['default_metrics']['TN']} FP={res['default_metrics']['FP']} "
                f"FN={res['default_metrics']['FN']} TP={res['default_metrics']['TP']}",
                "",
                "Youden J threshold (clean validation):",
                f"- exact threshold: {res['youden_threshold']:.12f}",
                f"- Youden J: {res['youden_j']:.8f}",
                f"- accuracy={res['youden_metrics']['accuracy']:.8f}",
                f"- balanced accuracy={res['youden_metrics']['balanced_accuracy']:.8f}",
                f"- precision={res['youden_metrics']['precision']:.8f}",
                f"- recall={res['youden_metrics']['recall']:.8f}",
                f"- specificity={res['youden_metrics']['specificity']:.8f}",
                f"- F1={res['youden_metrics']['f1']:.8f}",
                f"- FPR={res['youden_metrics']['false_positive_rate']:.8f}",
                f"- TN={res['youden_metrics']['TN']} FP={res['youden_metrics']['FP']} "
                f"FN={res['youden_metrics']['FN']} TP={res['youden_metrics']['TP']}",
                "",
                "Generator-specific clean validation AI recall (diagnostic):",
                res["generator_recall"].to_string(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "SCIENTIFIC INTEGRITY",
            "Training performed: NO",
            "Checkpoint changes: NO",
            "Test access: NO",
            "Transformed-test predictions inspected: NO",
            "Threshold selected using clean validation only: YES",
            "Transformation-specific thresholds: NO",
            "Generator-specific thresholds: NO",
            "A0 threshold changed: NO",
            "Primary candidate changed: NO",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("STAGE 23C — RQ3 CLEAN-VALIDATION THRESHOLD SELECTION")
    device = select_device()
    print(f"Device: {device}")

    val_meta = load_split_metadata("validation", SPLIT_META_PATH)
    stop_if(len(val_meta) != EXPECTED_VAL, f"validation count {len(val_meta)} != {EXPECTED_VAL}")
    stop_if(val_meta["split"].isin(["known_test", "unseen_test"]).any(), "test rows in validation metadata")

    transform = build_imagenet_transforms()
    val_ds = ControlledV1Dataset("validation", transform=transform)
    stop_if(len(val_ds) != EXPECTED_VAL, f"validation dataset {len(val_ds)}")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    summary = pd.read_csv(SUMMARY_PATH)
    a0_cfg = json.loads(A0_FROZEN_PATH.read_text(encoding="utf-8"))
    a0_summary = summary[summary["regime"] == "A0"].iloc[0]

    results = []
    for regime in REGIMES:
        print(f"\nProcessing {regime.key} ({regime.regime_id})...")
        res = process_regime(regime, val_loader, val_ds.rows, device, summary)
        results.append(res)
        print(
            f"  reproduced AUC={res['roc_auc']:.6f} AP={res['average_precision']:.6f} "
            f"Youden={res['youden_threshold']:.12f} J={res['youden_j']:.6f}"
        )

    frozen_rows = [
        {
            "regime": "A0",
            "regime_id": "mobilenet_clean_v1",
            "parameters": int(a0_cfg["total_parameters"]),
            "selected_phase": a0_cfg["selected_phase"],
            "selected_epoch": a0_cfg["selected_epoch"],
            "clean_validation_auc": float(a0_cfg["validation_roc_auc"]),
            "robust_val_auc": float(a0_summary["robust_val_auc"]),
            "frozen_threshold": float(a0_cfg["threshold"]),
            "threshold_selection_method": a0_cfg["threshold_method"],
            "primary_candidate": False,
        }
    ]
    for res in results:
        regime = res["regime"]
        frozen_rows.append(
            {
                "regime": regime.key,
                "regime_id": regime.regime_id,
                "parameters": res["total_params"],
                "selected_phase": regime.selected_phase,
                "selected_epoch": regime.selected_epoch,
                "clean_validation_auc": res["roc_auc"],
                "robust_val_auc": res["robust_val_auc"],
                "frozen_threshold": res["youden_threshold"],
                "threshold_selection_method": THRESHOLD_METHOD,
                "primary_candidate": regime.primary,
            }
        )
    pd.DataFrame(frozen_rows).to_csv(FROZEN_SUMMARY_CSV, index=False)
    write_report(a0_cfg, results)

    print("\nSTAGE 23C — RQ3 THRESHOLD SELECTION COMPLETE")
    print(f"\nA0 CLEAN\nFrozen threshold: {a0_cfg['threshold']:.12f}")
    for res in results:
        print(f"\n{res['regime'].key} {res['regime'].augmentation.upper()}")
        print(f"Clean validation AUC: {res['roc_auc']:.6f}")
        print(f"Youden threshold: {res['youden_threshold']:.12f}")
        print(f"Youden J: {res['youden_j']:.6f}")
    print("\nPRIMARY RQ3 CANDIDATE:\nA2 Resize+JPEG")
    print("\nModel training: NO")
    print("Test access: NO")
    print("Transformation-specific thresholds: NO")
    print("\nA1 frozen: YES")
    print("A2 frozen: YES")
    print("A3 frozen: YES")
    print("\nRQ3 MODELS FULLY FROZEN")
    print("\nSTOP BEFORE TEST EVALUATION.")


if __name__ == "__main__":
    main()
