"""Validation-only threshold selection and baseline freeze (Stage 10).

Why this file exists
--------------------
The selected Logistic Regression model (C=1.0) is frozen. This script chooses
a validation-only Youden J threshold and records both the default (0.50) and
selected operating points. Test sets remain locked.

How to run
----------
    source .venv/bin/activate
    python src/select_logreg_threshold_v1.py

What to expect
--------------
    results/logreg_threshold_tradeoff_v1.csv
    results/logreg_frozen_baseline_v1.json
    results/logreg_threshold_selection_v1.txt
    figures/logreg_validation_threshold_tradeoff_v1.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# --- named constants ---
DEFAULT_THRESHOLD = 0.50
YOUDEN_J_TIE_TOLERANCE = 1e-12
MODEL_VERSION = "logreg_handcrafted_selected_v1"
FEATURE_VERSION = "handcrafted_features_v1"
REPRESENTATION_VERSION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"
SELECTED_C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 2000
THRESHOLD_METHOD = "Youden J"
THRESHOLD_SELECTION_DATA = "validation only"
ALLOWED_SPLITS = {"train", "validation"}
KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "metadata" / "development_features_v1.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "metadata" / "handcrafted_feature_columns_v1.txt"
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"
SELECTED_CONFIG_PATH = PROJECT_ROOT / "results" / "logreg_selected_configuration_v1.json"
TRADEOFF_CSV_PATH = PROJECT_ROOT / "results" / "logreg_threshold_tradeoff_v1.csv"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "results" / "logreg_frozen_baseline_v1.json"
REPORT_PATH = PROJECT_ROOT / "results" / "logreg_threshold_selection_v1.txt"
FIGURE_PATH = PROJECT_ROOT / "figures" / "logreg_validation_threshold_tradeoff_v1.png"

FORBIDDEN_X_COLUMNS = {
    "image_id",
    "processed_path",
    "raw_path",
    "path",
    "filename",
    "original_filename",
    "label",
    "generator",
    "split",
    "feature_version",
    "file_format",
    "format",
    "dimensions",
    "width",
    "height",
    "hash",
    "raw_sha256",
    "processed_sha256",
}


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_feature_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stop_if(len(names) != 13, f"expected 13 feature names, found {len(names)}")
    return names


def load_development_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    unexpected = set(table["split"].unique()) - ALLOWED_SPLITS
    stop_if(bool(unexpected), f"unexpected splits in development table: {unexpected}")
    n_known_test = int((table["split"] == "known_test").sum())
    n_unseen_test = int((table["split"] == "unseen_test").sum())
    stop_if(n_known_test != 0, f"known_test rows used = {n_known_test}")
    stop_if(n_unseen_test != 0, f"unseen_test rows used = {n_unseen_test}")
    return table


def make_xy(table: pd.DataFrame, feature_names: list[str], split: str):
    subset = table[table["split"] == split].copy()
    leaked = [name for name in feature_names if name in FORBIDDEN_X_COLUMNS]
    stop_if(bool(leaked), f"feature list contains metadata columns: {leaked}")
    overlap = set(feature_names) & set(FORBIDDEN_X_COLUMNS)
    stop_if(bool(overlap), f"X would include metadata: {overlap}")
    X = subset[feature_names].to_numpy(dtype=float)
    y = subset["label"].to_numpy(dtype=int)
    return subset, X, y


def verify_selected_model_config() -> None:
    config = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    stop_if(config["selected_C"] != SELECTED_C, f"expected C={SELECTED_C}, found {config['selected_C']}")
    stop_if(config["feature_version"] != FEATURE_VERSION, "feature version mismatch")
    stop_if(config["representation_version"] != REPRESENTATION_VERSION, "representation version mismatch")
    stop_if(config["split_protocol"] != SPLIT_PROTOCOL, "split protocol mismatch")


def confusion_parts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_parts(y_true, y_pred)
    specificity = true_negative / (true_negative + false_positive)
    false_positive_rate = false_positive / (false_positive + true_negative)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(false_positive_rate),
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "TP": true_positive,
    }


def select_youden_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Select threshold maximising Youden J with predefined tie-breaking."""
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
        }
    )
    candidates = candidates.sort_values(
        ["youden_j", "fpr", "threshold"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    selected_threshold = float(candidates.loc[0, "threshold"])
    selected_j = float(candidates.loc[0, "youden_j"])
    return selected_threshold, selected_j


def build_tradeoff_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for idx, threshold in enumerate(thresholds):
        if not np.isfinite(threshold):
            continue
        row = {
            "threshold": float(threshold),
            "tpr": float(tpr[idx]),
            "fpr": float(fpr[idx]),
            "specificity": float(1.0 - fpr[idx]),
            "youden_j": float(tpr[idx] - fpr[idx]),
        }
        y_pred = (y_prob >= threshold).astype(int)
        row["precision"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        row["f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)


def ai_recall_by_generator(
    val_rows: pd.DataFrame,
    y_val: np.ndarray,
    predicted_label: np.ndarray,
) -> pd.DataFrame:
    rows = []
    ai_val = val_rows.copy()
    ai_val["predicted_label"] = predicted_label
    ai_val["true_label"] = y_val
    for generator in KNOWN_AI_GENERATORS:
        group = ai_val[(ai_val["generator"] == generator) & (ai_val["true_label"] == 1)]
        n_ai = int(len(group))
        n_correct = int((group["predicted_label"] == 1).sum())
        recall = float(n_correct / n_ai) if n_ai else float("nan")
        rows.append(
            {
                "generator": generator,
                "ai_samples": n_ai,
                "correctly_detected_ai": n_correct,
                "ai_recall": recall,
            }
        )
    return pd.DataFrame(rows)


def save_tradeoff_figure(
    tradeoff: pd.DataFrame,
    default_threshold: float,
    youden_threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(tradeoff["fpr"], tradeoff["tpr"], color="steelblue")
    default_row = tradeoff.iloc[(tradeoff["threshold"] - default_threshold).abs().argsort()[:1]]
    youden_row = tradeoff.iloc[(tradeoff["threshold"] - youden_threshold).abs().argsort()[:1]]
    ax.scatter(default_row["fpr"], default_row["tpr"], color="orange", s=60, label=f"Default 0.50")
    ax.scatter(youden_row["fpr"], youden_row["tpr"], color="crimson", s=60, label="Youden threshold")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC with operating points")
    ax.legend()

    ax = axes[1]
    ax.plot(tradeoff["threshold"], tradeoff["youden_j"], color="steelblue")
    ax.axvline(default_threshold, color="orange", linestyle="--", label="Default 0.50")
    ax.axvline(youden_threshold, color="crimson", linestyle="--", label="Youden threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Youden J (TPR - FPR)")
    ax.set_title("Validation Youden J by threshold")
    ax.legend()

    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_report(
    default_metrics: dict,
    youden_metrics: dict,
    youden_threshold: float,
    youden_j: float,
    roc_auc: float,
    average_precision: float,
    generator_recall: pd.DataFrame,
) -> str:
    fp_change = youden_metrics["FP"] - default_metrics["FP"]
    fn_change = youden_metrics["FN"] - default_metrics["FN"]
    lines = [
        "Logistic Regression Threshold Selection V1",
        "============================================",
        "",
        "Why threshold selection was performed on validation",
        "---------------------------------------------------",
        "The selected model (logreg_handcrafted_selected_v1, C=1.0) was frozen after",
        "a validation-only regularisation study. Threshold selection must also use",
        "validation data only so that test sets remain untouched for later evaluation.",
        "",
        "Why test sets were not used",
        "---------------------------",
        "known_test and unseen_test are locked. Using them for threshold selection",
        "would leak test information into model development and invalidate later",
        "generalisation claims.",
        "",
        f"Validation ROC-AUC (threshold-independent): {roc_auc:.6f}",
        f"Validation Average Precision (threshold-independent): {average_precision:.6f}",
        "",
        "Default threshold metrics (reference operating point)",
        f"DEFAULT_THRESHOLD = {DEFAULT_THRESHOLD}",
        f"Accuracy: {default_metrics['accuracy']:.6f}",
        f"Balanced Accuracy: {default_metrics['balanced_accuracy']:.6f}",
        f"Precision: {default_metrics['precision']:.6f}",
        f"Recall: {default_metrics['recall']:.6f}",
        f"Specificity: {default_metrics['specificity']:.6f}",
        f"F1: {default_metrics['f1']:.6f}",
        f"False Positive Rate: {default_metrics['false_positive_rate']:.6f}",
        f"TN={default_metrics['TN']} FP={default_metrics['FP']} "
        f"FN={default_metrics['FN']} TP={default_metrics['TP']}",
        "",
        "Selected Youden threshold",
        f"VALIDATION_YOUDEN_THRESHOLD = {youden_threshold:.6f}",
        f"Youden J = {youden_j:.6f}",
        "",
        "Selected-threshold metrics",
        f"Accuracy: {youden_metrics['accuracy']:.6f}",
        f"Balanced Accuracy: {youden_metrics['balanced_accuracy']:.6f}",
        f"Precision: {youden_metrics['precision']:.6f}",
        f"Recall: {youden_metrics['recall']:.6f}",
        f"Specificity: {youden_metrics['specificity']:.6f}",
        f"F1: {youden_metrics['f1']:.6f}",
        f"False Positive Rate: {youden_metrics['false_positive_rate']:.6f}",
        f"TN={youden_metrics['TN']} FP={youden_metrics['FP']} "
        f"FN={youden_metrics['FN']} TP={youden_metrics['TP']}",
        "",
        "Change compared with default threshold 0.50",
        f"FP change: {fp_change:+d}",
        f"FN change: {fn_change:+d}",
        "",
        "Generator-specific validation AI recall at Youden threshold (diagnostic)",
        generator_recall.to_string(index=False),
        "",
        "Interpretation note",
        "This threshold provides a validation-selected balanced operating point for",
        "Baseline V1. Deployment-oriented thresholding will be investigated later",
        "using calibration and selective prediction.",
        "",
        "Test-set lock confirmation",
        "known_test images opened: 0",
        "unseen_test images opened: 0",
        "known_test features accessed: 0",
        "unseen_test features accessed: 0",
    ]
    return "\n".join(lines)


def main() -> None:
    verify_selected_model_config()
    feature_names = load_feature_names(FEATURE_LIST_PATH)
    table = load_development_table(FEATURES_PATH)
    val_rows, X_val, y_val = make_xy(table, feature_names, "validation")

    stop_if(len(val_rows) != 456, f"validation rows = {len(val_rows)}, expected 456")
    n_val_real = int((y_val == 0).sum())
    n_val_ai = int((y_val == 1).sum())
    stop_if(n_val_real != 228 or n_val_ai != 228, f"validation class counts {n_val_real}/{n_val_ai}")

    pipeline = joblib.load(MODEL_PATH)
    classifier = pipeline.named_steps["classifier"]
    stop_if(float(classifier.C) != SELECTED_C, f"loaded model C={classifier.C}, expected {SELECTED_C}")
    stop_if(classifier.solver != SOLVER, f"loaded solver={classifier.solver}, expected {SOLVER}")
    stop_if(int(classifier.max_iter) != MAX_ITER, f"loaded max_iter={classifier.max_iter}, expected {MAX_ITER}")

    ai_probability = pipeline.predict_proba(X_val)[:, 1]
    roc_auc = float(roc_auc_score(y_val, ai_probability))
    average_precision = float(average_precision_score(y_val, ai_probability))

    fpr, tpr, roc_thresholds = roc_curve(y_val, ai_probability)
    youden_threshold, youden_j = select_youden_threshold(fpr, tpr, roc_thresholds)

    default_metrics = threshold_metrics(y_val, ai_probability, DEFAULT_THRESHOLD)
    youden_metrics = threshold_metrics(y_val, ai_probability, youden_threshold)

    tradeoff = build_tradeoff_table(y_val, ai_probability, fpr, tpr, roc_thresholds)
    youden_pred = (ai_probability >= youden_threshold).astype(int)
    generator_recall = ai_recall_by_generator(val_rows, y_val, youden_pred)

    frozen_config = {
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "representation_version": REPRESENTATION_VERSION,
        "feature_version": FEATURE_VERSION,
        "split_protocol": SPLIT_PROTOCOL,
        "C": SELECTED_C,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "default_threshold": DEFAULT_THRESHOLD,
        "validation_selected_threshold": youden_threshold,
        "validation_youden_j": youden_j,
        "threshold_method": THRESHOLD_METHOD,
        "threshold_selection_data": THRESHOLD_SELECTION_DATA,
        "validation_roc_auc": roc_auc,
        "validation_average_precision": average_precision,
        "default_threshold_metrics": default_metrics,
        "validation_selected_threshold_metrics": youden_metrics,
        "known_test_accessed": False,
        "unseen_test_accessed": False,
        "known_test_images_opened": 0,
        "unseen_test_images_opened": 0,
        "known_test_features_accessed": 0,
        "unseen_test_features_accessed": 0,
    }

    TRADEOFF_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tradeoff.to_csv(TRADEOFF_CSV_PATH, index=False)
    FROZEN_CONFIG_PATH.write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")
    save_tradeoff_figure(tradeoff, DEFAULT_THRESHOLD, youden_threshold)
    report = write_report(
        default_metrics,
        youden_metrics,
        youden_threshold,
        youden_j,
        roc_auc,
        average_precision,
        generator_recall,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"VALIDATION_YOUDEN_THRESHOLD = {youden_threshold:.6f}")
    print(f"Youden J = {youden_j:.6f}")
    print(
        f"Default 0.50: TN={default_metrics['TN']} FP={default_metrics['FP']} "
        f"FN={default_metrics['FN']} TP={default_metrics['TP']}"
    )
    print(
        f"Youden:       TN={youden_metrics['TN']} FP={youden_metrics['FP']} "
        f"FN={youden_metrics['FN']} TP={youden_metrics['TP']}"
    )
    print(f"FP change: {youden_metrics['FP'] - default_metrics['FP']:+d}")
    print(f"FN change: {youden_metrics['FN'] - default_metrics['FN']:+d}")
    print(f"C remains: {SELECTED_C}")
    print("known_test/unseen_test untouched")
    print(f"Wrote {FROZEN_CONFIG_PATH}")


if __name__ == "__main__":
    main()
