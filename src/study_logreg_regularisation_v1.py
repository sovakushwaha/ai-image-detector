"""Validation-only Logistic Regression regularisation study (Stage 9).

Why this file exists
--------------------
This script compares four predefined C values on the same train/validation
splits. Only regularisation strength changes. Test sets remain locked.

How to run
----------
    source .venv/bin/activate
    python src/study_logreg_regularisation_v1.py

What to expect
--------------
    results/logreg_regularisation_study_v1.csv
    results/logreg_regularisation_study_v1.txt
    results/logreg_regularisation_generator_recall_v1.csv
    results/logreg_selected_configuration_v1.json
    figures/logreg_regularisation_validation_v1.png
    models/logreg_handcrafted_selected_v1.joblib
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- named constants ---
C_VALUES = [0.01, 0.1, 1.0, 10.0]
BASELINE_C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 2000
THRESHOLD = 0.50
MODEL_VERSION = "logreg_handcrafted_selected_v1"
FEATURE_VERSION = "handcrafted_features_v1"
REPRESENTATION_VERSION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"
SELECTION_METRIC = "validation_roc_auc"
SELECTION_RULE = (
    "Primary: highest validation ROC-AUC. "
    "If ROC-AUC within 0.002, prefer higher Average Precision. "
    "If AP also within 0.002, prefer smaller C."
)
ROC_AUC_TIE_TOLERANCE = 0.002
AP_TIE_TOLERANCE = 0.002
ALLOWED_SPLITS = {"train", "validation"}
KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "metadata" / "development_features_v1.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "metadata" / "handcrafted_feature_columns_v1.txt"
STUDY_CSV_PATH = PROJECT_ROOT / "results" / "logreg_regularisation_study_v1.csv"
STUDY_REPORT_PATH = PROJECT_ROOT / "results" / "logreg_regularisation_study_v1.txt"
GENERATOR_RECALL_PATH = PROJECT_ROOT / "results" / "logreg_regularisation_generator_recall_v1.csv"
SELECTED_CONFIG_PATH = PROJECT_ROOT / "results" / "logreg_selected_configuration_v1.json"
SELECTED_MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"
FIGURE_PATH = PROJECT_ROOT / "figures" / "logreg_regularisation_validation_v1.png"

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


def build_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    solver=SOLVER,
                    max_iter=MAX_ITER,
                ),
            ),
        ]
    )


def confusion_parts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])
    return true_negative, false_positive, false_negative, true_positive


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    true_negative, false_positive, false_negative, true_positive = confusion_parts(y_true, y_pred)
    specificity = true_negative / (true_negative + false_positive)
    false_positive_rate = false_positive / (false_positive + true_negative)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "false_positive_rate": float(false_positive_rate),
        "TN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
        "TP": true_positive,
    }


def fit_and_evaluate(
    c_value: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict, Pipeline, np.ndarray, str, int]:
    pipeline = build_pipeline(c_value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
        convergence_warnings = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        stop_if(bool(convergence_warnings), f"C={c_value}: Logistic Regression did not converge")

    classifier = pipeline.named_steps["classifier"]
    n_iter = int(np.max(classifier.n_iter_))
    stop_if(n_iter >= MAX_ITER, f"C={c_value}: hit max_iter={MAX_ITER}")
    convergence_status = "converged"

    coef = classifier.coef_[0]
    coefficient_l2_norm = float(np.linalg.norm(coef))
    max_absolute_coefficient = float(np.max(np.abs(coef)))

    ai_probability = pipeline.predict_proba(X_val)[:, 1]
    predicted_label = (ai_probability >= THRESHOLD).astype(int)
    metrics = calculate_metrics(y_val, predicted_label, ai_probability)
    metrics.update(
        {
            "C": c_value,
            "iterations": n_iter,
            "coefficient_l2_norm": coefficient_l2_norm,
            "max_absolute_coefficient": max_absolute_coefficient,
            "convergence_status": convergence_status,
        }
    )
    return metrics, pipeline, ai_probability, convergence_status, n_iter


def ai_recall_by_generator(
    val_rows: pd.DataFrame,
    y_val: np.ndarray,
    predicted_label: np.ndarray,
    c_value: float,
) -> list[dict]:
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
                "C": c_value,
                "generator": generator,
                "ai_samples": n_ai,
                "correctly_detected_ai": n_correct,
                "ai_recall": recall,
            }
        )
    return rows


def select_c(study_table: pd.DataFrame) -> tuple[float, str]:
    """Apply the predefined selection rule without peeking at test data."""
    sorted_rows = study_table.sort_values(
        ["roc_auc", "average_precision", "C"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_roc = sorted_rows.loc[0, "roc_auc"]
    roc_tied = sorted_rows[np.isclose(sorted_rows["roc_auc"], best_roc, atol=ROC_AUC_TIE_TOLERANCE)]

    if len(roc_tied) == 1:
        selected = float(roc_tied.iloc[0]["C"])
        reason = (
            f"C={selected} had the highest validation ROC-AUC "
            f"({roc_tied.iloc[0]['roc_auc']:.6f}) with no tie within {ROC_AUC_TIE_TOLERANCE}."
        )
        return selected, reason

    best_ap = roc_tied["average_precision"].max()
    ap_tied = roc_tied[np.isclose(roc_tied["average_precision"], best_ap, atol=AP_TIE_TOLERANCE)]

    if len(ap_tied) == 1:
        selected = float(ap_tied.iloc[0]["C"])
        reason = (
            f"C={selected} tied on ROC-AUC within {ROC_AUC_TIE_TOLERANCE} and had the "
            f"highest Average Precision ({ap_tied.iloc[0]['average_precision']:.6f})."
        )
        return selected, reason

    selected = float(ap_tied["C"].min())
    reason = (
        f"C={selected} was chosen because ROC-AUC and Average Precision were both tied "
        f"within tolerance, so the smaller C was selected."
    )
    return selected, reason


def save_figure(study_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        study_table["C"],
        study_table["roc_auc"],
        marker="o",
        label="ROC-AUC",
    )
    ax.plot(
        study_table["C"],
        study_table["average_precision"],
        marker="s",
        label="Average Precision",
    )
    ax.set_xscale("log")
    ax.set_xlabel("C (regularisation strength, log scale)")
    ax.set_ylabel("Validation score")
    ax.set_title("Logistic Regression regularisation study (validation only)")
    ax.set_xticks(C_VALUES)
    ax.set_xticklabels([str(c) for c in C_VALUES])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_report(
    study_table: pd.DataFrame,
    generator_recall: pd.DataFrame,
    selected_c: float,
    selection_reason: str,
    baseline_row: pd.Series,
    selected_row: pd.Series,
) -> str:
    lines = [
        "Logistic Regression Regularisation Study V1",
        "=============================================",
        "",
        f"feature_version: {FEATURE_VERSION}",
        f"representation_version: {REPRESENTATION_VERSION}",
        f"split_protocol: {SPLIT_PROTOCOL}",
        f"threshold (diagnostic only): {THRESHOLD}",
        "",
        "Predefined C values",
        ", ".join(str(c) for c in C_VALUES),
        "",
        "Model-selection rule",
        SELECTION_RULE,
        "",
        "Results for all configurations",
        study_table.to_string(index=False),
        "",
        f"Selected C: {selected_c}",
        f"Selection reason: {selection_reason}",
        "",
        "Difference from Baseline V1 (C=1.0)",
        f"ROC-AUC change: {selected_row['roc_auc'] - baseline_row['roc_auc']:+.6f}",
        f"Average Precision change: {selected_row['average_precision'] - baseline_row['average_precision']:+.6f}",
        "",
        "Regularisation impact (cautious interpretation)",
    ]

    roc_range = study_table["roc_auc"].max() - study_table["roc_auc"].min()
    ap_range = study_table["average_precision"].max() - study_table["average_precision"].min()
    if roc_range < 0.01 and ap_range < 0.01:
        lines.append(
            "Across the predefined C grid, discrimination metrics changed only slightly. "
            "Regularisation strength did not materially alter validation ranking performance."
        )
    else:
        lines.append(
            "Regularisation strength produced measurable changes in validation discrimination "
            "or threshold-based metrics. Interpretation should remain cautious because differences "
            "may still be modest on this single validation split."
        )

    lines.extend(
        [
            "",
            "Generator-specific validation AI recall (diagnostic only)",
            generator_recall.to_string(index=False),
            "",
            "Coefficient magnitude behaviour",
            study_table[["C", "coefficient_l2_norm", "max_absolute_coefficient"]].to_string(index=False),
            "",
            "Convergence results",
            study_table[["C", "iterations", "convergence_status"]].to_string(index=False),
            "",
            "Test-set lock confirmation",
            "known_test features accessed: 0",
            "unseen_test features accessed: 0",
            "known_test images opened: 0",
            "unseen_test images opened: 0",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    feature_names = load_feature_names(FEATURE_LIST_PATH)
    table = load_development_table(FEATURES_PATH)

    train_rows, X_train, y_train = make_xy(table, feature_names, "train")
    val_rows, X_val, y_val = make_xy(table, feature_names, "validation")

    stop_if(X_train.shape != (1376, 13), f"X_train shape {X_train.shape} != (1376, 13)")
    stop_if(y_train.shape != (1376,), f"y_train shape {y_train.shape} != (1376,)")
    stop_if(X_val.shape != (456, 13), f"X_validation shape {X_val.shape} != (456, 13)")
    stop_if(y_val.shape != (456,), f"y_validation shape {y_val.shape} != (456,)")

    n_train_real = int((y_train == 0).sum())
    n_train_ai = int((y_train == 1).sum())
    n_val_real = int((y_val == 0).sum())
    n_val_ai = int((y_val == 1).sum())
    stop_if(n_train_real != 688 or n_train_ai != 688, f"train class counts {n_train_real}/{n_train_ai}")
    stop_if(n_val_real != 228 or n_val_ai != 228, f"validation class counts {n_val_real}/{n_val_ai}")

    study_rows = []
    generator_rows = []

    for c_value in C_VALUES:
        metrics, _, ai_probability, _, _ = fit_and_evaluate(
            c_value, X_train, y_train, X_val, y_val
        )
        predicted_label = (ai_probability >= THRESHOLD).astype(int)
        study_rows.append(metrics)
        generator_rows.extend(ai_recall_by_generator(val_rows, y_val, predicted_label, c_value))
        print(
            f"C={c_value}: ROC-AUC={metrics['roc_auc']:.4f}  "
            f"AP={metrics['average_precision']:.4f}  "
            f"L2={metrics['coefficient_l2_norm']:.4f}  "
            f"iter={metrics['iterations']}"
        )

    study_table = pd.DataFrame(study_rows).sort_values("C").reset_index(drop=True)
    generator_recall = pd.DataFrame(generator_rows)

    selected_c, selection_reason = select_c(study_table)
    selected_row = study_table.loc[study_table["C"] == selected_c].iloc[0]
    baseline_row = study_table.loc[study_table["C"] == BASELINE_C].iloc[0]

    selected_pipeline = build_pipeline(selected_c)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        selected_pipeline.fit(X_train, y_train)
        convergence_warnings = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        stop_if(bool(convergence_warnings), f"selected C={selected_c} did not converge")

    SELECTED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, SELECTED_MODEL_PATH)

    selected_config = {
        "model_version": MODEL_VERSION,
        "selected_C": selected_c,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "feature_version": FEATURE_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "split_protocol": SPLIT_PROTOCOL,
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
        "selection_reason": selection_reason,
        "threshold_diagnostic_only": THRESHOLD,
        "known_test_features_accessed": 0,
        "unseen_test_features_accessed": 0,
        "known_test_images_opened": 0,
        "unseen_test_images_opened": 0,
    }

    STUDY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    study_table.to_csv(STUDY_CSV_PATH, index=False)
    generator_recall.to_csv(GENERATOR_RECALL_PATH, index=False)
    SELECTED_CONFIG_PATH.write_text(json.dumps(selected_config, indent=2), encoding="utf-8")
    save_figure(study_table)
    report = write_report(
        study_table,
        generator_recall,
        selected_c,
        selection_reason,
        baseline_row,
        selected_row,
    )
    STUDY_REPORT_PATH.write_text(report, encoding="utf-8")

    print("")
    print(f"Selected C: {selected_c}")
    print(selection_reason)
    print(f"Wrote {STUDY_CSV_PATH}")
    print(f"Wrote {SELECTED_MODEL_PATH}")


if __name__ == "__main__":
    main()
