"""Train Baseline V1: StandardScaler + Logistic Regression on train only.

Why this file exists
--------------------
This is the first official model-training step. The pipeline is fitted on
TRAIN features only. VALIDATION is used only to measure performance.

known_test and unseen_test are locked and must not appear in the
development feature table. This script does not extract test features,
tune C, change the threshold, or remove features.

How to run
----------
    source .venv/bin/activate
    python src/train_logistic_baseline_v1.py

What to expect
--------------
    models/logreg_handcrafted_v1.joblib
    results/validation_predictions_logreg_v1.csv
    results/logreg_validation_metrics_v1.json
    results/logreg_validation_report_v1.txt
    results/logreg_coefficients_v1.csv
    results/logreg_validation_ai_recall_by_generator_v1.csv
    results/logreg_validation_failures_v1.csv
    figures/logreg_validation_confusion_matrix_v1.png
    figures/logreg_validation_roc_v1.png
    figures/logreg_validation_pr_v1.png
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- named constants ---
MODEL_VERSION = "logreg_handcrafted_v1"
FEATURE_VERSION = "handcrafted_features_v1"
REPRESENTATION_VERSION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"
C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 2000
THRESHOLD = 0.50
ALLOWED_SPLITS = {"train", "validation"}
KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "metadata" / "development_features_v1.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "metadata" / "handcrafted_feature_columns_v1.txt"
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"
PRED_PATH = PROJECT_ROOT / "results" / "validation_predictions_logreg_v1.csv"
METRICS_JSON_PATH = PROJECT_ROOT / "results" / "logreg_validation_metrics_v1.json"
REPORT_PATH = PROJECT_ROOT / "results" / "logreg_validation_report_v1.txt"
COEF_PATH = PROJECT_ROOT / "results" / "logreg_coefficients_v1.csv"
GENERATOR_RECALL_PATH = PROJECT_ROOT / "results" / "logreg_validation_ai_recall_by_generator_v1.csv"
FAILURES_PATH = PROJECT_ROOT / "results" / "logreg_validation_failures_v1.csv"
CM_FIG_PATH = PROJECT_ROOT / "figures" / "logreg_validation_confusion_matrix_v1.png"
ROC_FIG_PATH = PROJECT_ROOT / "figures" / "logreg_validation_roc_v1.png"
PR_FIG_PATH = PROJECT_ROOT / "figures" / "logreg_validation_pr_v1.png"

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
    present_forbidden = [name for name in subset.columns if name in FORBIDDEN_X_COLUMNS]
    overlap = set(feature_names) & set(present_forbidden)
    stop_if(bool(overlap), f"X would include metadata: {overlap}")
    X = subset[feature_names].to_numpy(dtype=float)
    y = subset["label"].to_numpy(dtype=int)
    return subset, X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
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
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def save_confusion_matrix_figure(metrics: dict) -> None:
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted Real", "Predicted AI"])
    ax.set_yticks([0, 1], ["Actual Real", "Actual AI"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title("Validation confusion matrix (logreg_handcrafted_v1)")
    fig.tight_layout()
    CM_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CM_FIG_PATH, dpi=150)
    plt.close(fig)


def save_roc_figure(y_true: np.ndarray, y_prob: np.ndarray, roc_auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC (AI probability)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROC_FIG_PATH, dpi=150)
    plt.close(fig)


def save_pr_figure(y_true: np.ndarray, y_prob: np.ndarray, average_precision: float) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Logistic Regression (AP = {average_precision:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Validation precision-recall (AI probability)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PR_FIG_PATH, dpi=150)
    plt.close(fig)


def write_report(metrics: dict, coef: pd.DataFrame, generator_recall: pd.DataFrame, n_fp: int, n_fn: int, n_iter: int) -> str:
    lines = [
        "Logistic Regression Baseline V1 — validation report",
        "===================================================",
        "",
        f"model_version: {MODEL_VERSION}",
        f"feature_version: {FEATURE_VERSION}",
        f"representation_version: {REPRESENTATION_VERSION}",
        f"split_protocol: {SPLIT_PROTOCOL}",
        f"training_samples: {metrics['n_train']}",
        f"validation_samples: {metrics['n_validation']}",
        f"feature_count: {metrics['feature_count']}",
        f"C: {C}",
        f"solver: {SOLVER}",
        f"max_iter: {MAX_ITER}",
        f"threshold: {THRESHOLD}",
        f"convergence_status: {metrics['convergence_status']}",
        f"n_iter: {n_iter}",
        "",
        "Scaler note: StandardScaler is inside the sklearn Pipeline and is",
        "fitted only during model.fit(X_train, y_train). Validation was not",
        "used to compute scaling statistics.",
        "",
        "Validation metrics (AI-generated = positive class = 1)",
        "-----------------------------------------------------",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Balanced Accuracy: {metrics['balanced_accuracy']:.6f}",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall: {metrics['recall']:.6f}",
        f"Specificity: {metrics['specificity']:.6f}",
        f"F1-score: {metrics['f1']:.6f}",
        f"ROC-AUC: {metrics['roc_auc']:.6f}",
        f"Average Precision: {metrics['average_precision']:.6f}",
        f"False Positive Rate: {metrics['false_positive_rate']:.6f}",
        "",
        "Confusion matrix",
        f"TN (Actual Real, Predicted Real): {metrics['true_negative']}",
        f"FP (Actual Real, Predicted AI): {metrics['false_positive']}",
        f"FN (Actual AI, Predicted Real): {metrics['false_negative']}",
        f"TP (Actual AI, Predicted AI): {metrics['true_positive']}",
        "",
        "Coefficients (standardised features)",
        "A positive coefficient is associated with predictions toward AI.",
        "A negative coefficient is associated with predictions toward Real.",
        "This is not proof of causation or of an AI forensic signature.",
        coef.sort_values("absolute_coefficient", ascending=False).to_string(index=False),
        "",
        "AI recall by known validation generator (diagnostic)",
        generator_recall.to_string(index=False),
        "",
        f"False positives: {n_fp}",
        f"False negatives: {n_fn}",
        "",
        "known_test rows used: 0",
        "unseen_test rows used: 0",
    ]
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

    pipeline = build_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
        convergence_warnings = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        stop_if(bool(convergence_warnings), "Logistic Regression did not converge")

    classifier = pipeline.named_steps["classifier"]
    n_iter = int(np.max(classifier.n_iter_))
    stop_if(n_iter >= MAX_ITER, f"Logistic Regression hit max_iter={MAX_ITER}")
    convergence_status = "converged"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    ai_probability = pipeline.predict_proba(X_val)[:, 1]
    predicted_label = (ai_probability >= THRESHOLD).astype(int)
    metrics = calculate_metrics(y_val, predicted_label, ai_probability)
    metrics.update(
        {
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "representation_version": REPRESENTATION_VERSION,
            "split_protocol": SPLIT_PROTOCOL,
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "feature_count": 13,
            "C": C,
            "solver": SOLVER,
            "max_iter": MAX_ITER,
            "threshold": THRESHOLD,
            "convergence_status": convergence_status,
            "n_iter": n_iter,
            "known_test_rows_used": 0,
            "unseen_test_rows_used": 0,
            "x_train_shape": list(X_train.shape),
            "x_validation_shape": list(X_val.shape),
            "scaler_fitted_on": "train_only_via_pipeline",
        }
    )

    predictions = pd.DataFrame(
        {
            "image_id": val_rows["image_id"].to_numpy(),
            "true_label": y_val,
            "predicted_label": predicted_label,
            "ai_probability": ai_probability,
            "generator": val_rows["generator"].to_numpy(),
            "split": val_rows["split"].to_numpy(),
        }
    )
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PRED_PATH, index=False)

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": classifier.coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)
    coefficients.to_csv(COEF_PATH, index=False)

    generator_rows = []
    ai_val = val_rows.copy()
    ai_val["predicted_label"] = predicted_label
    ai_val["true_label"] = y_val
    for generator in KNOWN_AI_GENERATORS:
        group = ai_val[(ai_val["generator"] == generator) & (ai_val["true_label"] == 1)]
        n_ai = int(len(group))
        n_correct = int((group["predicted_label"] == 1).sum())
        recall = float(n_correct / n_ai) if n_ai else float("nan")
        generator_rows.append(
            {
                "generator": generator,
                "ai_samples": n_ai,
                "correctly_detected_ai": n_correct,
                "ai_recall": recall,
            }
        )
    generator_recall = pd.DataFrame(generator_rows)
    generator_recall.to_csv(GENERATOR_RECALL_PATH, index=False)

    failures = predictions.copy()
    failures["processed_path"] = val_rows["processed_path"].to_numpy()
    failure_type = np.full(len(failures), "", dtype=object)
    failure_type[(failures["true_label"] == 0) & (failures["predicted_label"] == 1)] = "false_positive"
    failure_type[(failures["true_label"] == 1) & (failures["predicted_label"] == 0)] = "false_negative"
    failures["failure_type"] = failure_type
    failures = failures[failures["failure_type"] != ""].copy()
    failures = failures[
        [
            "image_id",
            "true_label",
            "predicted_label",
            "ai_probability",
            "generator",
            "processed_path",
            "failure_type",
        ]
    ]
    failures.to_csv(FAILURES_PATH, index=False)

    save_confusion_matrix_figure(metrics)
    save_roc_figure(y_val, ai_probability, metrics["roc_auc"])
    save_pr_figure(y_val, ai_probability, metrics["average_precision"])

    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = write_report(
        metrics,
        coefficients,
        generator_recall,
        n_fp=metrics["false_positive"],
        n_fn=metrics["false_negative"],
        n_iter=n_iter,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Baseline V1 trained")
    print(f"X_train: {X_train.shape}  X_validation: {X_val.shape}")
    print(f"convergence: {convergence_status}  n_iter: {n_iter}")
    print(
        f"TN={metrics['true_negative']} FP={metrics['false_positive']} "
        f"FN={metrics['false_negative']} TP={metrics['true_positive']}"
    )
    print(f"Accuracy={metrics['accuracy']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    print("known_test rows used: 0")
    print("unseen_test rows used: 0")
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
