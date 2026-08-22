"""Final test evaluation of the frozen classical baseline (Stage 11).

Why this file exists
--------------------
The model, features, regularisation, and thresholds were frozen before this
script runs. It loads the fitted pipeline without refitting and evaluates
known_test and unseen_test once. No post-test tuning is performed.

How to run
----------
    source .venv/bin/activate
    python src/extract_test_features_v1.py
    python src/evaluate_logreg_frozen_baseline_v1.py

What to expect
--------------
    results/logreg_known_test_predictions_v1.csv
    results/logreg_unseen_test_predictions_v1.csv
    results/logreg_test_ai_recall_by_generator_v1.csv
    results/logreg_known_vs_unseen_v1.csv
    results/logreg_known_test_failures_v1.csv
    results/logreg_unseen_test_failures_v1.csv
    results/logreg_final_test_report_v1.txt
    results/logreg_final_baseline_v1.json
    figures/logreg_known_test_confusion_matrix_v1.png
    figures/logreg_unseen_test_confusion_matrix_v1.png
    figures/logreg_known_vs_unseen_roc_v1.png
    figures/logreg_known_vs_unseen_pr_v1.png
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

MODEL_VERSION = "logreg_handcrafted_selected_v1"
FEATURE_VERSION = "handcrafted_features_v1"
REPRESENTATION_VERSION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"
SELECTED_C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 2000
DEFAULT_THRESHOLD = 0.50

KNOWN_AI_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15"]
UNSEEN_AI_GENERATORS = ["Midjourney", "VQDM", "Wukong"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FEATURES_PATH = PROJECT_ROOT / "metadata" / "test_features_v1.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "metadata" / "handcrafted_feature_columns_v1.txt"
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_VERSION}.joblib"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "results" / "logreg_frozen_baseline_v1.json"

KNOWN_PRED_PATH = PROJECT_ROOT / "results" / "logreg_known_test_predictions_v1.csv"
UNSEEN_PRED_PATH = PROJECT_ROOT / "results" / "logreg_unseen_test_predictions_v1.csv"
GENERATOR_RECALL_PATH = PROJECT_ROOT / "results" / "logreg_test_ai_recall_by_generator_v1.csv"
KNOWN_VS_UNSEEN_PATH = PROJECT_ROOT / "results" / "logreg_known_vs_unseen_v1.csv"
KNOWN_FAILURES_PATH = PROJECT_ROOT / "results" / "logreg_known_test_failures_v1.csv"
UNSEEN_FAILURES_PATH = PROJECT_ROOT / "results" / "logreg_unseen_test_failures_v1.csv"
FINAL_REPORT_PATH = PROJECT_ROOT / "results" / "logreg_final_test_report_v1.txt"
FINAL_JSON_PATH = PROJECT_ROOT / "results" / "logreg_final_baseline_v1.json"

KNOWN_CM_FIG = PROJECT_ROOT / "figures" / "logreg_known_test_confusion_matrix_v1.png"
UNSEEN_CM_FIG = PROJECT_ROOT / "figures" / "logreg_unseen_test_confusion_matrix_v1.png"
ROC_FIG = PROJECT_ROOT / "figures" / "logreg_known_vs_unseen_roc_v1.png"
PR_FIG = PROJECT_ROOT / "figures" / "logreg_known_vs_unseen_pr_v1.png"

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
}


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_feature_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stop_if(len(names) != 13, f"expected 13 feature names, found {len(names)}")
    return names


def load_frozen_thresholds() -> tuple[float, float]:
    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    stop_if(config["C"] != SELECTED_C, f"frozen C={config['C']}, expected {SELECTED_C}")
    stop_if(config["model_version"] != MODEL_VERSION, "model version mismatch")
    default_threshold = float(config["default_threshold"])
    youden_threshold = float(config["validation_selected_threshold"])
    return default_threshold, youden_threshold


def load_test_split(table: pd.DataFrame, feature_names: list[str], split: str):
    subset = table[table["split"] == split].copy()
    X = subset[feature_names].to_numpy(dtype=float)
    y = subset["label"].to_numpy(dtype=int)
    return subset, X, y


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


def evaluate_test_set(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_threshold: float,
    youden_threshold: float,
    test_condition: str,
) -> dict:
    threshold_independent = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }
    default_metrics = threshold_metrics(y_true, y_prob, default_threshold)
    youden_metrics = threshold_metrics(y_true, y_prob, youden_threshold)
    return {
        "test_condition": test_condition,
        "n_samples": int(len(y_true)),
        **threshold_independent,
        "default_threshold_metrics": default_metrics,
        "youden_threshold_metrics": youden_metrics,
    }


def generator_recall_rows(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    test_condition: str,
    generators: list[str],
    default_threshold: float,
    youden_threshold: float,
) -> list[dict]:
    pred_default = (y_prob >= default_threshold).astype(int)
    pred_youden = (y_prob >= youden_threshold).astype(int)
    out = []
    ai_rows = rows.copy()
    ai_rows["true_label"] = y_true
    ai_rows["pred_default"] = pred_default
    ai_rows["pred_youden"] = pred_youden
    for generator in generators:
        group = ai_rows[(ai_rows["generator"] == generator) & (ai_rows["true_label"] == 1)]
        n_ai = int(len(group))
        detected_050 = int((group["pred_default"] == 1).sum())
        detected_youden = int((group["pred_youden"] == 1).sum())
        out.append(
            {
                "test_condition": test_condition,
                "generator": generator,
                "ai_samples": n_ai,
                "detected_at_050": detected_050,
                "recall_at_050": float(detected_050 / n_ai) if n_ai else float("nan"),
                "detected_at_youden": detected_youden,
                "recall_at_youden": float(detected_youden / n_ai) if n_ai else float("nan"),
            }
        )
    return out


def save_predictions(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_threshold: float,
    youden_threshold: float,
    path: Path,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "image_id": rows["image_id"].to_numpy(),
            "true_label": y_true,
            "predicted_label_default_050": (y_prob >= default_threshold).astype(int),
            "predicted_label_youden": (y_prob >= youden_threshold).astype(int),
            "ai_probability": y_prob,
            "generator": rows["generator"].to_numpy(),
            "processed_path": rows["processed_path"].to_numpy(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return predictions


def save_failures(predictions: pd.DataFrame, youden_threshold: float, path: Path) -> pd.DataFrame:
    failures = predictions.copy()
    y_pred = failures["predicted_label_youden"].to_numpy()
    failure_type = np.full(len(failures), "", dtype=object)
    failure_type[(failures["true_label"] == 0) & (y_pred == 1)] = "false_positive"
    failure_type[(failures["true_label"] == 1) & (y_pred == 0)] = "false_negative"
    failures["predicted_label"] = y_pred
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
    failures.to_csv(path, index=False)
    return failures


def save_confusion_matrix_figure(metrics: dict, title: str, path: Path) -> None:
    matrix = np.array(
        [
            [metrics["TN"], metrics["FP"]],
            [metrics["FN"], metrics["TP"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted Real", "Predicted AI"])
    ax.set_yticks([0, 1], ["Actual Real", "Actual AI"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_figure(y_known: np.ndarray, p_known: np.ndarray, y_unseen: np.ndarray, p_unseen: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for y_true, y_prob, label in [
        (y_known, p_known, "known_test"),
        (y_unseen, p_unseen, "unseen_test"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Known vs unseen test ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROC_FIG, dpi=150)
    plt.close(fig)


def save_pr_figure(y_known: np.ndarray, p_known: np.ndarray, y_unseen: np.ndarray, p_unseen: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for y_true, y_prob, label in [
        (y_known, p_known, "known_test"),
        (y_unseen, p_unseen, "unseen_test"),
    ]:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Known vs unseen test precision-recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PR_FIG, dpi=150)
    plt.close(fig)


def build_known_vs_unseen_table(known_eval: dict, unseen_eval: dict) -> pd.DataFrame:
    rows = []
    for eval_dict in [known_eval, unseen_eval]:
        ym = eval_dict["youden_threshold_metrics"]
        rows.append(
            {
                "test_condition": eval_dict["test_condition"],
                "roc_auc": eval_dict["roc_auc"],
                "average_precision": eval_dict["average_precision"],
                "accuracy_youden": ym["accuracy"],
                "balanced_accuracy_youden": ym["balanced_accuracy"],
                "precision_youden": ym["precision"],
                "recall_youden": ym["recall"],
                "specificity_youden": ym["specificity"],
                "f1_youden": ym["f1"],
                "fpr_youden": ym["false_positive_rate"],
            }
        )
    table = pd.DataFrame(rows)
    diff = {
        "test_condition": "unseen_minus_known",
        "roc_auc": unseen_eval["roc_auc"] - known_eval["roc_auc"],
        "average_precision": unseen_eval["average_precision"] - known_eval["average_precision"],
        "accuracy_youden": unseen_eval["youden_threshold_metrics"]["accuracy"]
        - known_eval["youden_threshold_metrics"]["accuracy"],
        "balanced_accuracy_youden": unseen_eval["youden_threshold_metrics"]["balanced_accuracy"]
        - known_eval["youden_threshold_metrics"]["balanced_accuracy"],
        "precision_youden": unseen_eval["youden_threshold_metrics"]["precision"]
        - known_eval["youden_threshold_metrics"]["precision"],
        "recall_youden": unseen_eval["youden_threshold_metrics"]["recall"]
        - known_eval["youden_threshold_metrics"]["recall"],
        "specificity_youden": unseen_eval["youden_threshold_metrics"]["specificity"]
        - known_eval["youden_threshold_metrics"]["specificity"],
        "f1_youden": unseen_eval["youden_threshold_metrics"]["f1"]
        - known_eval["youden_threshold_metrics"]["f1"],
        "fpr_youden": unseen_eval["youden_threshold_metrics"]["false_positive_rate"]
        - known_eval["youden_threshold_metrics"]["false_positive_rate"],
    }
    return pd.concat([table, pd.DataFrame([diff])], ignore_index=True)


def write_final_report(
    known_eval: dict,
    unseen_eval: dict,
    generator_recall: pd.DataFrame,
    known_failures: pd.DataFrame,
    unseen_failures: pd.DataFrame,
    default_threshold: float,
    youden_threshold: float,
    comparison: pd.DataFrame,
) -> str:
    k_def = known_eval["default_threshold_metrics"]
    k_you = known_eval["youden_threshold_metrics"]
    u_def = unseen_eval["default_threshold_metrics"]
    u_you = unseen_eval["youden_threshold_metrics"]
    diff_row = comparison[comparison["test_condition"] == "unseen_minus_known"].iloc[0]

    lines = [
        "Frozen Classical Baseline — Final Test Report V1",
        "================================================",
        "",
        "1. FROZEN MODEL CONFIGURATION",
        f"model_version: {MODEL_VERSION}",
        f"model_path: models/{MODEL_VERSION}.joblib",
        f"representation_version: {REPRESENTATION_VERSION}",
        f"feature_version: {FEATURE_VERSION}",
        f"split_protocol: {SPLIT_PROTOCOL}",
        f"C: {SELECTED_C}",
        f"solver: {SOLVER}",
        f"max_iter: {MAX_ITER}",
        f"default_threshold: {default_threshold}",
        f"validation_selected_youden_threshold: {youden_threshold}",
        "Model and scaler were loaded without refitting.",
        "",
        "2. KNOWN-GENERATOR TEST RESULTS",
        f"ROC-AUC: {known_eval['roc_auc']:.6f}",
        f"Average Precision: {known_eval['average_precision']:.6f}",
        "Youden-threshold metrics:",
        f"  Accuracy={k_you['accuracy']:.6f}, Balanced Acc={k_you['balanced_accuracy']:.6f}, "
        f"Precision={k_you['precision']:.6f}, Recall={k_you['recall']:.6f}, "
        f"Specificity={k_you['specificity']:.6f}, F1={k_you['f1']:.6f}, FPR={k_you['false_positive_rate']:.6f}",
        f"  TN={k_you['TN']} FP={k_you['FP']} FN={k_you['FN']} TP={k_you['TP']}",
        "",
        "3. UNSEEN-GENERATOR TEST RESULTS",
        f"ROC-AUC: {unseen_eval['roc_auc']:.6f}",
        f"Average Precision: {unseen_eval['average_precision']:.6f}",
        "Youden-threshold metrics:",
        f"  Accuracy={u_you['accuracy']:.6f}, Balanced Acc={u_you['balanced_accuracy']:.6f}, "
        f"Precision={u_you['precision']:.6f}, Recall={u_you['recall']:.6f}, "
        f"Specificity={u_you['specificity']:.6f}, F1={u_you['f1']:.6f}, FPR={u_you['false_positive_rate']:.6f}",
        f"  TN={u_you['TN']} FP={u_you['FP']} FN={u_you['FN']} TP={u_you['TP']}",
        "",
        "4. KNOWN VS UNSEEN PERFORMANCE GAP (observed, not significance-tested)",
        f"ROC-AUC difference (unseen - known): {diff_row['roc_auc']:+.6f}",
        f"AP difference (unseen - known): {diff_row['average_precision']:+.6f}",
        f"Balanced Accuracy difference (Youden): {diff_row['balanced_accuracy_youden']:+.6f}",
        f"F1 difference (Youden): {diff_row['f1_youden']:+.6f}",
        f"AI Recall difference (Youden): {diff_row['recall_youden']:+.6f}",
        f"FPR difference (Youden): {diff_row['fpr_youden']:+.6f}",
        "",
        "5. GENERATOR-SPECIFIC RECALL",
        generator_recall.to_string(index=False),
        "",
        "6. DEFAULT VS YOUDEN THRESHOLD",
        "Known test at 0.50:",
        f"  TN={k_def['TN']} FP={k_def['FP']} FN={k_def['FN']} TP={k_def['TP']}",
        "Known test at Youden:",
        f"  TN={k_you['TN']} FP={k_you['FP']} FN={k_you['FN']} TP={k_you['TP']}",
        "Unseen test at 0.50:",
        f"  TN={u_def['TN']} FP={u_def['FP']} FN={u_def['FN']} TP={u_def['TP']}",
        "Unseen test at Youden:",
        f"  TN={u_you['TN']} FP={u_you['FP']} FN={u_you['FN']} TP={u_you['TP']}",
        "",
        "7. FAILURE COUNTS (Youden threshold)",
        f"Known false positives: {int((known_failures['failure_type'] == 'false_positive').sum())}",
        f"Known false negatives: {int((known_failures['failure_type'] == 'false_negative').sum())}",
        f"Unseen false positives: {int((unseen_failures['failure_type'] == 'false_positive').sum())}",
        f"Unseen false negatives: {int((unseen_failures['failure_type'] == 'false_negative').sum())}",
        "",
        "8. SCIENTIFIC INTERPRETATION",
    ]

    if diff_row["roc_auc"] < 0:
        lines.append(
            f"The unseen-generator ROC-AUC was lower than the known-generator ROC-AUC by "
            f"{abs(diff_row['roc_auc']):.6f} on this pilot split."
        )
    elif diff_row["roc_auc"] > 0:
        lines.append(
            f"The unseen-generator ROC-AUC was higher than the known-generator ROC-AUC by "
            f"{diff_row['roc_auc']:.6f} on this pilot split."
        )
    else:
        lines.append("The observed ROC-AUC values were equal on known and unseen tests.")

    lines.extend(
        [
            "These results describe one frozen classical baseline on a controlled pilot subset.",
            "They should not be interpreted as proof of reliable deployment performance.",
            "Generator-specific recall differences may reflect feature-generator coupling",
            "rather than a universal AI forensic signature.",
            "",
            "9. LIMITATIONS",
            "- Tiny-GenImage does not provide verified source IDs.",
            "- Exact and perceptual duplicate screening reduces but does not eliminate leakage risk.",
            "- controlled_v1 mitigates known format/geometry shortcuts but does not guarantee elimination of all biases.",
            "- Real images retain prior JPEG history.",
            "- Generator native resolutions may leave resampling artefacts.",
            "- Handcrafted features capture only simple image statistics.",
            "- Results from this pilot protocol should not automatically be generalised to every modern AI generator.",
            "",
            "Post-test lock confirmation",
            "model_retrained_after_test: false",
            "threshold_changed_after_test: false",
            "features_changed_after_test: false",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    feature_names = load_feature_names(FEATURE_LIST_PATH)
    default_threshold, youden_threshold = load_frozen_thresholds()

    table = pd.read_csv(TEST_FEATURES_PATH)
    stop_if(set(table["split"].unique()) - {"known_test", "unseen_test"}, "unexpected splits in test features")

    known_rows, X_known, y_known = load_test_split(table, feature_names, "known_test")
    unseen_rows, X_unseen, y_unseen = load_test_split(table, feature_names, "unseen_test")
    stop_if(len(known_rows) != 456, f"known_test rows = {len(known_rows)}")
    stop_if(len(unseen_rows) != 1712, f"unseen_test rows = {len(unseen_rows)}")

    pipeline = joblib.load(MODEL_PATH)
    classifier = pipeline.named_steps["classifier"]
    stop_if(float(classifier.C) != SELECTED_C, f"loaded model C={classifier.C}")
    stop_if(classifier.solver != SOLVER, f"loaded solver={classifier.solver}")

    prob_known = pipeline.predict_proba(X_known)[:, 1]
    prob_unseen = pipeline.predict_proba(X_unseen)[:, 1]

    known_eval = evaluate_test_set(
        known_rows, y_known, prob_known, default_threshold, youden_threshold, "known_test"
    )
    unseen_eval = evaluate_test_set(
        unseen_rows, y_unseen, prob_unseen, default_threshold, youden_threshold, "unseen_test"
    )

    known_predictions = save_predictions(
        known_rows, y_known, prob_known, default_threshold, youden_threshold, KNOWN_PRED_PATH
    )
    unseen_predictions = save_predictions(
        unseen_rows, y_unseen, prob_unseen, default_threshold, youden_threshold, UNSEEN_PRED_PATH
    )

    generator_rows = []
    generator_rows.extend(
        generator_recall_rows(
            known_rows, y_known, prob_known, "known_test", KNOWN_AI_GENERATORS, default_threshold, youden_threshold
        )
    )
    generator_rows.extend(
        generator_recall_rows(
            unseen_rows, y_unseen, prob_unseen, "unseen_test", UNSEEN_AI_GENERATORS, default_threshold, youden_threshold
        )
    )
    generator_recall = pd.DataFrame(generator_rows)
    generator_recall.to_csv(GENERATOR_RECALL_PATH, index=False)

    comparison = build_known_vs_unseen_table(known_eval, unseen_eval)
    comparison.to_csv(KNOWN_VS_UNSEEN_PATH, index=False)

    known_failures = save_failures(known_predictions, youden_threshold, KNOWN_FAILURES_PATH)
    unseen_failures = save_failures(unseen_predictions, youden_threshold, UNSEEN_FAILURES_PATH)

    save_confusion_matrix_figure(
        known_eval["youden_threshold_metrics"],
        "Known test confusion matrix (Youden threshold)",
        KNOWN_CM_FIG,
    )
    save_confusion_matrix_figure(
        unseen_eval["youden_threshold_metrics"],
        "Unseen test confusion matrix (Youden threshold)",
        UNSEEN_CM_FIG,
    )
    save_roc_figure(y_known, prob_known, y_unseen, prob_unseen)
    save_pr_figure(y_known, prob_known, y_unseen, prob_unseen)

    final_json = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "split_protocol": SPLIT_PROTOCOL,
        "C": SELECTED_C,
        "default_threshold": default_threshold,
        "youden_threshold": youden_threshold,
        "known_test": known_eval,
        "unseen_test": unseen_eval,
        "generator_recall": generator_rows,
        "known_vs_unseen_difference": comparison[comparison["test_condition"] == "unseen_minus_known"].iloc[0].to_dict(),
        "model_retrained_after_test": False,
        "threshold_changed_after_test": False,
        "features_changed_after_test": False,
    }
    FINAL_JSON_PATH.write_text(json.dumps(final_json, indent=2), encoding="utf-8")

    report = write_final_report(
        known_eval,
        unseen_eval,
        generator_recall,
        known_failures,
        unseen_failures,
        default_threshold,
        youden_threshold,
        comparison,
    )
    FINAL_REPORT_PATH.write_text(report, encoding="utf-8")

    print("Frozen baseline test evaluation complete")
    print(f"known_test ROC-AUC={known_eval['roc_auc']:.4f}  unseen_test ROC-AUC={unseen_eval['roc_auc']:.4f}")
    print(f"Youden threshold={youden_threshold}")
    print("model/scaler not refitted")
    print(f"Wrote {FINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
