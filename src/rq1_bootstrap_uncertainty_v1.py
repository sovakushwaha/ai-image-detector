"""RQ1 bootstrap uncertainty analysis for frozen test metrics (Stage 21A).

Why this file exists
--------------------
Frozen RQ1 test predictions are analysed with stratified bootstrap resampling
to quantify uncertainty around discrimination metrics and paired model
comparisons. No training, inference, or threshold changes.

How to run
----------
    source .venv/bin/activate
    python src/rq1_bootstrap_uncertainty_v1.py

What to expect
--------------
    results/rq1_bootstrap_uncertainty_v1.json
    results/rq1_bootstrap_uncertainty_v1.csv
    results/rq1_bootstrap_uncertainty_report_v1.txt
    figures/rq1_auc_confidence_intervals_v1.png
    figures/rq1_pretrained_auc_difference_bootstrap_v1.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
METHOD = "stratified_bootstrap_percentile_ci"

EXPECTED_COUNTS = {
    "known_test": {"total": 456, "real": 228, "ai": 228},
    "unseen_test": {"total": 1712, "real": 856, "ai": 856},
}

MODEL_PREDICTIONS = {
    "LogReg": {
        "known_test": PROJECT_ROOT / "results" / "logreg_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results" / "logreg_unseen_test_predictions_v1.csv",
    },
    "SmallCNNV1": {
        "known_test": PROJECT_ROOT / "results" / "smallcnn_v1_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results" / "smallcnn_v1_unseen_test_predictions_v1.csv",
    },
    "MobileNetV3-Small": {
        "known_test": PROJECT_ROOT / "results" / "mobilenet_v3_small_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results" / "mobilenet_v3_small_unseen_test_predictions_v1.csv",
    },
    "EfficientNet-B0": {
        "known_test": PROJECT_ROOT / "results" / "efficientnet_b0_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results" / "efficientnet_b0_unseen_test_predictions_v1.csv",
    },
}

JSON_PATH = PROJECT_ROOT / "results" / "rq1_bootstrap_uncertainty_v1.json"
CSV_PATH = PROJECT_ROOT / "results" / "rq1_bootstrap_uncertainty_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "rq1_bootstrap_uncertainty_report_v1.txt"
AUC_CI_FIG_PATH = PROJECT_ROOT / "figures" / "rq1_auc_confidence_intervals_v1.png"
PAIRED_DIFF_FIG_PATH = PROJECT_ROOT / "figures" / "rq1_pretrained_auc_difference_bootstrap_v1.png"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_predictions(path: Path) -> pd.DataFrame:
    stop_if(not path.exists(), f"missing prediction file: {path}")
    df = pd.read_csv(path)
    stop_if("image_id" not in df.columns, f"{path} missing image_id column")
    stop_if("true_label" not in df.columns, f"{path} missing true_label column")
    stop_if("ai_probability" not in df.columns, f"{path} missing ai_probability column")
    return df.sort_values("image_id").reset_index(drop=True)


def verify_split_metadata(split: str, image_ids: set[str]) -> None:
    meta = pd.read_csv(SPLIT_META_PATH)
    split_rows = meta[meta["split"] == split]
    expected_ids = set(split_rows["image_id"].astype(str))
    stop_if(image_ids != expected_ids, f"{split} image_id set does not match split metadata")


def align_split_predictions(split: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """Align all model predictions for one split using image_id."""
    aligned_frames: dict[str, pd.DataFrame] = {}
    for model, paths in MODEL_PREDICTIONS.items():
        aligned_frames[model] = load_predictions(paths[split])

    reference = aligned_frames["LogReg"]
    reference_ids = reference["image_id"].astype(str).tolist()
    y_true = reference["true_label"].to_numpy(dtype=int)

    expected = EXPECTED_COUNTS[split]
    stop_if(len(reference) != expected["total"], f"{split} count {len(reference)} != {expected['total']}")
    stop_if(int((y_true == 0).sum()) != expected["real"], f"{split} real count mismatch")
    stop_if(int((y_true == 1).sum()) != expected["ai"], f"{split} ai count mismatch")
    verify_split_metadata(split, set(reference_ids))

    probs_by_model: dict[str, np.ndarray] = {}
    for model, frame in aligned_frames.items():
        frame_ids = frame["image_id"].astype(str).tolist()
        stop_if(frame_ids != reference_ids, f"{model} {split} row order/image_id mismatch vs LogReg")
        labels = frame["true_label"].to_numpy(dtype=int)
        stop_if(not np.array_equal(labels, y_true), f"{model} {split} labels mismatch vs LogReg")
        probs_by_model[model] = frame["ai_probability"].to_numpy(dtype=float)

    return reference, y_true, probs_by_model


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    real_idx = np.flatnonzero(y_true == 0)
    ai_idx = np.flatnonzero(y_true == 1)
    sampled_real = rng.choice(real_idx, size=len(real_idx), replace=True)
    sampled_ai = rng.choice(ai_idx, size=len(ai_idx), replace=True)
    return np.concatenate([sampled_real, sampled_ai])


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def summarize_bootstrap(
    observed: float,
    samples: np.ndarray,
) -> dict:
    clean = samples[np.isfinite(samples)]
    return {
        "observed": float(observed),
        "bootstrap_mean": float(np.mean(clean)),
        "bootstrap_std": float(np.std(clean, ddof=1)),
        "ci_95_low": float(np.percentile(clean, 2.5)),
        "ci_95_high": float(np.percentile(clean, 97.5)),
    }


def bootstrap_model_metrics(
    y_true: np.ndarray,
    probs_by_model: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model, probs in probs_by_model.items():
        out[model] = {
            "roc_auc": safe_roc_auc(y_true, probs),
            "average_precision": safe_average_precision(y_true, probs),
        }
    return out


def precompute_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.empty((N_BOOTSTRAP, len(y_true)), dtype=int)
    for b in range(N_BOOTSTRAP):
        indices[b] = stratified_bootstrap_indices(y_true, rng)
    return indices


def run_split_bootstrap(
    split: str,
    y_true: np.ndarray,
    probs_by_model: dict[str, np.ndarray],
    bootstrap_indices: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    observed = bootstrap_model_metrics(y_true, probs_by_model)
    auc_samples = {model: np.empty(N_BOOTSTRAP, dtype=float) for model in probs_by_model}
    ap_samples = {model: np.empty(N_BOOTSTRAP, dtype=float) for model in probs_by_model}

    for b, idx in enumerate(bootstrap_indices):
        y_boot = y_true[idx]
        for model, probs in probs_by_model.items():
            p_boot = probs[idx]
            auc_samples[model][b] = safe_roc_auc(y_boot, p_boot)
            ap_samples[model][b] = safe_average_precision(y_boot, p_boot)

    summaries = {}
    for model in probs_by_model:
        summaries[model] = {
            "roc_auc": summarize_bootstrap(observed[model]["roc_auc"], auc_samples[model]),
            "average_precision": summarize_bootstrap(
                observed[model]["average_precision"], ap_samples[model]
            ),
        }
    return summaries, auc_samples, ap_samples


def run_gap_bootstrap(
    known_y: np.ndarray,
    known_probs: dict[str, np.ndarray],
    unseen_y: np.ndarray,
    unseen_probs: dict[str, np.ndarray],
    known_indices: np.ndarray,
    unseen_indices: np.ndarray,
) -> dict[str, dict[str, dict]]:
    observed = {}
    auc_gap_samples = {model: np.empty(N_BOOTSTRAP, dtype=float) for model in known_probs}
    ap_gap_samples = {model: np.empty(N_BOOTSTRAP, dtype=float) for model in known_probs}

    for model in known_probs:
        known_auc = safe_roc_auc(known_y, known_probs[model])
        unseen_auc = safe_roc_auc(unseen_y, unseen_probs[model])
        known_ap = safe_average_precision(known_y, known_probs[model])
        unseen_ap = safe_average_precision(unseen_y, unseen_probs[model])
        observed[model] = {
            "roc_auc_gap": unseen_auc - known_auc,
            "average_precision_gap": unseen_ap - known_ap,
        }

    for b in range(N_BOOTSTRAP):
        known_idx = known_indices[b]
        unseen_idx = unseen_indices[b]
        known_y_boot = known_y[known_idx]
        unseen_y_boot = unseen_y[unseen_idx]
        for model in known_probs:
            known_auc = safe_roc_auc(known_y_boot, known_probs[model][known_idx])
            unseen_auc = safe_roc_auc(unseen_y_boot, unseen_probs[model][unseen_idx])
            known_ap = safe_average_precision(known_y_boot, known_probs[model][known_idx])
            unseen_ap = safe_average_precision(unseen_y_boot, unseen_probs[model][unseen_idx])
            auc_gap_samples[model][b] = unseen_auc - known_auc
            ap_gap_samples[model][b] = unseen_ap - known_ap

    summaries = {}
    for model in known_probs:
        summaries[model] = {
            "roc_auc_gap": summarize_bootstrap(observed[model]["roc_auc_gap"], auc_gap_samples[model]),
            "average_precision_gap": summarize_bootstrap(
                observed[model]["average_precision_gap"], ap_gap_samples[model]
            ),
        }
    return summaries


def run_paired_difference_bootstrap(
    split: str,
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    model_a: str,
    model_b: str,
    bootstrap_indices: np.ndarray,
) -> dict:
    observed_auc = safe_roc_auc(y_true, probs_a) - safe_roc_auc(y_true, probs_b)
    observed_ap = safe_average_precision(y_true, probs_a) - safe_average_precision(y_true, probs_b)
    auc_diff = np.empty(N_BOOTSTRAP, dtype=float)
    ap_diff = np.empty(N_BOOTSTRAP, dtype=float)

    for b, idx in enumerate(bootstrap_indices):
        y_boot = y_true[idx]
        auc_diff[b] = safe_roc_auc(y_boot, probs_a[idx]) - safe_roc_auc(y_boot, probs_b[idx])
        ap_diff[b] = safe_average_precision(y_boot, probs_a[idx]) - safe_average_precision(y_boot, probs_b[idx])

    return {
        "split": split,
        "model_a": model_a,
        "model_b": model_b,
        "comparison": f"{model_a} minus {model_b}",
        "roc_auc_difference": summarize_bootstrap(observed_auc, auc_diff),
        "average_precision_difference": summarize_bootstrap(observed_ap, ap_diff),
        "roc_auc_difference_samples": auc_diff,
    }


def ci_includes_zero(summary: dict) -> bool:
    return summary["ci_95_low"] <= 0.0 <= summary["ci_95_high"]


def interpretation_phrase(summary: dict) -> str:
    if ci_includes_zero(summary):
        return (
            "The observed difference is not clearly distinguishable from zero "
            "under this bootstrap analysis (95% percentile CI includes zero)."
        )
    direction = "positive" if summary["observed"] > 0 else "negative"
    return (
        f"The observed {direction} difference is not clearly distinguishable from zero "
        "under this bootstrap analysis when interpreted cautiously; "
        "the 95% percentile CI excludes zero for this fixed test sample only."
    )


def build_tidy_rows(results: dict) -> pd.DataFrame:
    rows = []
    for split in ("known_test", "unseen_test"):
        for model, metrics in results["model_confidence_intervals"][split].items():
            for metric_name, summary in metrics.items():
                rows.append(
                    {
                        "analysis_type": "model_metric",
                        "split": split,
                        "model": model,
                        "metric": metric_name,
                        "observed": summary["observed"],
                        "bootstrap_mean": summary["bootstrap_mean"],
                        "bootstrap_std": summary["bootstrap_std"],
                        "ci_95_low": summary["ci_95_low"],
                        "ci_95_high": summary["ci_95_high"],
                        "comparison": None,
                    }
                )

    for model, metrics in results["generalisation_gap_confidence_intervals"].items():
        for metric_name, summary in metrics.items():
            rows.append(
                {
                    "analysis_type": "generalisation_gap",
                    "split": "known_to_unseen",
                    "model": model,
                    "metric": metric_name,
                    "observed": summary["observed"],
                    "bootstrap_mean": summary["bootstrap_mean"],
                    "bootstrap_std": summary["bootstrap_std"],
                    "ci_95_low": summary["ci_95_low"],
                    "ci_95_high": summary["ci_95_high"],
                    "comparison": None,
                }
            )

    for key in ("primary_paired_comparison", "secondary_paired_comparisons"):
        comparisons = results[key]
        if key == "primary_paired_comparison":
            comparisons = list(comparisons.values())
        elif isinstance(comparisons, dict) and "comparison" in comparisons:
            comparisons = [comparisons]
        for comp in comparisons:
            for metric_name in ("roc_auc_difference", "average_precision_difference"):
                summary = comp[metric_name]
                rows.append(
                    {
                        "analysis_type": "paired_difference",
                        "split": comp["split"],
                        "model": None,
                        "metric": metric_name,
                        "observed": summary["observed"],
                        "bootstrap_mean": summary["bootstrap_mean"],
                        "bootstrap_std": summary["bootstrap_std"],
                        "ci_95_low": summary["ci_95_low"],
                        "ci_95_high": summary["ci_95_high"],
                        "comparison": comp["comparison"],
                    }
                )
    return pd.DataFrame(rows)


def save_auc_ci_figure(model_cis: dict) -> None:
    models = ["LogReg", "SmallCNNV1", "MobileNetV3-Small", "EfficientNet-B0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    titles = ["Known test ROC-AUC", "Unseen test ROC-AUC"]
    splits = ["known_test", "unseen_test"]

    for ax, split, title in zip(axes, splits, titles):
        y_pos = np.arange(len(models))
        obs = [model_cis[split][m]["roc_auc"]["observed"] for m in models]
        low = [model_cis[split][m]["roc_auc"]["ci_95_low"] for m in models]
        high = [model_cis[split][m]["roc_auc"]["ci_95_high"] for m in models]
        err_low = np.array(obs) - np.array(low)
        err_high = np.array(high) - np.array(obs)
        ax.errorbar(
            obs,
            y_pos,
            xerr=[err_low, err_high],
            fmt="o",
            color="#4C72B0",
            ecolor="#4C72B0",
            capsize=4,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel("ROC-AUC")
        ax.set_title(title)
        ax.set_xlim(0.55, 1.0)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("RQ1 frozen test ROC-AUC point estimates with 95% bootstrap CIs")
    fig.tight_layout()
    AUC_CI_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(AUC_CI_FIG_PATH, dpi=150)
    plt.close(fig)


def save_paired_diff_figure(samples: np.ndarray, observed: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=50, color="#DD8452", alpha=0.85, edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero difference")
    ax.axvline(observed, color="#4C72B0", linestyle="-", linewidth=1.2, label=f"Observed = {observed:.4f}")
    ax.set_xlabel("EfficientNet-B0 unseen AUC − MobileNetV3-Small unseen AUC")
    ax.set_ylabel("Bootstrap replicate count")
    ax.set_title("Paired bootstrap distribution (5,000 stratified resamples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PAIRED_DIFF_FIG_PATH, dpi=150)
    plt.close(fig)


def format_ci(summary: dict) -> str:
    return f"[{summary['ci_95_low']:.4f}, {summary['ci_95_high']:.4f}]"


def write_report(results: dict) -> str:
    lines = [
        "RQ1 Bootstrap Uncertainty Analysis — Stage 21A",
        "==============================================",
        "",
        "METHOD",
        f"- seed: {results['seed']}",
        f"- bootstrap replicates: {results['bootstrap_replicates']}",
        f"- method: {results['method']}",
        "- stratified resampling within each test split (Real and AI counts preserved)",
        "- paired resampling uses identical bootstrap indices across models within a split",
        "- generalisation gaps resample known_test and unseen_test independently per replicate",
        "",
        "MODEL-WISE CONFIDENCE INTERVALS",
    ]

    for split in ("known_test", "unseen_test"):
        lines.append(f"\n{split.upper()}")
        for model, metrics in results["model_confidence_intervals"][split].items():
            auc = metrics["roc_auc"]
            ap = metrics["average_precision"]
            lines.append(
                f"- {model}: ROC-AUC observed={auc['observed']:.6f}, "
                f"95% CI {format_ci(auc)}; AP observed={ap['observed']:.6f}, "
                f"95% CI {format_ci(ap)}"
            )

    lines.extend(["", "GENERALISATION-GAP CONFIDENCE INTERVALS"])
    for model, metrics in results["generalisation_gap_confidence_intervals"].items():
        auc_gap = metrics["roc_auc_gap"]
        ap_gap = metrics["average_precision_gap"]
        lines.append(
            f"- {model}: AUC gap observed={auc_gap['observed']:+.6f}, "
            f"95% CI {format_ci(auc_gap)}; AP gap observed={ap_gap['observed']:+.6f}, "
            f"95% CI {format_ci(ap_gap)}"
        )

    lines.extend(["", "MOBILENET VS EFFICIENTNET PAIRED COMPARISON"])
    for split in ("known_test", "unseen_test"):
        comp = results["primary_paired_comparison"][split]
        auc = comp["roc_auc_difference"]
        ap = comp["average_precision_difference"]
        lines.append(f"\n{split.upper()}")
        lines.append(
            f"- ROC-AUC difference observed={auc['observed']:+.6f}, "
            f"bootstrap mean={auc['bootstrap_mean']:+.6f}, 95% CI {format_ci(auc)}"
        )
        lines.append(f"  {results['interpretation'][split]['roc_auc_difference']}")
        lines.append(
            f"- AP difference observed={ap['observed']:+.6f}, "
            f"bootstrap mean={ap['bootstrap_mean']:+.6f}, 95% CI {format_ci(ap)}"
        )
        lines.append(f"  {results['interpretation'][split]['average_precision_difference']}")

    if results["secondary_paired_comparisons"]:
        lines.extend(["", "SECONDARY PAIRED COMPARISONS (unseen_test)"])
        for comp in results["secondary_paired_comparisons"]:
            auc = comp["roc_auc_difference"]
            ap = comp["average_precision_difference"]
            lines.append(
                f"- {comp['comparison']}: ROC-AUC diff observed={auc['observed']:+.6f}, "
                f"95% CI {format_ci(auc)}; AP diff observed={ap['observed']:+.6f}, "
                f"95% CI {format_ci(ap)}"
            )

    lines.extend(
        [
            "",
            "LIMITATIONS",
            "- Bootstrap uncertainty reflects the current fixed test samples only.",
            "- Does not account for uncertainty from different generator holdout choices,",
            "  training seeds, dataset sampling, unavailable source IDs, or broader",
            "  real-world generator populations.",
            "- Confidence intervals must not be interpreted as causal evidence when",
            "  excluding zero, nor as proof of equivalence when including zero.",
            "",
            "SCIENTIFIC INTEGRITY",
            "- Model training performed: NO",
            "- New inference-based tuning: NO",
            "- Threshold changed: NO",
            "- Samples removed after results: NO",
            "- Architecture selection changed: NO",
            "- RQ1 baseline development reopened: NO",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    _, known_y, known_probs = align_split_predictions("known_test")
    _, unseen_y, unseen_probs = align_split_predictions("unseen_test")

    known_indices = precompute_bootstrap_indices(known_y, rng)
    unseen_indices = precompute_bootstrap_indices(unseen_y, rng)

    known_summaries, _, _ = run_split_bootstrap("known_test", known_y, known_probs, known_indices)
    unseen_summaries, _, _ = run_split_bootstrap("unseen_test", unseen_y, unseen_probs, unseen_indices)
    gap_summaries = run_gap_bootstrap(
        known_y, known_probs, unseen_y, unseen_probs, known_indices, unseen_indices
    )

    primary = {}
    primary_samples = {}
    interpretation = {}
    for split, y_true, probs, indices in (
        ("known_test", known_y, known_probs, known_indices),
        ("unseen_test", unseen_y, unseen_probs, unseen_indices),
    ):
        comp = run_paired_difference_bootstrap(
            split,
            y_true,
            probs["EfficientNet-B0"],
            probs["MobileNetV3-Small"],
            "EfficientNet-B0",
            "MobileNetV3-Small",
            indices,
        )
        primary[split] = {
            k: v for k, v in comp.items() if k != "roc_auc_difference_samples"
        }
        primary_samples[split] = comp["roc_auc_difference_samples"]
        interpretation[split] = {
            "roc_auc_difference": interpretation_phrase(comp["roc_auc_difference"]),
            "average_precision_difference": interpretation_phrase(comp["average_precision_difference"]),
        }

    secondary = []
    for model_a, model_b in (
        ("MobileNetV3-Small", "SmallCNNV1"),
        ("EfficientNet-B0", "SmallCNNV1"),
    ):
        comp = run_paired_difference_bootstrap(
            "unseen_test",
            unseen_y,
            unseen_probs[model_a],
            unseen_probs[model_b],
            model_a,
            model_b,
            unseen_indices,
        )
        secondary.append({k: v for k, v in comp.items() if k != "roc_auc_difference_samples"})

    results = {
        "seed": RANDOM_SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "method": METHOD,
        "model_confidence_intervals": {
            "known_test": known_summaries,
            "unseen_test": unseen_summaries,
        },
        "generalisation_gap_confidence_intervals": gap_summaries,
        "primary_paired_comparison": primary,
        "secondary_paired_comparisons": secondary,
        "interpretation": interpretation,
        "limitations": [
            "Bootstrap uncertainty reflects the current fixed test samples only.",
            "Does not account for generator holdout choice, training seed, dataset sampling, "
            "unavailable source IDs, or broader generator populations.",
        ],
        "scientific_integrity": {
            "model_training_performed": False,
            "new_inference_based_tuning": False,
            "threshold_changed": False,
            "samples_removed_after_results": False,
            "architecture_selection_changed": False,
            "rq1_baseline_development_reopened": False,
        },
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    tidy = build_tidy_rows(results)
    tidy.to_csv(CSV_PATH, index=False)

    save_auc_ci_figure(results["model_confidence_intervals"])
    save_paired_diff_figure(
        primary_samples["unseen_test"],
        primary["unseen_test"]["roc_auc_difference"]["observed"],
    )

    report = write_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("STAGE 21A — RQ1 BOOTSTRAP UNCERTAINTY COMPLETE")
    print("")
    print(f"Bootstrap replicates: {N_BOOTSTRAP}")
    print(f"Seed: {RANDOM_SEED}")
    print("")
    print("KNOWN AUC 95% CI")
    for model in ("LogReg", "SmallCNNV1", "MobileNetV3-Small", "EfficientNet-B0"):
        s = known_summaries[model]["roc_auc"]
        print(f"{model}: {format_ci(s)}")
    print("")
    print("UNSEEN AUC 95% CI")
    for model in ("LogReg", "SmallCNNV1", "MobileNetV3-Small", "EfficientNet-B0"):
        s = unseen_summaries[model]["roc_auc"]
        print(f"{model}: {format_ci(s)}")
    print("")
    print("UNSEEN AUC DIFFERENCE")
    unseen_auc_diff = primary["unseen_test"]["roc_auc_difference"]
    print("EfficientNet - MobileNet:")
    print(f"Observed: {unseen_auc_diff['observed']:+.6f}")
    print(f"95% CI: {format_ci(unseen_auc_diff)}")
    print("")
    print("GENERALISATION GAP 95% CI")
    for model in ("LogReg", "SmallCNNV1", "MobileNetV3-Small", "EfficientNet-B0"):
        s = gap_summaries[model]["roc_auc_gap"]
        print(f"{model}: {format_ci(s)}")
    print("")
    print("Model training: NO")
    print("Threshold changes: NO")
    print("RQ1 model development reopened: NO")
    print("")
    print("STAGE 21A COMPLETE")
    print("STOP BEFORE RQ2 TRANSFORMATION EXPERIMENTS")
    print(f"\nWrote {JSON_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {AUC_CI_FIG_PATH}")
    print(f"Wrote {PAIRED_DIFF_FIG_PATH}")


if __name__ == "__main__":
    main()
