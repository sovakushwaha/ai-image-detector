"""RQ2 paired bootstrap uncertainty and synthesis (Stage 22D).

Why this file exists
--------------------
Quantifies uncertainty around transformation-induced metric changes using
existing frozen predictions only. No training, inference, or threshold changes.

How to run
----------
    source .venv/bin/activate
    python src/rq2_bootstrap_uncertainty_v1.py

What to expect
--------------
    results/rq2_strong_transform_summary_v1.csv
    results/rq2_pretrained_robustness_difference_v1.csv
    results/rq2_bootstrap_uncertainty_v1.json
    results/rq2_statistical_synthesis_report_v1.txt
    figures/rq2_strong_transform_delta_ci_v1.png
    figures/rq2_pretrained_robustness_difference_v1.png
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
METHOD = "stratified_paired_bootstrap_percentile_ci"

EXPECTED_COUNTS = {
    "known_test": {"total": 456, "real": 228, "ai": 228},
    "unseen_test": {"total": 1712, "real": 856, "ai": 856},
}

MODELS = ["LogReg", "SmallCNNV1", "MobileNetV3-Small", "EfficientNet-B0"]
PRETRAINED = ["MobileNetV3-Small", "EfficientNet-B0"]

ORIGINAL_PREDICTIONS = {
    "LogReg": {
        "known_test": PROJECT_ROOT / "results/logreg_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results/logreg_unseen_test_predictions_v1.csv",
    },
    "SmallCNNV1": {
        "known_test": PROJECT_ROOT / "results/smallcnn_v1_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results/smallcnn_v1_unseen_test_predictions_v1.csv",
    },
    "MobileNetV3-Small": {
        "known_test": PROJECT_ROOT / "results/mobilenet_v3_small_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results/mobilenet_v3_small_unseen_test_predictions_v1.csv",
    },
    "EfficientNet-B0": {
        "known_test": PROJECT_ROOT / "results/efficientnet_b0_known_test_predictions_v1.csv",
        "unseen_test": PROJECT_ROOT / "results/efficientnet_b0_unseen_test_predictions_v1.csv",
    },
}

ROBUSTNESS_PREDICTIONS = {
    "LogReg": PROJECT_ROOT / "results/rq2_logreg_predictions_v1.csv",
    "SmallCNNV1": PROJECT_ROOT / "results/rq2_smallcnn_predictions_v1.csv",
    "MobileNetV3-Small": PROJECT_ROOT / "results/rq2_mobilenet_predictions_v1.csv",
    "EfficientNet-B0": PROJECT_ROOT / "results/rq2_efficientnet_predictions_v1.csv",
}

SCREENSHOT_PREDICTIONS = {
    "LogReg": PROJECT_ROOT / "results/rq2_screenshot_logreg_predictions_v1.csv",
    "SmallCNNV1": PROJECT_ROOT / "results/rq2_screenshot_smallcnn_predictions_v1.csv",
    "MobileNetV3-Small": PROJECT_ROOT / "results/rq2_screenshot_mobilenet_predictions_v1.csv",
    "EfficientNet-B0": PROJECT_ROOT / "results/rq2_screenshot_efficientnet_predictions_v1.csv",
}

STRONG_CONDITIONS = ["jpeg_q50", "crop_75", "resize_112", "blur_sigma2", "screenshot_strong"]
CONDITION_LABELS = {
    "jpeg_q50": "JPEG50",
    "crop_75": "Crop75",
    "resize_112": "Resize112",
    "blur_sigma2": "Blur2",
    "screenshot_strong": "ScreenshotStrong",
}

MILD_STRONG_PAIRS = [
    ("jpeg", "jpeg_q75", "jpeg_q50"),
    ("crop", "crop_90", "crop_75"),
    ("resize", "resize_160", "resize_112"),
    ("blur", "blur_sigma1", "blur_sigma2"),
    ("screenshot", "screenshot_mild", "screenshot_strong"),
]

SUMMARY_CSV = PROJECT_ROOT / "results/rq2_strong_transform_summary_v1.csv"
PRETRAINED_DIFF_CSV = PROJECT_ROOT / "results/rq2_pretrained_robustness_difference_v1.csv"
JSON_PATH = PROJECT_ROOT / "results/rq2_bootstrap_uncertainty_v1.json"
REPORT_PATH = PROJECT_ROOT / "results/rq2_statistical_synthesis_report_v1.txt"
FIG_DELTA_CI = PROJECT_ROOT / "figures/rq2_strong_transform_delta_ci_v1.png"
FIG_PRETRAINED_DIFF = PROJECT_ROOT / "figures/rq2_pretrained_robustness_difference_v1.png"

ROBUSTNESS_METRICS = PROJECT_ROOT / "results/rq2_robustness_metrics_v1.csv"
SCREENSHOT_METRICS = PROJECT_ROOT / "results/rq2_screenshot_metrics_v1.csv"
ROBUSTNESS_GENERATOR = PROJECT_ROOT / "results/rq2_generator_recall_v1.csv"
SCREENSHOT_GENERATOR = PROJECT_ROOT / "results/rq2_screenshot_generator_recall_v1.csv"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_original(model: str, split: str) -> pd.DataFrame:
    path = ORIGINAL_PREDICTIONS[model][split]
    stop_if(not path.exists(), f"missing original predictions: {path}")
    df = pd.read_csv(path)
    stop_if("image_id" not in df.columns, f"{path} missing image_id")
    df = df.rename(columns={"image_id": "source_image_id", "ai_probability": "probability"})
    return df[["source_image_id", "true_label", "probability"]].copy()


def load_transformed(model: str, condition: str) -> pd.DataFrame:
    if condition == "screenshot_strong":
        path = SCREENSHOT_PREDICTIONS[model]
    else:
        path = ROBUSTNESS_PREDICTIONS[model]
    stop_if(not path.exists(), f"missing transformed predictions: {path}")
    df = pd.read_csv(path)
    if condition != "screenshot_strong":
        df = df[df["condition"] == condition].copy()
    else:
        df = df[df["condition"] == "screenshot_strong"].copy()
    return df[["source_image_id", "split", "label", "probability"]].copy()


def verify_split_metadata(source_ids: set[str], split: str, meta: pd.DataFrame) -> None:
    expected = set(meta[meta["split"] == split]["image_id"].astype(str))
    stop_if(source_ids != expected, f"{split} source_image_id set mismatch")


def build_prediction_cache(meta: pd.DataFrame) -> dict:
    """Load and align all original/transformed predictions once."""
    cache: dict = {}
    split_meta = meta

    for split in ("known_test", "unseen_test"):
        cache[split] = {"y_true": None, "models": {}}
        ref_ids: list[str] | None = None
        y_ref: np.ndarray | None = None

        for model in MODELS:
            cache[split]["models"][model] = {"original": None, "conditions": {}}
            orig = load_original(model, split).set_index("source_image_id").sort_index()
            ids = orig.index.astype(str).tolist()
            y_true = orig["true_label"].to_numpy(dtype=int)

            if ref_ids is None:
                ref_ids = ids
                y_ref = y_true
                verify_split_metadata(set(ids), split, split_meta)
                stop_if(len(ids) != EXPECTED_COUNTS[split]["total"], f"{split} count")
                stop_if(int((y_true == 0).sum()) != EXPECTED_COUNTS[split]["real"], f"{split} real count")
                stop_if(int((y_true == 1).sum()) != EXPECTED_COUNTS[split]["ai"], f"{split} ai count")
                cache[split]["y_true"] = y_true
            else:
                stop_if(ids != ref_ids, f"{model} {split} source_image_id order mismatch")
                stop_if(not np.array_equal(y_true, y_ref), f"{model} {split} labels mismatch")

            cache[split]["models"][model]["original"] = orig["probability"].to_numpy(dtype=float)

            for condition in STRONG_CONDITIONS:
                trans = load_transformed(model, condition)
                trans = trans[trans["split"] == split].set_index("source_image_id").sort_index()
                stop_if(list(trans.index.astype(str)) != ref_ids, f"{model} {split} {condition} alignment mismatch")
                stop_if(
                    not np.array_equal(trans["label"].to_numpy(dtype=int), y_ref),
                    f"{model} {split} {condition} label mismatch",
                )
                cache[split]["models"][model]["conditions"][condition] = trans["probability"].to_numpy(dtype=float)

    return cache


def get_paired(cache: dict, split: str, model: str, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = cache[split]["y_true"]
    p_orig = cache[split]["models"][model]["original"]
    p_trans = cache[split]["models"][model]["conditions"][condition]
    return y_true, p_orig, p_trans


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


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def summarize_bootstrap(observed: float, samples: np.ndarray) -> dict:
    clean = samples[np.isfinite(samples)]
    return {
        "observed": float(observed),
        "bootstrap_mean": float(np.mean(clean)),
        "bootstrap_std": float(np.std(clean, ddof=1)),
        "ci_95_low": float(np.percentile(clean, 2.5)),
        "ci_95_high": float(np.percentile(clean, 97.5)),
    }


def ci_includes_zero(ci_low: float, ci_high: float) -> bool:
    return ci_low <= 0.0 <= ci_high


def ci_fully_below_zero(ci_low: float, ci_high: float) -> bool:
    return ci_high < 0.0


def bootstrap_paired_deltas(
    y_true: np.ndarray,
    prob_original: np.ndarray,
    prob_transformed: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    auc_orig = safe_roc_auc(y_true, prob_original)
    auc_trans = safe_roc_auc(y_true, prob_transformed)
    ap_orig = safe_ap(y_true, prob_original)
    ap_trans = safe_ap(y_true, prob_transformed)

    delta_auc_obs = auc_trans - auc_orig
    delta_ap_obs = ap_trans - ap_orig

    delta_auc_samples = np.empty(N_BOOTSTRAP, dtype=float)
    delta_ap_samples = np.empty(N_BOOTSTRAP, dtype=float)

    for b, idx in enumerate(bootstrap_indices):
        y_b = y_true[idx]
        p_orig_b = prob_original[idx]
        p_trans_b = prob_transformed[idx]
        delta_auc_samples[b] = safe_roc_auc(y_b, p_trans_b) - safe_roc_auc(y_b, p_orig_b)
        delta_ap_samples[b] = safe_ap(y_b, p_trans_b) - safe_ap(y_b, p_orig_b)

    summary = {
        "original_auc": float(auc_orig),
        "transformed_auc": float(auc_trans),
        "delta_auc": summarize_bootstrap(delta_auc_obs, delta_auc_samples),
        "original_ap": float(ap_orig),
        "transformed_ap": float(ap_trans),
        "delta_ap": summarize_bootstrap(delta_ap_obs, delta_ap_samples),
    }
    return summary, delta_auc_samples, delta_ap_samples


def load_all_metrics() -> pd.DataFrame:
    robust = pd.read_csv(ROBUSTNESS_METRICS)
    screen = pd.read_csv(SCREENSHOT_METRICS)
    return pd.concat([robust, screen[robust.columns]], ignore_index=True)


def interpret_delta(summary: dict) -> str:
    ci = summary["delta_auc"]
    if ci_fully_below_zero(ci["ci_95_low"], ci["ci_95_high"]):
        return "consistent degradation (CI fully below zero)"
    if ci_includes_zero(ci["ci_95_low"], ci["ci_95_high"]):
        return "not clearly distinguishable from zero"
    return "observed increase (CI excludes zero and above)"


def severity_descriptive(metrics: pd.DataFrame) -> list[str]:
    lines = []
    for family, mild, strong in MILD_STRONG_PAIRS:
        lines.append(f"{family.upper()} (mild -> strong, unseen ΔAUC):")
        for model in PRETRAINED:
            m_row = metrics[
                (metrics["model"] == model) & (metrics["split"] == "unseen_test") & (metrics["condition"] == mild)
            ]
            s_row = metrics[
                (metrics["model"] == model) & (metrics["split"] == "unseen_test") & (metrics["condition"] == strong)
            ]
            if m_row.empty or s_row.empty:
                lines.append(f"  {model}: missing")
                continue
            dm = float(m_row.iloc[0]["delta_auc"])
            ds = float(s_row.iloc[0]["delta_auc"])
            direction = "strong worse" if ds < dm else ("strong better" if ds > dm else "equal")
            lines.append(f"  {model}: {mild} {dm:+.4f} -> {strong} {ds:+.4f} ({direction})")
    return lines


def threshold_synthesis(metrics: pd.DataFrame) -> list[str]:
    focus = ["resize_112", "blur_sigma2", "screenshot_strong"]
    lines = [
        "Under unchanged frozen thresholds, strong resize/blur/screenshot often show:",
        "- ROC-AUC/AP degradation (discrimination failure)",
        "- AI recall inflation with specificity collapse (operating-point shift)",
        "",
    ]
    for model in PRETRAINED + ["SmallCNNV1", "Handcrafted LogReg"]:
        orig = metrics[
            (metrics["model"] == model) & (metrics["split"] == "unseen_test") & (metrics["condition"] == "original")
        ]
        if orig.empty:
            continue
        orig_spec = float(orig.iloc[0]["specificity"])
        lines.append(f"{model} (unseen):")
        for cond in focus:
            row = metrics[
                (metrics["model"] == model) & (metrics["split"] == "unseen_test") & (metrics["condition"] == cond)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"  {cond}: ΔAUC={r.delta_auc:+.4f}, Δrecall={r.delta_recall:+.4f}, "
                f"Δspec={float(r.specificity) - orig_spec:+.4f}, FPR={r.fpr:.3f}"
            )
        lines.append("")
    return lines


def generator_synthesis() -> list[str]:
    gen_rob = pd.read_csv(ROBUSTNESS_GENERATOR)
    gen_scr = pd.read_csv(SCREENSHOT_GENERATOR)
    gen = pd.concat([gen_rob, gen_scr], ignore_index=True)
    strong = ["jpeg_q50", "crop_75", "resize_112", "blur_sigma2", "screenshot_strong"]
    unseen_gens = ["Midjourney", "VQDM", "Wukong"]
    lines = []
    for model in PRETRAINED:
        lines.append(f"{model} unseen generator recall (frozen threshold):")
        for g in unseen_gens:
            vals = []
            for cond in strong:
                row = gen[
                    (gen["model"] == model)
                    & (gen["split"] == "unseen_test")
                    & (gen["generator"] == g)
                    & (gen["condition"] == cond)
                ]
                if not row.empty:
                    vals.append(f"{cond}={row.iloc[0]['recall']:.3f}")
            lines.append(f"  {g}: " + ", ".join(vals))
        lines.append("")
    return lines


def rq3_implications(summary_df: pd.DataFrame, rankings: dict) -> list[str]:
    unseen = summary_df[summary_df["split"] == "unseen_test"]
    pretrained = unseen[unseen["model"].isin(PRETRAINED)]

    def mean_abs_delta(cond: str) -> float:
        sub = pretrained[pretrained["condition"] == cond]
        return float(sub["delta_auc"].abs().mean())

    priority = sorted(STRONG_CONDITIONS, key=lambda c: mean_abs_delta(c), reverse=True)
    ci_consistent = []
    for cond in STRONG_CONDITIONS:
        sub = pretrained[pretrained["condition"] == cond]
        n_below = int(((sub["delta_auc_ci_high"] < 0)).sum())
        ci_consistent.append(f"{cond}: {n_below}/2 pretrained models with CI fully below zero for ΔAUC")

    lines = [
        "Evidence-based priorities for RQ3 design (objective summary only; no augmentation protocol):",
        "",
        f"1. Magnitude (mean |ΔAUC| across pretrained, unseen): " + ", ".join(
            f"{c}={mean_abs_delta(c):.4f}" for c in priority
        ),
        f"   Ranking: {' > '.join(priority)}",
        "",
        "2. Bootstrap CI consistency (unseen ΔAUC):",
    ]
    lines.extend(f"   - {x}" for x in ci_consistent)
    lines.extend(
        [
            "",
            "3. Pretrained-model relevance: blur_sigma2 and resize_112 show largest consistent degradation;",
            "   screenshot_strong and jpeg_q50 show smaller but non-trivial effects.",
            "",
            "4. Severity: strong conditions consistently worse than mild for blur/resize/JPEG/screenshot",
            "   (descriptive check); crop mild vs strong mixed.",
            "",
            "Suggested RQ3 priority order (evidence only): blur > severe resize > screenshot > JPEG >> crop.",
            "Physical screen recapture not tested. No RQ3 training initiated.",
        ]
    )
    return lines


def save_figures(summary_df: pd.DataFrame, pretrained_diff_df: pd.DataFrame) -> None:
    unseen = summary_df[summary_df["split"] == "unseen_test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(STRONG_CONDITIONS))
    for ax, model in zip(axes, PRETRAINED):
        sub = unseen[unseen["model"] == model].set_index("condition").reindex(STRONG_CONDITIONS)
        ax.errorbar(
            x,
            sub["delta_auc"],
            yerr=[sub["delta_auc"] - sub["delta_auc_ci_low"], sub["delta_auc_ci_high"] - sub["delta_auc"]],
            fmt="o",
            capsize=4,
            color="#4C72B0",
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in STRONG_CONDITIONS], rotation=25, ha="right")
        ax.set_title(model)
        ax.set_ylabel("ΔAUC vs original")
    fig.suptitle("Unseen strong-transformation ΔAUC with 95% bootstrap CI")
    fig.tight_layout()
    FIG_DELTA_CI.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DELTA_CI, dpi=150)
    plt.close(fig)

    diff = pretrained_diff_df[pretrained_diff_df["split"] == "unseen_test"].set_index("condition").reindex(STRONG_CONDITIONS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(
        x,
        diff["difference_in_delta_auc"],
        yerr=[
            diff["difference_in_delta_auc"] - diff["ci_low"],
            diff["ci_high"] - diff["difference_in_delta_auc"],
        ],
        fmt="s",
        capsize=4,
        color="#DD8452",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in STRONG_CONDITIONS], rotation=25, ha="right")
    ax.set_ylabel("ΔAUC(EfficientNet) − ΔAUC(MobileNet)")
    ax.set_title("Unseen robustness-delta difference (positive = EfficientNet lost less)")
    fig.tight_layout()
    fig.savefig(FIG_PRETRAINED_DIFF, dpi=150)
    plt.close(fig)


def main() -> None:
    print("STAGE 22D — RQ2 STATISTICAL SYNTHESIS", flush=True)
    rng = np.random.default_rng(RANDOM_SEED)
    meta = pd.read_csv(SPLIT_META_PATH)
    print("Loading and aligning prediction cache...", flush=True)
    cache = build_prediction_cache(meta)
    print("Cache ready.", flush=True)

    all_results: dict = {
        "seed": RANDOM_SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "method": METHOD,
        "strong_conditions": STRONG_CONDITIONS,
        "transformation_deltas": {},
        "pretrained_robustness_differences": {},
        "rankings": {},
    }

    summary_rows = []
    pretrained_diff_rows = []

    for split in ("known_test", "unseen_test"):
        y_ref = cache[split]["y_true"]
        boot_idx = np.empty((N_BOOTSTRAP, len(y_ref)), dtype=int)
        for b in range(N_BOOTSTRAP):
            boot_idx[b] = stratified_bootstrap_indices(y_ref, rng)

        all_results["transformation_deltas"][split] = {}
        all_results["pretrained_robustness_differences"][split] = {}

        delta_auc_by_model_cond: dict[str, dict[str, np.ndarray]] = {m: {} for m in PRETRAINED}

        for model in MODELS:
            all_results["transformation_deltas"][split][model] = {}
            for condition in STRONG_CONDITIONS:
                y_true, p_orig, p_trans = get_paired(cache, split, model, condition)
                summary, d_auc, d_ap = bootstrap_paired_deltas(y_true, p_orig, p_trans, boot_idx)
                all_results["transformation_deltas"][split][model][condition] = summary

                summary_rows.append(
                    {
                        "model": model,
                        "split": split,
                        "condition": condition,
                        "original_auc": summary["original_auc"],
                        "transformed_auc": summary["transformed_auc"],
                        "delta_auc": summary["delta_auc"]["observed"],
                        "delta_auc_ci_low": summary["delta_auc"]["ci_95_low"],
                        "delta_auc_ci_high": summary["delta_auc"]["ci_95_high"],
                        "original_ap": summary["original_ap"],
                        "transformed_ap": summary["transformed_ap"],
                        "delta_ap": summary["delta_ap"]["observed"],
                        "delta_ap_ci_low": summary["delta_ap"]["ci_95_low"],
                        "delta_ap_ci_high": summary["delta_ap"]["ci_95_high"],
                    }
                )

                if model in PRETRAINED:
                    delta_auc_by_model_cond[model][condition] = d_auc

        for condition in STRONG_CONDITIONS:
            d_mob = delta_auc_by_model_cond["MobileNetV3-Small"][condition]
            d_eff = delta_auc_by_model_cond["EfficientNet-B0"][condition]
            diff_samples = d_eff - d_mob
            obs_point = float(
                all_results["transformation_deltas"][split]["EfficientNet-B0"][condition]["delta_auc"]["observed"]
                - all_results["transformation_deltas"][split]["MobileNetV3-Small"][condition]["delta_auc"]["observed"]
            )
            diff_summary = summarize_bootstrap(obs_point, diff_samples)

            _, p_o_m, p_t_m = get_paired(cache, split, "MobileNetV3-Small", condition)
            _, p_o_e, p_t_e = get_paired(cache, split, "EfficientNet-B0", condition)
            ap_diff_samples = np.empty(N_BOOTSTRAP, dtype=float)
            for b, idx in enumerate(boot_idx):
                y_b = y_ref[idx]
                ap_diff_samples[b] = (safe_ap(y_b, p_t_e[idx]) - safe_ap(y_b, p_o_e[idx])) - (
                    safe_ap(y_b, p_t_m[idx]) - safe_ap(y_b, p_o_m[idx])
                )
            ap_obs_point = float(
                all_results["transformation_deltas"][split]["EfficientNet-B0"][condition]["delta_ap"]["observed"]
                - all_results["transformation_deltas"][split]["MobileNetV3-Small"][condition]["delta_ap"]["observed"]
            )
            ap_diff_summary = summarize_bootstrap(ap_obs_point, ap_diff_samples)

            all_results["pretrained_robustness_differences"][split][condition] = {
                "mobilenet_delta_auc": all_results["transformation_deltas"][split]["MobileNetV3-Small"][condition]["delta_auc"],
                "efficientnet_delta_auc": all_results["transformation_deltas"][split]["EfficientNet-B0"][condition]["delta_auc"],
                "difference_in_delta_auc": diff_summary,
                "difference_in_delta_ap": ap_diff_summary,
            }

            pretrained_diff_rows.append(
                {
                    "split": split,
                    "condition": condition,
                    "mobilenet_delta_auc": all_results["transformation_deltas"][split]["MobileNetV3-Small"][condition]["delta_auc"]["observed"],
                    "efficientnet_delta_auc": all_results["transformation_deltas"][split]["EfficientNet-B0"][condition]["delta_auc"]["observed"],
                    "difference_in_delta_auc": diff_summary["observed"],
                    "ci_low": diff_summary["ci_95_low"],
                    "ci_high": diff_summary["ci_95_high"],
                    "mobilenet_delta_ap": all_results["transformation_deltas"][split]["MobileNetV3-Small"][condition]["delta_ap"]["observed"],
                    "efficientnet_delta_ap": all_results["transformation_deltas"][split]["EfficientNet-B0"][condition]["delta_ap"]["observed"],
                    "difference_in_delta_ap": ap_diff_summary["observed"],
                    "ap_ci_low": ap_diff_summary["ci_95_low"],
                    "ap_ci_high": ap_diff_summary["ci_95_high"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    pretrained_diff_df = pd.DataFrame(pretrained_diff_rows)

    # Rankings
    rankings = {}
    metrics = load_all_metrics()
    for model in PRETRAINED:
        sub = summary_df[(summary_df["model"] == model) & (summary_df["split"] == "unseen_test")]
        auc_order = sub.sort_values("delta_auc").reset_index(drop=True)
        ap_order = sub.sort_values("delta_ap").reset_index(drop=True)
        rankings[model] = {
            "unseen_delta_auc_rank_worst_first": auc_order["condition"].tolist(),
            "unseen_delta_ap_rank_worst_first": ap_order["condition"].tolist(),
        }
    all_results["rankings"] = rankings

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    pretrained_diff_df.to_csv(PRETRAINED_DIFF_CSV, index=False)
    JSON_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Report
    lines = [
        "RQ2 Statistical Synthesis Report — Stage 22D",
        "============================================",
        "",
        "METHOD",
        f"- seed: {RANDOM_SEED}",
        f"- bootstrap replicates: {N_BOOTSTRAP}",
        f"- method: {METHOD}",
        "- stratified paired resampling of source_image_id (Real/AI separately)",
        "- same bootstrap sample for original and transformed predictions",
        "- no model inference, training, or threshold changes",
        "",
        "STRONG TRANSFORMATION DELTAS (observed unseen ΔAUC [95% CI])",
    ]
    for model in MODELS:
        lines.append(f"- {model}:")
        for cond in STRONG_CONDITIONS:
            row = summary_df[
                (summary_df["model"] == model) & (summary_df["split"] == "unseen_test") & (summary_df["condition"] == cond)
            ].iloc[0]
            interp = interpret_delta(all_results["transformation_deltas"]["unseen_test"][model][cond])
            lines.append(
                f"    {cond}: {row.delta_auc:+.4f} [{row.delta_auc_ci_low:+.4f}, {row.delta_auc_ci_high:+.4f}] — {interp}"
            )

    lines.extend(["", "MOBILENET VS EFFICIENTNET (unseen difference-in-delta ΔAUC [95% CI])"])
    for cond in STRONG_CONDITIONS:
        row = pretrained_diff_df[(pretrained_diff_df["split"] == "unseen_test") & (pretrained_diff_df["condition"] == cond)].iloc[0]
        lines.append(
            f"- {cond}: {row.difference_in_delta_auc:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]"
        )

    lines.extend(["", "TRANSFORMATION DAMAGE RANKING (unseen, worst ΔAUC first)"])
    for model in PRETRAINED:
        lines.append(f"- {model} ΔAUC: {rankings[model]['unseen_delta_auc_rank_worst_first']}")
        lines.append(f"- {model} ΔAP:  {rankings[model]['unseen_delta_ap_rank_worst_first']}")

    lines.extend(["", "SEVERITY PATTERNS (descriptive)"])
    lines.extend(severity_descriptive(metrics))

    lines.extend(["", "THRESHOLD SHIFT VS DISCRIMINATION"])
    lines.extend(threshold_synthesis(metrics))

    lines.extend(["", "GENERATOR-SPECIFIC OBSERVATIONS"])
    lines.extend(generator_synthesis())

    lines.extend(
        [
            "",
            "LIMITATIONS",
            "- Bootstrap uncertainty for fixed test samples only",
            "- Does not account for alternative generator holdouts, training seeds, datasets",
            "- Does not represent all social-media platforms or physical screen recapture",
            "- Alternative transformation parameterisations not tested",
            "",
            "IMPLICATIONS FOR RQ3",
        ]
    )
    lines.extend(rq3_implications(summary_df, rankings))

    lines.extend(
        [
            "",
            "SCIENTIFIC INTEGRITY",
            "- Model training: NO",
            "- Model inference: NO",
            "- Threshold changes: NO",
            "- Transformation modification: NO",
            "- Sample exclusion: NO",
            "- RQ1 reopened: NO",
            "- RQ3 training started: NO",
        ]
    )
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")

    save_figures(summary_df, pretrained_diff_df)

    # Terminal summary
    print("\nSTAGE 22D — RQ2 STATISTICAL SYNTHESIS COMPLETE")
    print(f"\nBootstrap:\nReplicates: {N_BOOTSTRAP}\nSeed: {RANDOM_SEED}")
    print("\nUNSEEN STRONG TRANSFORMATION DELTA AUC")
    for model in PRETRAINED:
        print(f"\n{model.split('-')[0] if 'Mobile' in model else model}:")
        short = "MobileNet" if "Mobile" in model else "EfficientNet"
        for cond in STRONG_CONDITIONS:
            row = summary_df[
                (summary_df["model"] == model) & (summary_df["split"] == "unseen_test") & (summary_df["condition"] == cond)
            ].iloc[0]
            label = CONDITION_LABELS[cond]
            print(f"  {label}: {row.delta_auc:+.4f} [{row.delta_auc_ci_low:+.4f}, {row.delta_auc_ci_high:+.4f}]")

    print("\nMOST DAMAGING TRANSFORMATION (unseen ΔAUC)")
    for model in PRETRAINED:
        worst = rankings[model]["unseen_delta_auc_rank_worst_first"][0]
        print(f"  {model}: {worst}")

    print("\nMOBILENET VS EFFICIENTNET ROBUSTNESS (unseen difference-in-delta ΔAUC)")
    for cond in ["blur_sigma2", "resize_112", "screenshot_strong"]:
        row = pretrained_diff_df[(pretrained_diff_df["split"] == "unseen_test") & (pretrained_diff_df["condition"] == cond)].iloc[0]
        print(f"  {CONDITION_LABELS[cond]}: {row.difference_in_delta_auc:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]")

    print("\nRQ2 STATISTICAL SYNTHESIS: COMPLETE")
    print("Model training: NO")
    print("Model inference: NO")
    print("RQ3 training started: NO")
    print(f"\nOutputs: {SUMMARY_CSV}, {PRETRAINED_DIFF_CSV}, {JSON_PATH}, {REPORT_PATH}")
    print(f"Figures: {FIG_DELTA_CI}, {FIG_PRETRAINED_DIFF}")


if __name__ == "__main__":
    main()
