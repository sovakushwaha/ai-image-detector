"""RQ4 Stage 24D paired bootstrap statistical analysis.

Why this file exists
--------------------
Analysis-only. Uses frozen F0/F1/F2 prediction CSVs. No new inference.
Primary comparison: F2 vs F0. Secondary: F1 vs F0 and F2 vs F1.

How to run
----------
    source .venv/bin/activate
    PYTHONPATH=src python src/rq4_bootstrap_uncertainty_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
METHOD = "paired_class_stratified_source_bootstrap_percentile_ci"

EXPECTED_KNOWN = 456
EXPECTED_UNSEEN = 1712
EXPECTED_ROWS = 10840
EXPECTED_METRICS_ROWS = 30
AUC_REPRO_TOL = 1e-4
AP_REPRO_TOL = 1e-4

CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
TRANSFORMED = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
STRONG_SCORE_CONDS = ["original", "jpeg_q50", "resize_112", "blur_sigma2"]
SPLITS = ["known_test", "unseen_test"]
REGIMES = ["F0", "F1", "F2"]

F0_PRED = PROJECT_ROOT / "results/rq3_A2_test_predictions_v1.csv"
F1_PRED = PROJECT_ROOT / "results/rq4_F1_test_predictions_v1.csv"
F2_PRED = PROJECT_ROOT / "results/rq4_F2_test_predictions_v1.csv"
METRICS_CSV = PROJECT_ROOT / "results/rq4_test_metrics_v1.csv"

F0_FROZEN = PROJECT_ROOT / "results/rq3_A2_frozen_config_v1.json"
F1_FROZEN = PROJECT_ROOT / "results/rq4_F1_frozen_config_v1.json"
F2_FROZEN = PROJECT_ROOT / "results/rq4_F2_frozen_config_v1.json"

JSON_OUT = PROJECT_ROOT / "results/rq4_bootstrap_uncertainty_v1.json"
CSV_OUT = PROJECT_ROOT / "results/rq4_bootstrap_uncertainty_v1.csv"
PRIMARY_TABLE = PROJECT_ROOT / "paper/tables/rq4_primary_fusion_bootstrap.csv"
REPORT_OUT = PROJECT_ROOT / "results/rq4_statistical_analysis_report_v1.txt"

FIG_F2_CI = PROJECT_ROOT / "figures/rq4_fusion_vs_rgb_auc_difference_ci_v1.png"
FIG_ROBUST = PROJECT_ROOT / "figures/rq4_strong_robust_bootstrap_v1.png"
FIG_DID = PROJECT_ROOT / "figures/rq4_difference_in_delta_v1.png"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def summarize(observed: float, samples: np.ndarray) -> dict:
    clean = samples[np.isfinite(samples)]
    lo = float(np.percentile(clean, 2.5))
    hi = float(np.percentile(clean, 97.5))
    return {
        "observed": float(observed),
        "bootstrap_mean": float(np.mean(clean)),
        "bootstrap_std": float(np.std(clean, ddof=1)),
        "ci_95_low": lo,
        "ci_95_high": hi,
        "includes_zero": bool(lo <= 0.0 <= hi),
    }


def interpret_diff(summary: dict) -> str:
    if summary["ci_95_high"] < 0:
        return "consistent negative difference under the current fixed-sample bootstrap"
    if summary["ci_95_low"] > 0:
        return "consistent positive difference under the current fixed-sample bootstrap"
    return "not clearly distinguishable from zero under this fixed-sample bootstrap"


def load_pred(path: Path, regime: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    stop_if(len(df) != EXPECTED_ROWS, f"{regime} rows {len(df)}")
    df = df.copy()
    df["regime"] = regime
    return df[["regime", "source_image_id", "split", "generator", "label", "condition", "probability"]]


def verify_and_build_cache() -> tuple[dict, pd.DataFrame, dict]:
    print("PART A — STAGE 24C VERIFICATION")
    preds = {
        "F0": load_pred(F0_PRED, "F0"),
        "F1": load_pred(F1_PRED, "F1"),
        "F2": load_pred(F2_PRED, "F2"),
    }
    metrics = pd.read_csv(METRICS_CSV)
    stop_if(len(metrics) != EXPECTED_METRICS_ROWS, f"metrics rows {len(metrics)}")

    thresholds = {
        "F0": float(json.loads(F0_FROZEN.read_text())["threshold"]),
        "F1": float(json.loads(F1_FROZEN.read_text())["threshold"]),
        "F2": float(json.loads(F2_FROZEN.read_text())["threshold"]),
    }
    f2_cfg = json.loads(F2_FROZEN.read_text())
    stop_if(not f2_cfg.get("primary_rq4_intervention", False), "F2 not marked primary")

    cache: dict = {}
    for split in SPLITS:
        expected_n = EXPECTED_KNOWN if split == "known_test" else EXPECTED_UNSEEN
        ref = (
            preds["F0"][(preds["F0"]["split"] == split) & (preds["F0"]["condition"] == "original")]
            .sort_values("source_image_id")
            .reset_index(drop=True)
        )
        stop_if(len(ref) != expected_n, f"{split} F0 original count")
        ids = ref["source_image_id"].astype(str).tolist()
        y_true = ref["label"].to_numpy(dtype=int)
        gens = ref["generator"].astype(str).tolist()
        stop_if(int((y_true == 0).sum()) != expected_n // 2, f"{split} real count")
        stop_if(int((y_true == 1).sum()) != expected_n // 2, f"{split} AI count")

        cache[split] = {"ids": ids, "y_true": y_true, "generators": gens, "regimes": {}}
        for regime in REGIMES:
            cache[split]["regimes"][regime] = {}
            for condition in CONDITIONS:
                sub = (
                    preds[regime][(preds[regime]["split"] == split) & (preds[regime]["condition"] == condition)]
                    .sort_values("source_image_id")
                    .reset_index(drop=True)
                )
                stop_if(len(sub) != expected_n, f"{regime} {split} {condition} count")
                stop_if(sub["source_image_id"].astype(str).tolist() != ids, f"{regime} id mismatch")
                stop_if(not np.array_equal(sub["label"].to_numpy(dtype=int), y_true), f"{regime} label")
                probs = sub["probability"].to_numpy(dtype=float)
                cache[split]["regimes"][regime][condition] = probs

                mrow = metrics[
                    (metrics["regime"] == regime) & (metrics["split"] == split) & (metrics["condition"] == condition)
                ].iloc[0]
                auc = safe_auc(y_true, probs)
                ap = safe_ap(y_true, probs)
                stop_if(abs(auc - float(mrow["roc_auc"])) > AUC_REPRO_TOL, f"{regime} AUC repro")
                stop_if(abs(ap - float(mrow["average_precision"])) > AP_REPRO_TOL, f"{regime} AP repro")
                stop_if(abs(float(mrow["threshold"]) - thresholds[regime]) > 1e-12, f"{regime} threshold")

    print("Stage 24C verification: PASSED")
    return cache, metrics, thresholds


def stratified_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    real_idx = np.flatnonzero(y_true == 0)
    ai_idx = np.flatnonzero(y_true == 1)
    return np.concatenate(
        [
            rng.choice(real_idx, size=len(real_idx), replace=True),
            rng.choice(ai_idx, size=len(ai_idx), replace=True),
        ]
    )


def precompute_bootstrap_indices(cache: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    out = {}
    for split in SPLITS:
        y = cache[split]["y_true"]
        idx = np.empty((N_BOOTSTRAP, len(y)), dtype=np.int64)
        for b in range(N_BOOTSTRAP):
            idx[b] = stratified_indices(y, rng)
        out[split] = idx
    return out


def condition_metric_for_indices(y: np.ndarray, probs: np.ndarray, indices: np.ndarray, fn) -> np.ndarray:
    out = np.empty(len(indices), dtype=float)
    for b, idx in enumerate(indices):
        out[b] = fn(y[idx], probs[idx])
    return out


def run_bootstrap(cache: dict, boot_idx: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    tidy: list[dict] = []
    payload: dict = {
        "seed": RANDOM_SEED,
        "replicate_count": N_BOOTSTRAP,
        "bootstrap_method": METHOD,
        "primary_intervention": "F2",
        "primary_changed_after_test": False,
        "known_test": {},
        "unseen_test": {},
        "generalisation_gap": {},
        "limitations": [
            "Sequential follow-up to RQ1–RQ3 on the same Tiny-GenImage pilot benchmark.",
            "One predefined FFT magnitude representation; phase discarded.",
            "Estimates sampling uncertainty for fixed pilot test samples only.",
            "Does not establish independent external/new-generator confirmation.",
        ],
    }

    boot_auc: dict = {}
    boot_ap: dict = {}
    obs_auc: dict = {}
    obs_ap: dict = {}
    for split in SPLITS:
        y = cache[split]["y_true"]
        idx = boot_idx[split]
        boot_auc[split] = {}
        boot_ap[split] = {}
        obs_auc[split] = {}
        obs_ap[split] = {}
        for regime in REGIMES:
            boot_auc[split][regime] = {}
            boot_ap[split][regime] = {}
            obs_auc[split][regime] = {}
            obs_ap[split][regime] = {}
            for condition in CONDITIONS:
                p = cache[split]["regimes"][regime][condition]
                obs_auc[split][regime][condition] = safe_auc(y, p)
                obs_ap[split][regime][condition] = safe_ap(y, p)
                boot_auc[split][regime][condition] = condition_metric_for_indices(y, p, idx, safe_auc)
                boot_ap[split][regime][condition] = condition_metric_for_indices(y, p, idx, safe_ap)

    def add_row(comparison, split, condition, metric, ref_obs, cand_obs, diff_summary, analysis_type):
        tidy.append(
            {
                "comparison": comparison,
                "split": split,
                "condition": condition,
                "metric": metric,
                "reference_value": ref_obs,
                "candidate_value": cand_obs,
                "observed_difference": diff_summary["observed"],
                "bootstrap_mean": diff_summary["bootstrap_mean"],
                "bootstrap_std": diff_summary["bootstrap_std"],
                "ci_low": diff_summary["ci_95_low"],
                "ci_high": diff_summary["ci_95_high"],
                "includes_zero": diff_summary["includes_zero"],
                "analysis_type": analysis_type,
                "interpretation": interpret_diff(diff_summary),
            }
        )

    for split in SPLITS:
        payload[split]["F2_vs_F0"] = {}
        payload[split]["difference_in_delta_F2_vs_F0"] = {}
        payload[split]["strong_robust"] = {}
        payload[split]["secondary"] = {}

        for condition in CONDITIONS:
            f0_auc = obs_auc[split]["F0"][condition]
            f2_auc = obs_auc[split]["F2"][condition]
            f0_ap = obs_ap[split]["F0"][condition]
            f2_ap = obs_ap[split]["F2"][condition]
            d_auc = summarize(f2_auc - f0_auc, boot_auc[split]["F2"][condition] - boot_auc[split]["F0"][condition])
            d_ap = summarize(f2_ap - f0_ap, boot_ap[split]["F2"][condition] - boot_ap[split]["F0"][condition])
            payload[split]["F2_vs_F0"][condition] = {
                "f0_auc": f0_auc,
                "f2_auc": f2_auc,
                "auc_diff": d_auc,
                "f0_ap": f0_ap,
                "f2_ap": f2_ap,
                "ap_diff": d_ap,
            }
            add_row("F2_vs_F0", split, condition, "roc_auc", f0_auc, f2_auc, d_auc, "primary_absolute")
            add_row("F2_vs_F0", split, condition, "average_precision", f0_ap, f2_ap, d_ap, "primary_absolute")

        for condition in TRANSFORMED:
            d0 = obs_auc[split]["F0"][condition] - obs_auc[split]["F0"]["original"]
            d2 = obs_auc[split]["F2"][condition] - obs_auc[split]["F2"]["original"]
            d0b = boot_auc[split]["F0"][condition] - boot_auc[split]["F0"]["original"]
            d2b = boot_auc[split]["F2"][condition] - boot_auc[split]["F2"]["original"]
            did_auc = summarize(d2 - d0, d2b - d0b)
            d0_ap = obs_ap[split]["F0"][condition] - obs_ap[split]["F0"]["original"]
            d2_ap = obs_ap[split]["F2"][condition] - obs_ap[split]["F2"]["original"]
            d0_apb = boot_ap[split]["F0"][condition] - boot_ap[split]["F0"]["original"]
            d2_apb = boot_ap[split]["F2"][condition] - boot_ap[split]["F2"]["original"]
            did_ap = summarize(d2_ap - d0_ap, d2_apb - d0_apb)
            payload[split]["difference_in_delta_F2_vs_F0"][condition] = {
                "delta_f0_auc": d0,
                "delta_f2_auc": d2,
                "difference_in_delta_auc": did_auc,
                "delta_f0_ap": d0_ap,
                "delta_f2_ap": d2_ap,
                "difference_in_delta_ap": did_ap,
            }
            add_row("F2_vs_F0", split, condition, "difference_in_delta_auc", d0, d2, did_auc, "primary_rel_robustness")
            add_row("F2_vs_F0", split, condition, "difference_in_delta_ap", d0_ap, d2_ap, did_ap, "primary_rel_robustness")

        for regime in REGIMES:
            obs_s_auc = float(np.mean([obs_auc[split][regime][c] for c in STRONG_SCORE_CONDS]))
            obs_s_ap = float(np.mean([obs_ap[split][regime][c] for c in STRONG_SCORE_CONDS]))
            boot_s_auc = np.mean(np.vstack([boot_auc[split][regime][c] for c in STRONG_SCORE_CONDS]), axis=0)
            boot_s_ap = np.mean(np.vstack([boot_ap[split][regime][c] for c in STRONG_SCORE_CONDS]), axis=0)
            payload[split]["strong_robust"][regime] = {
                "strong_robust_test_auc_observed": obs_s_auc,
                "strong_robust_test_ap_observed": obs_s_ap,
                "boot_auc": boot_s_auc.tolist(),
                "boot_ap": boot_s_ap.tolist(),
            }

        # Primary strong robust F2-F0
        f0s = payload[split]["strong_robust"]["F0"]
        f2s = payload[split]["strong_robust"]["F2"]
        d_auc = summarize(
            f2s["strong_robust_test_auc_observed"] - f0s["strong_robust_test_auc_observed"],
            np.array(f2s["boot_auc"]) - np.array(f0s["boot_auc"]),
        )
        d_ap = summarize(
            f2s["strong_robust_test_ap_observed"] - f0s["strong_robust_test_ap_observed"],
            np.array(f2s["boot_ap"]) - np.array(f0s["boot_ap"]),
        )
        payload[split]["F2_vs_F0"]["StrongRobustTest"] = {"auc_diff": d_auc, "ap_diff": d_ap}
        add_row(
            "F2_vs_F0",
            split,
            "StrongRobustTest",
            "roc_auc",
            f0s["strong_robust_test_auc_observed"],
            f2s["strong_robust_test_auc_observed"],
            d_auc,
            "primary_strong_robust",
        )
        add_row(
            "F2_vs_F0",
            split,
            "StrongRobustTest",
            "average_precision",
            f0s["strong_robust_test_ap_observed"],
            f2s["strong_robust_test_ap_observed"],
            d_ap,
            "primary_strong_robust",
        )

        # Secondary F1-F0 and F2-F1 on unseen (and also known for completeness on strong)
        for comparison, cand, ref in [("F1_vs_F0", "F1", "F0"), ("F2_vs_F1", "F2", "F1")]:
            payload[split]["secondary"][comparison] = {}
            for condition in ["original", "resize_112", "blur_sigma2"]:
                ref_auc = obs_auc[split][ref][condition]
                cand_auc = obs_auc[split][cand][condition]
                ref_ap = obs_ap[split][ref][condition]
                cand_ap = obs_ap[split][cand][condition]
                da = summarize(cand_auc - ref_auc, boot_auc[split][cand][condition] - boot_auc[split][ref][condition])
                dp = summarize(cand_ap - ref_ap, boot_ap[split][cand][condition] - boot_ap[split][ref][condition])
                payload[split]["secondary"][comparison][condition] = {"auc_diff": da, "ap_diff": dp}
                if split == "unseen_test":
                    add_row(comparison, split, condition, "roc_auc", ref_auc, cand_auc, da, "secondary")
                    add_row(comparison, split, condition, "average_precision", ref_ap, cand_ap, dp, "secondary")

            rs = payload[split]["strong_robust"][ref]
            cs = payload[split]["strong_robust"][cand]
            da = summarize(
                cs["strong_robust_test_auc_observed"] - rs["strong_robust_test_auc_observed"],
                np.array(cs["boot_auc"]) - np.array(rs["boot_auc"]),
            )
            dp = summarize(
                cs["strong_robust_test_ap_observed"] - rs["strong_robust_test_ap_observed"],
                np.array(cs["boot_ap"]) - np.array(rs["boot_ap"]),
            )
            payload[split]["secondary"][comparison]["StrongRobustTest"] = {"auc_diff": da, "ap_diff": dp}
            if split == "unseen_test":
                add_row(
                    comparison,
                    split,
                    "StrongRobustTest",
                    "roc_auc",
                    rs["strong_robust_test_auc_observed"],
                    cs["strong_robust_test_auc_observed"],
                    da,
                    "secondary_strong_robust",
                )
                add_row(
                    comparison,
                    split,
                    "StrongRobustTest",
                    "average_precision",
                    rs["strong_robust_test_ap_observed"],
                    cs["strong_robust_test_ap_observed"],
                    dp,
                    "secondary_strong_robust",
                )

        # Drop large boot arrays from JSON payload for size
        for regime in REGIMES:
            payload[split]["strong_robust"][regime].pop("boot_auc", None)
            payload[split]["strong_robust"][regime].pop("boot_ap", None)
            # keep summaries only — re-attach observed diffs already stored

    # Generalisation gap original AUC/AP
    for regime in REGIMES:
        known_auc = obs_auc["known_test"][regime]["original"]
        unseen_auc = obs_auc["unseen_test"][regime]["original"]
        known_ap = obs_ap["known_test"][regime]["original"]
        unseen_ap = obs_ap["unseen_test"][regime]["original"]
        # Paired across splits is not the same source set — report separate bootstrap of each gap as
        # independent stratified bootstrap of each split's original scores, then difference of means.
        # Per protocol: bootstrap uncertainty for ORIGINAL AUC gap = unseen - known.
        # We bootstrap each split independently (different images), then form gap distribution.
        gap_auc_boot = boot_auc["unseen_test"][regime]["original"] - boot_auc["known_test"][regime]["original"]
        gap_ap_boot = boot_ap["unseen_test"][regime]["original"] - boot_ap["known_test"][regime]["original"]
        # Note: known and unseen boot indices are independent (different RNG streams after sequential generation).
        g_auc = summarize(unseen_auc - known_auc, gap_auc_boot)
        g_ap = summarize(unseen_ap - known_ap, gap_ap_boot)
        payload["generalisation_gap"][regime] = {"auc_gap": g_auc, "ap_gap": g_ap}
        add_row(f"{regime}_gap", "unseen_minus_known", "original", "roc_auc", known_auc, unseen_auc, g_auc, "generalisation_gap")
        add_row(f"{regime}_gap", "unseen_minus_known", "original", "average_precision", known_ap, unseen_ap, g_ap, "generalisation_gap")

    # Clean trade-off already covered by F2-F0 original; store explicit pointer
    payload["clean_performance_tradeoff_unseen"] = payload["unseen_test"]["F2_vs_F0"]["original"]

    return tidy, payload


def write_primary_table(payload: dict) -> None:
    rows = []
    for condition in CONDITIONS + ["StrongRobustTest"]:
        block = payload["unseen_test"]["F2_vs_F0"][condition]
        if condition == "StrongRobustTest":
            f0_auc = payload["unseen_test"]["strong_robust"]["F0"]["strong_robust_test_auc_observed"]
            f2_auc = payload["unseen_test"]["strong_robust"]["F2"]["strong_robust_test_auc_observed"]
            f0_ap = payload["unseen_test"]["strong_robust"]["F0"]["strong_robust_test_ap_observed"]
            f2_ap = payload["unseen_test"]["strong_robust"]["F2"]["strong_robust_test_ap_observed"]
            auc_diff = block["auc_diff"]
            ap_diff = block["ap_diff"]
        else:
            f0_auc = block["f0_auc"]
            f2_auc = block["f2_auc"]
            f0_ap = block["f0_ap"]
            f2_ap = block["f2_ap"]
            auc_diff = block["auc_diff"]
            ap_diff = block["ap_diff"]
        rows.append(
            {
                "condition": condition,
                "F0_AUC": f0_auc,
                "F2_AUC": f2_auc,
                "AUC_Difference": auc_diff["observed"],
                "AUC_95CI": f"[{auc_diff['ci_95_low']:.4f}, {auc_diff['ci_95_high']:.4f}]",
                "F0_AP": f0_ap,
                "F2_AP": f2_ap,
                "AP_Difference": ap_diff["observed"],
                "AP_95CI": f"[{ap_diff['ci_95_low']:.4f}, {ap_diff['ci_95_high']:.4f}]",
            }
        )
    PRIMARY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PRIMARY_TABLE, index=False)


def plot_figures(payload: dict) -> None:
    # F2-F0 AUC CI
    conds = CONDITIONS
    labels = ["Original", "JPEG50", "Resize112", "Blur2", "Screenshot"]
    diffs = []
    los = []
    his = []
    for c in conds:
        d = payload["unseen_test"]["F2_vs_F0"][c]["auc_diff"]
        diffs.append(d["observed"])
        los.append(d["observed"] - d["ci_95_low"])
        his.append(d["ci_95_high"] - d["observed"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(conds))
    ax.errorbar(x, diffs, yerr=[los, his], fmt="o", capsize=4, color="C0")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC difference (F2 − F0)")
    ax.set_title("RQ4 unseen F2−F0 AUC difference with 95% bootstrap CI")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_F2_CI, dpi=150)
    plt.close(fig)

    # Strong robust F1-F0 and F2-F0
    fig, ax = plt.subplots(figsize=(6, 4))
    comps = ["F1_vs_F0", "F2_vs_F0"]
    vals = []
    yerr = [[], []]
    for comp in comps:
        if comp == "F2_vs_F0":
            d = payload["unseen_test"]["F2_vs_F0"]["StrongRobustTest"]["auc_diff"]
        else:
            d = payload["unseen_test"]["secondary"]["F1_vs_F0"]["StrongRobustTest"]["auc_diff"]
        vals.append(d["observed"])
        yerr[0].append(d["observed"] - d["ci_95_low"])
        yerr[1].append(d["ci_95_high"] - d["observed"])
    ax.bar(np.arange(2), vals, yerr=yerr, capsize=4, color=["C1", "C0"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["F1−F0", "F2−F0"])
    ax.set_ylabel("StrongRobustTestAUC difference")
    ax.set_title("RQ4 unseen strong-robust AUC diffs vs F0")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_ROBUST, dpi=150)
    plt.close(fig)

    # Difference-in-delta
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["JPEG50", "Resize112", "Blur2", "Screenshot"]
    diffs = []
    los = []
    his = []
    for c in TRANSFORMED:
        d = payload["unseen_test"]["difference_in_delta_F2_vs_F0"][c]["difference_in_delta_auc"]
        diffs.append(d["observed"])
        los.append(d["observed"] - d["ci_95_low"])
        his.append(d["ci_95_high"] - d["observed"])
    x = np.arange(len(TRANSFORMED))
    ax.errorbar(x, diffs, yerr=[los, his], fmt="o", capsize=4, color="C2")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("(ΔAUC_F2 − ΔAUC_F0)")
    ax.set_title("RQ4 unseen difference-in-delta AUC (relative robustness)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DID, dpi=150)
    plt.close(fig)


def write_report(payload: dict, tidy: pd.DataFrame) -> None:
    u = payload["unseen_test"]
    lines = [
        "RQ4 Stage 24D — Paired Bootstrap Statistical Analysis",
        "=" * 60,
        "",
        f"Bootstrap replicates: {N_BOOTSTRAP}",
        f"Seed: {RANDOM_SEED}",
        f"Method: {METHOD}",
        "Primary intervention: F2 (unchanged after test)",
        "New inference: NO",
        "",
        "PRIMARY F2 − F0 UNSEEN AUC",
    ]
    for cond in CONDITIONS + ["StrongRobustTest"]:
        d = u["F2_vs_F0"][cond]["auc_diff"]
        lines.append(
            f"  {cond}: diff={d['observed']:+.4f} 95% CI [{d['ci_95_low']:+.4f}, {d['ci_95_high']:+.4f}] — {interpret_diff(d)}"
        )
    lines.append("")
    lines.append("StrongRobustTestAP F2−F0:")
    d = u["F2_vs_F0"]["StrongRobustTest"]["ap_diff"]
    lines.append(
        f"  diff={d['observed']:+.4f} 95% CI [{d['ci_95_low']:+.4f}, {d['ci_95_high']:+.4f}] — {interpret_diff(d)}"
    )
    lines.append("")
    lines.append("FREQUENCY-ONLY SECONDARY (unseen StrongRobustTestAUC)")
    for comp in ["F1_vs_F0", "F2_vs_F1"]:
        d = u["secondary"][comp]["StrongRobustTest"]["auc_diff"]
        lines.append(
            f"  {comp}: {d['observed']:+.4f} [{d['ci_95_low']:+.4f}, {d['ci_95_high']:+.4f}] — {interpret_diff(d)}"
        )
    lines.append("")
    lines.append("Clean trade-off (original unseen AUC F2−F0):")
    d = u["F2_vs_F0"]["original"]["auc_diff"]
    lines.append(f"  {d['observed']:+.4f} [{d['ci_95_low']:+.4f}, {d['ci_95_high']:+.4f}] — {interpret_diff(d)}")
    lines.append("")
    lines.append("Integrity: no new inference; F2 remains primary; no threshold/checkpoint changes.")
    lines.append(f"Tidy CSV rows: {len(tidy)}")
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("=== Stage 24D — RQ4 paired bootstrap ===")
    cache, metrics, thresholds = verify_and_build_cache()
    print("Precomputing bootstrap indices...")
    boot_idx = precompute_bootstrap_indices(cache)
    print("Running bootstrap analyses (this may take several minutes)...")
    tidy_rows, payload = run_bootstrap(cache, boot_idx)

    # Convert numpy types in payload for JSON
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    payload_out = convert(payload)
    with open(JSON_OUT, "w") as f:
        json.dump(payload_out, f, indent=2)
        f.write("\n")

    tidy = pd.DataFrame(tidy_rows)
    tidy.to_csv(CSV_OUT, index=False)
    write_primary_table(payload)
    plot_figures(payload)
    write_report(payload, tidy)
    print(f"Saved {JSON_OUT}")
    print(f"Saved {CSV_OUT}")
    print("Stage 24D COMPLETE.")


if __name__ == "__main__":
    main()
