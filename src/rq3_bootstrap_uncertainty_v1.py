"""RQ3 Stage 23D verification + Stage 23E paired bootstrap analysis.

Why this file exists
--------------------
Analysis-only. Verifies Stage-23D prediction integrity, then estimates paired
bootstrap uncertainty for A2 vs A0 (primary) and A1/A3 ablations. No training,
inference, threshold changes, or primary-candidate reselection.

How to run
----------
    source .venv/bin/activate
    python src/rq3_bootstrap_uncertainty_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEED = 42
N_BOOTSTRAP = 5000
METHOD = "paired_class_stratified_source_bootstrap_percentile_ci"

EXPECTED_KNOWN = 456
EXPECTED_UNSEEN = 1712
EXPECTED_SOURCES = 2168
EXPECTED_ROWS = 10840
EXPECTED_METRICS_ROWS = 40
AUC_REPRO_TOL = 1e-4
AP_REPRO_TOL = 1e-4

CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
TRANSFORMED = ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
STRONG_SCORE_CONDS = ["original", "jpeg_q50", "resize_112", "blur_sigma2"]
SPLITS = ["known_test", "unseen_test"]
REGIMES = ["A0", "A1", "A2", "A3"]
UNSEEN_GENERATORS = ["Midjourney", "VQDM", "Wukong"]

A0_KNOWN = PROJECT_ROOT / "results/mobilenet_v3_small_known_test_predictions_v1.csv"
A0_UNSEEN = PROJECT_ROOT / "results/mobilenet_v3_small_unseen_test_predictions_v1.csv"
A0_RQ2 = PROJECT_ROOT / "results/rq2_mobilenet_predictions_v1.csv"
A0_SCREEN = PROJECT_ROOT / "results/rq2_screenshot_mobilenet_predictions_v1.csv"
A1_PRED = PROJECT_ROOT / "results/rq3_A1_test_predictions_v1.csv"
A2_PRED = PROJECT_ROOT / "results/rq3_A2_test_predictions_v1.csv"
A3_PRED = PROJECT_ROOT / "results/rq3_A3_test_predictions_v1.csv"
METRICS_CSV = PROJECT_ROOT / "results/rq3_test_metrics_v1.csv"
GEN_RECALL_CSV = PROJECT_ROOT / "results/rq3_generator_recall_v1.csv"

A0_FROZEN = PROJECT_ROOT / "results/mobilenet_v3_small_frozen_config_v1.json"
A1_FROZEN = PROJECT_ROOT / "results/rq3_A1_frozen_config_v1.json"
A2_FROZEN = PROJECT_ROOT / "results/rq3_A2_frozen_config_v1.json"
A3_FROZEN = PROJECT_ROOT / "results/rq3_A3_frozen_config_v1.json"

JSON_OUT = PROJECT_ROOT / "results/rq3_bootstrap_uncertainty_v1.json"
CSV_OUT = PROJECT_ROOT / "results/rq3_bootstrap_uncertainty_v1.csv"
PRIMARY_TABLE = PROJECT_ROOT / "paper/tables/rq3_primary_a2_vs_a0_bootstrap.csv"
ABLATION_TABLE = PROJECT_ROOT / "paper/tables/rq3_ablation_bootstrap_summary.csv"
REPORT_OUT = PROJECT_ROOT / "results/rq3_statistical_analysis_report_v1.txt"

FIG_A2_CI = PROJECT_ROOT / "figures/rq3_a2_vs_a0_auc_difference_ci_v1.png"
FIG_ROBUST = PROJECT_ROOT / "figures/rq3_robust_score_bootstrap_v1.png"
FIG_DID = PROJECT_ROOT / "figures/rq3_a2_difference_in_delta_v1.png"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_a0() -> pd.DataFrame:
    known = pd.read_csv(A0_KNOWN).rename(
        columns={"image_id": "source_image_id", "true_label": "label", "ai_probability": "probability"}
    )
    unseen = pd.read_csv(A0_UNSEEN).rename(
        columns={"image_id": "source_image_id", "true_label": "label", "ai_probability": "probability"}
    )
    original = pd.concat([known, unseen], ignore_index=True)
    original["condition"] = "original"
    original["regime"] = "A0"
    original = original[["regime", "source_image_id", "split", "generator", "label", "condition", "probability"]]

    rq2 = pd.read_csv(A0_RQ2)
    rq2 = rq2[rq2["condition"].isin(["jpeg_q50", "resize_112", "blur_sigma2"])].copy()
    rq2["regime"] = "A0"
    rq2 = rq2[["regime", "source_image_id", "split", "generator", "label", "condition", "probability"]]

    screen = pd.read_csv(A0_SCREEN)
    screen = screen[screen["condition"] == "screenshot_strong"].copy()
    screen["regime"] = "A0"
    screen = screen[["regime", "source_image_id", "split", "generator", "label", "condition", "probability"]]
    out = pd.concat([original, rq2, screen], ignore_index=True)
    stop_if(len(out) != EXPECTED_ROWS, f"A0 rows {len(out)}")
    return out


def load_regime_pred(path: Path, regime: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    stop_if(len(df) != EXPECTED_ROWS, f"{regime} rows {len(df)}")
    stop_if(set(df["condition"]) != set(CONDITIONS), f"{regime} conditions mismatch")
    stop_if((df["split"] == "known_test").sum() != EXPECTED_KNOWN * 5, f"{regime} known rows")
    stop_if((df["split"] == "unseen_test").sum() != EXPECTED_UNSEEN * 5, f"{regime} unseen rows")
    return df[["regime", "source_image_id", "split", "generator", "label", "condition", "probability"]].copy()


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


def verify_and_build_cache() -> tuple[dict, pd.DataFrame, dict]:
    print("PART A — STAGE 23D VERIFICATION")
    preds = {
        "A0": load_a0(),
        "A1": load_regime_pred(A1_PRED, "A1"),
        "A2": load_regime_pred(A2_PRED, "A2"),
        "A3": load_regime_pred(A3_PRED, "A3"),
    }
    metrics = pd.read_csv(METRICS_CSV)
    stop_if(len(metrics) != EXPECTED_METRICS_ROWS, f"metrics rows {len(metrics)}")

    thresholds = {
        "A0": float(json.loads(A0_FROZEN.read_text())["threshold"]),
        "A1": float(json.loads(A1_FROZEN.read_text())["threshold"]),
        "A2": float(json.loads(A2_FROZEN.read_text())["threshold"]),
        "A3": float(json.loads(A3_FROZEN.read_text())["threshold"]),
    }
    a2_cfg = json.loads(A2_FROZEN.read_text())
    stop_if(not a2_cfg.get("primary_rq3_candidate", False), "A2 not marked primary")

    cache: dict = {}
    for split in SPLITS:
        expected_n = EXPECTED_KNOWN if split == "known_test" else EXPECTED_UNSEEN
        ref = (
            preds["A0"][(preds["A0"]["split"] == split) & (preds["A0"]["condition"] == "original")]
            .sort_values("source_image_id")
            .reset_index(drop=True)
        )
        stop_if(len(ref) != expected_n, f"{split} A0 original count")
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
                stop_if(sub["source_image_id"].astype(str).tolist() != ids, f"{regime} {split} {condition} id mismatch")
                stop_if(not np.array_equal(sub["label"].to_numpy(dtype=int), y_true), f"{regime} {split} {condition} label")
                stop_if(sub["generator"].astype(str).tolist() != gens, f"{regime} {split} {condition} generator")
                probs = sub["probability"].to_numpy(dtype=float)
                cache[split]["regimes"][regime][condition] = probs

                # reproduce Stage-23D metrics
                mrow = metrics[
                    (metrics["regime"] == regime) & (metrics["split"] == split) & (metrics["condition"] == condition)
                ].iloc[0]
                auc = safe_auc(y_true, probs)
                ap = safe_ap(y_true, probs)
                stop_if(abs(auc - float(mrow["roc_auc"])) > AUC_REPRO_TOL, f"{regime} {split} {condition} AUC repro")
                stop_if(abs(ap - float(mrow["average_precision"])) > AP_REPRO_TOL, f"{regime} {split} {condition} AP repro")
                stop_if(abs(float(mrow["threshold"]) - thresholds[regime]) > 1e-12, f"{regime} threshold mismatch")

    print("Stage 23D verification: PASSED")
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


def condition_aucs_for_indices(y: np.ndarray, probs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.empty(len(indices), dtype=float)
    for b, idx in enumerate(indices):
        out[b] = safe_auc(y[idx], probs[idx])
    return out


def condition_aps_for_indices(y: np.ndarray, probs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.empty(len(indices), dtype=float)
    for b, idx in enumerate(indices):
        out[b] = safe_ap(y[idx], probs[idx])
    return out


def run_bootstrap(cache: dict, boot_idx: dict[str, np.ndarray]) -> tuple[list[dict], dict]:
    tidy: list[dict] = []
    payload: dict = {
        "seed": RANDOM_SEED,
        "replicate_count": N_BOOTSTRAP,
        "bootstrap_method": METHOD,
        "primary_candidate": "A2",
        "primary_candidate_changed_after_test": False,
        "known_test": {},
        "unseen_test": {},
        "limitations": [
            "Sequential follow-up to RQ2 on the same pilot benchmark.",
            "Estimates sampling uncertainty for fixed pilot test samples only.",
            "Does not account for training seeds, alternative holdouts, new generators, or physical recapture.",
        ],
    }

    # Precompute per (split, regime, condition) bootstrap AUC/AP arrays
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
                boot_auc[split][regime][condition] = condition_aucs_for_indices(y, p, idx)
                boot_ap[split][regime][condition] = condition_aps_for_indices(y, p, idx)

    def add_row(
        comparison: str,
        split: str,
        condition: str,
        metric: str,
        ref_obs: float,
        cand_obs: float,
        diff_summary: dict,
        analysis_type: str,
    ) -> None:
        tidy.append(
            {
                "comparison": comparison,
                "split": split,
                "condition": condition,
                "metric": metric,
                "observed_reference": ref_obs,
                "observed_candidate": cand_obs,
                "observed_difference": diff_summary["observed"],
                "bootstrap_mean_difference": diff_summary["bootstrap_mean"],
                "bootstrap_std": diff_summary["bootstrap_std"],
                "ci_low": diff_summary["ci_95_low"],
                "ci_high": diff_summary["ci_95_high"],
                "includes_zero": diff_summary["includes_zero"],
                "analysis_type": analysis_type,
                "interpretation": interpret_diff(diff_summary),
            }
        )

    for split in SPLITS:
        split_key = split
        payload[split_key]["condition_a2_vs_a0"] = {}
        payload[split_key]["difference_in_delta_a2_vs_a0"] = {}
        payload[split_key]["strong_robust"] = {}
        payload[split_key]["ablations"] = {}

        # Absolute A2 - A0 by condition
        for condition in CONDITIONS:
            a0_auc = obs_auc[split]["A0"][condition]
            a2_auc = obs_auc[split]["A2"][condition]
            a0_ap = obs_ap[split]["A0"][condition]
            a2_ap = obs_ap[split]["A2"][condition]
            d_auc = summarize(a2_auc - a0_auc, boot_auc[split]["A2"][condition] - boot_auc[split]["A0"][condition])
            d_ap = summarize(a2_ap - a0_ap, boot_ap[split]["A2"][condition] - boot_ap[split]["A0"][condition])
            payload[split_key]["condition_a2_vs_a0"][condition] = {
                "a0_auc": a0_auc,
                "a2_auc": a2_auc,
                "auc_diff": d_auc,
                "a0_ap": a0_ap,
                "a2_ap": a2_ap,
                "ap_diff": d_ap,
            }
            add_row("A2_vs_A0", split, condition, "roc_auc", a0_auc, a2_auc, d_auc, "primary_absolute")
            add_row("A2_vs_A0", split, condition, "average_precision", a0_ap, a2_ap, d_ap, "primary_absolute")

        # Difference-in-delta
        for condition in TRANSFORMED:
            d0_obs = obs_auc[split]["A0"][condition] - obs_auc[split]["A0"]["original"]
            d2_obs = obs_auc[split]["A2"][condition] - obs_auc[split]["A2"]["original"]
            d0_boot = boot_auc[split]["A0"][condition] - boot_auc[split]["A0"]["original"]
            d2_boot = boot_auc[split]["A2"][condition] - boot_auc[split]["A2"]["original"]
            did_auc = summarize(d2_obs - d0_obs, d2_boot - d0_boot)

            d0_ap_obs = obs_ap[split]["A0"][condition] - obs_ap[split]["A0"]["original"]
            d2_ap_obs = obs_ap[split]["A2"][condition] - obs_ap[split]["A2"]["original"]
            d0_ap_boot = boot_ap[split]["A0"][condition] - boot_ap[split]["A0"]["original"]
            d2_ap_boot = boot_ap[split]["A2"][condition] - boot_ap[split]["A2"]["original"]
            did_ap = summarize(d2_ap_obs - d0_ap_obs, d2_ap_boot - d0_ap_boot)

            payload[split_key]["difference_in_delta_a2_vs_a0"][condition] = {
                "delta_a0_auc": d0_obs,
                "delta_a2_auc": d2_obs,
                "difference_in_delta_auc": did_auc,
                "delta_a0_ap": d0_ap_obs,
                "delta_a2_ap": d2_ap_obs,
                "difference_in_delta_ap": did_ap,
            }
            add_row("A2_vs_A0", split, condition, "difference_in_delta_auc", d0_obs, d2_obs, did_auc, "primary_rel_robustness")
            add_row("A2_vs_A0", split, condition, "difference_in_delta_ap", d0_ap_obs, d2_ap_obs, did_ap, "primary_rel_robustness")

        # StrongRobust scores
        for regime in REGIMES:
            obs_s_auc = float(np.mean([obs_auc[split][regime][c] for c in STRONG_SCORE_CONDS]))
            obs_s_ap = float(np.mean([obs_ap[split][regime][c] for c in STRONG_SCORE_CONDS]))
            boot_s_auc = np.mean(np.vstack([boot_auc[split][regime][c] for c in STRONG_SCORE_CONDS]), axis=0)
            boot_s_ap = np.mean(np.vstack([boot_ap[split][regime][c] for c in STRONG_SCORE_CONDS]), axis=0)
            payload[split_key]["strong_robust"][regime] = {
                "strong_robust_test_auc_observed": obs_s_auc,
                "strong_robust_test_ap_observed": obs_s_ap,
                "boot_auc": boot_s_auc,
                "boot_ap": boot_s_ap,
            }

        for cand in ["A1", "A2", "A3"]:
            a0s = payload[split_key]["strong_robust"]["A0"]
            cs = payload[split_key]["strong_robust"][cand]
            d_auc = summarize(
                cs["strong_robust_test_auc_observed"] - a0s["strong_robust_test_auc_observed"],
                cs["boot_auc"] - a0s["boot_auc"],
            )
            d_ap = summarize(
                cs["strong_robust_test_ap_observed"] - a0s["strong_robust_test_ap_observed"],
                cs["boot_ap"] - a0s["boot_ap"],
            )
            analysis = "primary_strong_robust" if cand == "A2" else "secondary_ablation"
            payload[split_key]["ablations"][f"{cand}_vs_A0_strong_robust"] = {
                "auc_diff": d_auc,
                "ap_diff": d_ap,
            }
            add_row(
                f"{cand}_vs_A0",
                split,
                "StrongRobustTest",
                "roc_auc",
                a0s["strong_robust_test_auc_observed"],
                cs["strong_robust_test_auc_observed"],
                d_auc,
                analysis,
            )
            add_row(
                f"{cand}_vs_A0",
                split,
                "StrongRobustTest",
                "average_precision",
                a0s["strong_robust_test_ap_observed"],
                cs["strong_robust_test_ap_observed"],
                d_ap,
                analysis,
            )

        # Secondary absolute A1/A3 vs A0 conditions (unseen emphasized later)
        for cand in ["A1", "A3"]:
            payload[split_key]["ablations"][f"{cand}_vs_A0_conditions"] = {}
            for condition in CONDITIONS:
                d_auc = summarize(
                    obs_auc[split][cand][condition] - obs_auc[split]["A0"][condition],
                    boot_auc[split][cand][condition] - boot_auc[split]["A0"][condition],
                )
                d_ap = summarize(
                    obs_ap[split][cand][condition] - obs_ap[split]["A0"][condition],
                    boot_ap[split][cand][condition] - boot_ap[split]["A0"][condition],
                )
                payload[split_key]["ablations"][f"{cand}_vs_A0_conditions"][condition] = {
                    "auc_diff": d_auc,
                    "ap_diff": d_ap,
                }
                add_row(
                    f"{cand}_vs_A0",
                    split,
                    condition,
                    "roc_auc",
                    obs_auc[split]["A0"][condition],
                    obs_auc[split][cand][condition],
                    d_auc,
                    "secondary_ablation",
                )
                add_row(
                    f"{cand}_vs_A0",
                    split,
                    condition,
                    "average_precision",
                    obs_ap[split]["A0"][condition],
                    obs_ap[split][cand][condition],
                    d_ap,
                    "secondary_ablation",
                )

        # Secondary A3 - A2 StrongRobust
        a3s = payload[split_key]["strong_robust"]["A3"]
        a2s = payload[split_key]["strong_robust"]["A2"]
        d_a3_a2 = summarize(
            a3s["strong_robust_test_auc_observed"] - a2s["strong_robust_test_auc_observed"],
            a3s["boot_auc"] - a2s["boot_auc"],
        )
        payload[split_key]["ablations"]["A3_vs_A2_strong_robust_auc"] = d_a3_a2
        add_row(
            "A3_vs_A2",
            split,
            "StrongRobustTest",
            "roc_auc",
            a2s["strong_robust_test_auc_observed"],
            a3s["strong_robust_test_auc_observed"],
            d_a3_a2,
            "secondary_ablation",
        )

        # Secondary balanced accuracy A2 vs A0 for strong conditions
        y = cache[split]["y_true"]
        thr0 = float(json.loads(A0_FROZEN.read_text())["threshold"])
        thr2 = float(json.loads(A2_FROZEN.read_text())["threshold"])
        payload[split_key]["secondary_balanced_accuracy_a2_vs_a0"] = {}
        for condition in ["original", "resize_112", "blur_sigma2", "screenshot_strong"]:
            p0 = cache[split]["regimes"]["A0"][condition]
            p2 = cache[split]["regimes"]["A2"][condition]
            ba0 = float(balanced_accuracy_score(y, (p0 >= thr0).astype(int)))
            ba2 = float(balanced_accuracy_score(y, (p2 >= thr2).astype(int)))
            samples = np.empty(N_BOOTSTRAP, dtype=float)
            for b, idx in enumerate(boot_idx[split]):
                yb = y[idx]
                samples[b] = balanced_accuracy_score(yb, (p2[idx] >= thr2).astype(int)) - balanced_accuracy_score(
                    yb, (p0[idx] >= thr0).astype(int)
                )
            d_ba = summarize(ba2 - ba0, samples)
            payload[split_key]["secondary_balanced_accuracy_a2_vs_a0"][condition] = {
                "a0": ba0,
                "a2": ba2,
                "diff": d_ba,
            }
            add_row("A2_vs_A0", split, condition, "balanced_accuracy", ba0, ba2, d_ba, "secondary_operating_point")

        # Strip large bootstrap arrays from JSON payload
        for regime in REGIMES:
            payload[split_key]["strong_robust"][regime] = {
                "strong_robust_test_auc_observed": payload[split_key]["strong_robust"][regime][
                    "strong_robust_test_auc_observed"
                ],
                "strong_robust_test_ap_observed": payload[split_key]["strong_robust"][regime][
                    "strong_robust_test_ap_observed"
                ],
            }

    return tidy, payload


def write_tables(tidy: pd.DataFrame, payload: dict) -> None:
    PRIMARY_TABLE.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    unseen = payload["unseen_test"]["condition_a2_vs_a0"]
    for condition, label in [
        ("original", "Original"),
        ("jpeg_q50", "JPEG50"),
        ("resize_112", "Resize112"),
        ("blur_sigma2", "Blur2"),
        ("screenshot_strong", "ScreenshotStrong"),
    ]:
        c = unseen[condition]
        rows.append(
            {
                "condition": label,
                "a0_auc": c["a0_auc"],
                "a2_auc": c["a2_auc"],
                "a2_minus_a0_auc": c["auc_diff"]["observed"],
                "auc_ci_low": c["auc_diff"]["ci_95_low"],
                "auc_ci_high": c["auc_diff"]["ci_95_high"],
                "a0_ap": c["a0_ap"],
                "a2_ap": c["a2_ap"],
                "a2_minus_a0_ap": c["ap_diff"]["observed"],
                "ap_ci_low": c["ap_diff"]["ci_95_low"],
                "ap_ci_high": c["ap_diff"]["ci_95_high"],
            }
        )
    sr_a0 = payload["unseen_test"]["strong_robust"]["A0"]
    sr_a2 = payload["unseen_test"]["strong_robust"]["A2"]
    sr_diff = payload["unseen_test"]["ablations"]["A2_vs_A0_strong_robust"]
    rows.append(
        {
            "condition": "StrongRobustTest",
            "a0_auc": sr_a0["strong_robust_test_auc_observed"],
            "a2_auc": sr_a2["strong_robust_test_auc_observed"],
            "a2_minus_a0_auc": sr_diff["auc_diff"]["observed"],
            "auc_ci_low": sr_diff["auc_diff"]["ci_95_low"],
            "auc_ci_high": sr_diff["auc_diff"]["ci_95_high"],
            "a0_ap": sr_a0["strong_robust_test_ap_observed"],
            "a2_ap": sr_a2["strong_robust_test_ap_observed"],
            "a2_minus_a0_ap": sr_diff["ap_diff"]["observed"],
            "ap_ci_low": sr_diff["ap_diff"]["ci_95_low"],
            "ap_ci_high": sr_diff["ap_diff"]["ci_95_high"],
        }
    )
    pd.DataFrame(rows).to_csv(PRIMARY_TABLE, index=False)

    abl_rows = []
    for cand in ["A1", "A2", "A3"]:
        if cand == "A2":
            conds = {
                condition: {"auc_diff": payload["unseen_test"]["condition_a2_vs_a0"][condition]["auc_diff"]}
                for condition in CONDITIONS
            }
        else:
            conds = payload["unseen_test"]["ablations"][f"{cand}_vs_A0_conditions"]
        sr = payload["unseen_test"]["ablations"][f"{cand}_vs_A0_strong_robust"]["auc_diff"]
        abl_rows.append(
            {
                "comparison": f"{cand}_vs_A0",
                "original_auc_diff": conds["original"]["auc_diff"]["observed"],
                "jpeg50_auc_diff": conds["jpeg_q50"]["auc_diff"]["observed"],
                "resize112_auc_diff": conds["resize_112"]["auc_diff"]["observed"],
                "blur2_auc_diff": conds["blur_sigma2"]["auc_diff"]["observed"],
                "screenshot_strong_auc_diff": conds["screenshot_strong"]["auc_diff"]["observed"],
                "strong_robust_auc_diff": sr["observed"],
                "strong_robust_auc_ci_low": sr["ci_95_low"],
                "strong_robust_auc_ci_high": sr["ci_95_high"],
                "primary_candidate": cand == "A2",
            }
        )
    pd.DataFrame(abl_rows).to_csv(ABLATION_TABLE, index=False)
    tidy.to_csv(CSV_OUT, index=False)


def plot_figures(payload: dict) -> None:
    FIGURES = PROJECT_ROOT / "figures"
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = ["Original", "JPEG50", "Resize112", "Blur2", "ScreenshotStrong"]
    keys = ["original", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
    diffs = [payload["unseen_test"]["condition_a2_vs_a0"][k]["auc_diff"] for k in keys]
    means = [d["observed"] for d in diffs]
    yerr = np.array([[m - d["ci_95_low"], d["ci_95_high"] - m] for m, d in zip(means, diffs)]).T

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(labels, means, yerr=yerr, fmt="o", capsize=4, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("A2 − A0 unseen ROC-AUC")
    ax.set_title("RQ3 primary: A2 vs A0 unseen AUC difference (95% paired CI)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_A2_CI, dpi=150)
    plt.close(fig)

    # Robust score diffs vs A0
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cands = ["A1", "A2", "A3"]
    colors = ["#55A868", "#C44E52", "#8172B2"]
    for i, cand in enumerate(cands):
        d = payload["unseen_test"]["ablations"][f"{cand}_vs_A0_strong_robust"]["auc_diff"]
        ax.errorbar(
            [i],
            [d["observed"]],
            yerr=[[d["observed"] - d["ci_95_low"]], [d["ci_95_high"] - d["observed"]]],
            fmt="o",
            capsize=5,
            color=colors[i],
            label=("A2 primary" if cand == "A2" else cand),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["A1 Blur", "A2 Resize+JPEG\n(primary)", "A3 Combined"])
    ax.set_ylabel("StrongRobustTestAUC − A0")
    ax.set_title("RQ3 unseen StrongRobustTestAUC vs A0 (95% CI)")
    fig.tight_layout()
    fig.savefig(FIG_ROBUST, dpi=150)
    plt.close(fig)

    # Difference-in-delta
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tlabels = ["JPEG50", "Resize112", "Blur2", "ScreenshotStrong"]
    tkeys = TRANSFORMED
    dids = [payload["unseen_test"]["difference_in_delta_a2_vs_a0"][k]["difference_in_delta_auc"] for k in tkeys]
    means = [d["observed"] for d in dids]
    yerr = np.array([[m - d["ci_95_low"], d["ci_95_high"] - m] for m, d in zip(means, dids)]).T
    ax.errorbar(tlabels, means, yerr=yerr, fmt="o", capsize=4, color="#2ca02c")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("(ΔAUC_A2 − ΔAUC_A0)")
    ax.set_title("A2 vs A0 difference-in-delta (relative robustness; unseen)")
    fig.tight_layout()
    fig.savefig(FIG_DID, dpi=150)
    plt.close(fig)


def write_report(payload: dict, metrics: pd.DataFrame, thresholds: dict) -> None:
    u = payload["unseen_test"]
    k = payload["known_test"]

    def fmt_diff(d: dict) -> str:
        return f"{d['observed']:+.6f}  [{d['ci_95_low']:+.6f}, {d['ci_95_high']:+.6f}]  ({interpret_diff(d)})"

    gen = pd.read_csv(GEN_RECALL_CSV)
    lines = [
        "RQ3 Statistical Analysis Report — Stage 23E",
        "===========================================",
        "",
        "1. STAGE 23D VERIFICATION",
        "PASSED. Prediction counts, source alignment, labels, generators, thresholds,",
        "and Stage-23D ROC-AUC/AP reproduction all verified. No new inference.",
        "",
        "2. BOOTSTRAP METHOD",
        f"Replicates: {N_BOOTSTRAP}",
        f"Seed: {RANDOM_SEED}",
        f"Method: {METHOD}",
        "Same resampled source IDs applied across A0/A1/A2/A3 and all five conditions.",
        "",
        "3. PRIMARY A2 VS A0 (UNSEEN AUC)",
    ]
    for condition in CONDITIONS:
        lines.append(f"{condition}: {fmt_diff(u['condition_a2_vs_a0'][condition]['auc_diff'])}")
    lines.extend(
        [
            "",
            "4. CLEAN PERFORMANCE TRADE-OFF",
            f"Unseen original AUC A2−A0: {fmt_diff(u['condition_a2_vs_a0']['original']['auc_diff'])}",
            f"Unseen original AP A2−A0: {fmt_diff(u['condition_a2_vs_a0']['original']['ap_diff'])}",
            f"Known original AUC A2−A0: {fmt_diff(k['condition_a2_vs_a0']['original']['auc_diff'])}",
            "",
            "5. JPEG TARGETED RESULT",
            f"Unseen JPEG50 AUC: {fmt_diff(u['condition_a2_vs_a0']['jpeg_q50']['auc_diff'])}",
            f"Unseen JPEG50 difference-in-delta AUC: {fmt_diff(u['difference_in_delta_a2_vs_a0']['jpeg_q50']['difference_in_delta_auc'])}",
            "",
            "6. RESIZE TARGETED RESULT",
            f"Unseen Resize112 AUC: {fmt_diff(u['condition_a2_vs_a0']['resize_112']['auc_diff'])}",
            f"Unseen Resize112 difference-in-delta AUC: {fmt_diff(u['difference_in_delta_a2_vs_a0']['resize_112']['difference_in_delta_auc'])}",
            "",
            "7. BLUR CROSS-TRANSFORMATION RESULT",
            "A2 did not receive blur augmentation.",
            f"Unseen Blur2 AUC: {fmt_diff(u['condition_a2_vs_a0']['blur_sigma2']['auc_diff'])}",
            f"Unseen Blur2 difference-in-delta AUC: {fmt_diff(u['difference_in_delta_a2_vs_a0']['blur_sigma2']['difference_in_delta_auc'])}",
            "",
            "8. SCREENSHOT CROSS-TRANSFORMATION RESULT",
            "ScreenshotStrong was not explicitly included as a training augmentation.",
            f"Unseen ScreenshotStrong AUC: {fmt_diff(u['condition_a2_vs_a0']['screenshot_strong']['auc_diff'])}",
            f"Unseen ScreenshotStrong difference-in-delta AUC: {fmt_diff(u['difference_in_delta_a2_vs_a0']['screenshot_strong']['difference_in_delta_auc'])}",
            "",
            "9. STRONG ROBUST TEST SCORE",
        ]
    )
    for regime in REGIMES:
        lines.append(
            f"{regime}: AUC={u['strong_robust'][regime]['strong_robust_test_auc_observed']:.6f} "
            f"AP={u['strong_robust'][regime]['strong_robust_test_ap_observed']:.6f}"
        )
    lines.append(f"A2−A0 StrongRobustTestAUC: {fmt_diff(u['ablations']['A2_vs_A0_strong_robust']['auc_diff'])}")
    lines.append(f"A2−A0 StrongRobustTestAP: {fmt_diff(u['ablations']['A2_vs_A0_strong_robust']['ap_diff'])}")
    lines.extend(
        [
            "",
            "10. DIFFERENCE-IN-DELTA ANALYSIS",
            "Positive values: A2 lost less relative to its own original than A0.",
        ]
    )
    for condition in TRANSFORMED:
        lines.append(
            f"{condition}: {fmt_diff(u['difference_in_delta_a2_vs_a0'][condition]['difference_in_delta_auc'])}"
        )
    lines.extend(
        [
            "",
            "11. A1 BLUR ABLATION (SECONDARY)",
            f"A1−A0 Blur2 AUC: {fmt_diff(u['ablations']['A1_vs_A0_conditions']['blur_sigma2']['auc_diff'])}",
            f"A1−A0 StrongRobustTestAUC: {fmt_diff(u['ablations']['A1_vs_A0_strong_robust']['auc_diff'])}",
            "A1 remains a predefined ablation; A2 remains the validation-selected primary.",
            "",
            "12. A3 COMBINED ABLATION (SECONDARY)",
            f"A3−A0 StrongRobustTestAUC: {fmt_diff(u['ablations']['A3_vs_A0_strong_robust']['auc_diff'])}",
            f"A3−A2 StrongRobustTestAUC: {fmt_diff(u['ablations']['A3_vs_A2_strong_robust_auc'])}",
            "",
            "13. THRESHOLD BEHAVIOUR",
            "Primary conclusions use ROC-AUC/AP. Frozen-threshold metrics are descriptive.",
            "High AI recall with collapsed specificity is NOT a robustness gain.",
        ]
    )
    for condition in ["original", "resize_112", "blur_sigma2", "screenshot_strong"]:
        for regime in ["A0", "A2"]:
            row = metrics[
                (metrics.regime == regime) & (metrics.split == "unseen_test") & (metrics.condition == condition)
            ].iloc[0]
            lines.append(
                f"{regime} {condition}: recall={row.recall:.3f} spec={row.specificity:.3f} FPR={row.fpr:.3f} "
                f"BalAcc={row.balanced_accuracy:.3f}"
            )
    lines.extend(["", "14. GENERATOR-SPECIFIC OBSERVATIONS (descriptive Stage-23D recalls)"])
    for generator in UNSEEN_GENERATORS:
        lines.append(f"{generator}:")
        for regime in REGIMES:
            vals = []
            for condition in CONDITIONS:
                g = gen[
                    (gen.regime == regime)
                    & (gen.split == "unseen_test")
                    & (gen.generator == generator)
                    & (gen.condition == condition)
                ].iloc[0]
                vals.append(f"{condition}={g.recall:.3f}")
            lines.append(f"  {regime}: " + "; ".join(vals))
    lines.extend(
        [
            "",
            "15. SEQUENTIAL-DESIGN LIMITATION",
            "RQ3 was motivated by RQ2 results on the same pilot benchmark.",
            "This is a sequential follow-up, not fully independent confirmation.",
            "Bootstrap estimates sampling uncertainty for fixed samples only.",
            "External/new-generator validation remains required.",
            "",
            "16. RQ3 CONCLUSION",
            "Primary validation-selected candidate: A2 (resize+JPEG-aware).",
            f"Unseen StrongRobustTestAUC A2−A0: {fmt_diff(u['ablations']['A2_vs_A0_strong_robust']['auc_diff'])}",
            f"Clean trade-off (unseen original AUC): {fmt_diff(u['condition_a2_vs_a0']['original']['auc_diff'])}",
            "Targeted JPEG/resize and several cross-transform conditions show positive A2−A0 ranking differences;",
            "interpret each CI individually. A1/A3 are secondary ablations and do not replace A2.",
            "",
            "17. SCIENTIFIC INTEGRITY",
            "New model training: NO",
            "New model inference: NO",
            "Model weights changed: NO",
            "Checkpoints changed: NO",
            "Thresholds changed: NO",
            "Primary candidate changed: NO",
            "Transformation-specific thresholds: NO",
            "Generator-specific thresholds: NO",
            "Samples removed after test results: NO",
            "A0 rerun: NO",
            "A1/A2/A3 rerun: NO",
            "Bootstrap uses existing frozen predictions: YES",
            "RQ3 model development reopened: NO",
            "RQ4 started: NO",
            f"Frozen thresholds used: A0={thresholds['A0']:.12f}, A1={thresholds['A1']:.12f}, "
            f"A2={thresholds['A2']:.12f}, A3={thresholds['A3']:.12f}",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("STAGE 23D VERIFICATION + STAGE 23E PAIRED BOOTSTRAP ANALYSIS")
    cache, metrics, thresholds = verify_and_build_cache()
    print("Precomputing paired bootstrap indices...")
    boot_idx = precompute_bootstrap_indices(cache)
    print(f"Running {N_BOOTSTRAP} paired replicates (analysis only)...")
    tidy_rows, payload = run_bootstrap(cache, boot_idx)
    tidy = pd.DataFrame(tidy_rows)

    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_tables(tidy, payload)
    plot_figures(payload)
    write_report(payload, metrics, thresholds)

    u = payload["unseen_test"]
    print("\nSTAGE 23D VERIFICATION + STAGE 23E STATISTICAL ANALYSIS COMPLETE")
    print("\nSTAGE 23D VERIFIED:\nYES")
    print(f"\nBootstrap:\nReplicates: {N_BOOTSTRAP}\nSeed: {RANDOM_SEED}")
    print(f"Method: {METHOD}")
    print("\nPRIMARY A2 VS A0 — UNSEEN AUC")
    for condition, label in [
        ("original", "Original"),
        ("jpeg_q50", "JPEG50"),
        ("resize_112", "Resize112"),
        ("blur_sigma2", "Blur2"),
        ("screenshot_strong", "ScreenshotStrong"),
    ]:
        d = u["condition_a2_vs_a0"][condition]["auc_diff"]
        print(f"\n{label}:\nDifference {d['observed']:+.6f}\n95% CI [{d['ci_95_low']:+.6f}, {d['ci_95_high']:+.6f}]")

    print("\nSTRONG ROBUST TEST AUC")
    for regime in REGIMES:
        print(f"{regime}: {u['strong_robust'][regime]['strong_robust_test_auc_observed']:.6f}")
    d = u["ablations"]["A2_vs_A0_strong_robust"]["auc_diff"]
    print(f"\nA2 - A0:\nDifference {d['observed']:+.6f}\n95% CI [{d['ci_95_low']:+.6f}, {d['ci_95_high']:+.6f}]")

    print("\nSECONDARY ABLATIONS")
    for cand in ["A1", "A3"]:
        d = u["ablations"][f"{cand}_vs_A0_strong_robust"]["auc_diff"]
        print(f"{cand} - A0 StrongRobustTestAUC:\n{d['observed']:+.6f} CI [{d['ci_95_low']:+.6f}, {d['ci_95_high']:+.6f}]")

    d = u["condition_a2_vs_a0"]["original"]["auc_diff"]
    print(f"\nCLEAN TRADE-OFF\nA2 - A0 original AUC:\n{d['observed']:+.6f} CI [{d['ci_95_low']:+.6f}, {d['ci_95_high']:+.6f}]")

    print("\nRQ3 PRIMARY CANDIDATE:\nA2 Resize+JPEG-aware MobileNetV3-Small")
    print("\nPrimary candidate changed after test:\nNO")
    print("\nRQ3:\nCOMPLETE")
    print("\nExternal independent confirmation:\nPENDING")
    print("\nModel training:\nNO")
    print("\nNew inference:\nNO")
    print("\nThreshold changes:\nNO")
    print("\nRQ4 started:\nNO")
    print("\nSTOP BEFORE RQ4.")


if __name__ == "__main__":
    main()
