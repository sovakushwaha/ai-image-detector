#!/usr/bin/env python3
"""Stage 26B — evidence-based final model selection (analysis only, no inference)."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_TABLES = ROOT / "paper" / "tables"
MODELS = ROOT / "models"


def load_json(path: Path) -> dict | list:
    with path.open() as f:
        return json.load(f)


def get_rq4_row(metrics: list[dict], regime: str, split: str, condition: str) -> dict:
    for row in metrics:
        if row["regime"] == regime and row["split"] == split and row["condition"] == condition:
            return row
    raise KeyError(f"Missing RQ4 row: {regime}/{split}/{condition}")


def get_cal_row(rows: list[dict], model: str, split: str, condition: str) -> dict:
    for row in rows:
        if row["model"] == model and row["split"] == split and row["condition"] == condition:
            return row
    raise KeyError(f"Missing calibration row: {model}/{split}/{condition}")


def get_sel_row(rows: list[dict], model: str, split: str, condition: str, cov: int) -> dict:
    for row in rows:
        if (
            row["model"] == model
            and row["split"] == split
            and row["condition"] == condition
            and int(float(row["target_validation_coverage"])) == cov
        ):
            return row
    raise KeyError(f"Missing selective row: {model}/{split}/{condition}/{cov}")


def get_aurc(rows: list[dict], model: str, split: str, condition: str) -> float:
    for row in rows:
        if row["model"] == model and row["split"] == split and row["condition"] == condition:
            return float(row["aurc"])
    raise KeyError(f"Missing AURC: {model}/{split}/{condition}")


def get_bootstrap_diff(bootstrap: dict, path: list[str]) -> dict:
    node = bootstrap
    for key in path:
        node = node[key]
    return node


def pct_overhead(c1: float, c0: float) -> float:
    return (c1 / c0 - 1.0) * 100.0


def pct_delta(c1: float, c0: float) -> float:
    return ((c1 - c0) / c0) * 100.0


def build_evidence_table(evidence: dict) -> list[dict]:
    d = evidence["domains"]
    rows = [
        {
            "metric/domain": "Unseen original ROC-AUC",
            "C0": f"{d['discrimination']['unseen_original_auc_c0']:.6f}",
            "C1": f"{d['discrimination']['unseen_original_auc_c1']:.6f}",
            "preferred_candidate": "C1",
            "difference": f"+{d['discrimination']['unseen_original_auc_diff']:.6f}",
            "interpretation": "Modest C1 gain; 95% CI excludes zero",
            "evidence_source": "results/rq4_test_metrics_v1.csv; rq4_bootstrap_uncertainty_v1.json",
        },
        {
            "metric/domain": "Unseen original AP",
            "C0": f"{d['discrimination']['unseen_original_ap_c0']:.6f}",
            "C1": f"{d['discrimination']['unseen_original_ap_c1']:.6f}",
            "preferred_candidate": "C1",
            "difference": f"+{d['discrimination']['unseen_original_ap_diff']:.6f}",
            "interpretation": "Small C1 gain; 95% CI includes zero",
            "evidence_source": "results/rq4_bootstrap_uncertainty_v1.json",
        },
        {
            "metric/domain": "Known original ROC-AUC",
            "C0": f"{d['generalisation']['known_original_auc_c0']:.6f}",
            "C1": f"{d['generalisation']['known_original_auc_c1']:.6f}",
            "preferred_candidate": "tie",
            "difference": f"+{d['generalisation']['known_original_auc_c1'] - d['generalisation']['known_original_auc_c0']:.6f}",
            "interpretation": "Essentially tied on known generators",
            "evidence_source": "results/rq4_test_metrics_v1.csv",
        },
        {
            "metric/domain": "Original known→unseen AUC gap",
            "C0": f"{d['generalisation']['auc_gap_c0']:.6f}",
            "C1": f"{d['generalisation']['auc_gap_c1']:.6f}",
            "preferred_candidate": "C1",
            "difference": f"+{d['generalisation']['auc_gap_c1'] - d['generalisation']['auc_gap_c0']:.6f}",
            "interpretation": "C1 shows slightly smaller generalisation gap (less negative)",
            "evidence_source": "results/rq4_test_metrics_v1.csv",
        },
        {
            "metric/domain": "StrongRobustTestAUC (unseen)",
            "C0": f"{d['robustness']['strong_robust_auc_c0']:.6f}",
            "C1": f"{d['robustness']['strong_robust_auc_c1']:.6f}",
            "preferred_candidate": "C1",
            "difference": f"+{d['robustness']['strong_robust_auc_diff']:.6f}",
            "interpretation": "Modest consistent C1 gain; 95% CI excludes zero",
            "evidence_source": "results/rq4_bootstrap_uncertainty_v1.json",
        },
        {
            "metric/domain": "Blur2 unseen AUC (largest absolute C1 gain)",
            "C0": f"{d['robustness']['blur2_auc_c0']:.6f}",
            "C1": f"{d['robustness']['blur2_auc_c1']:.6f}",
            "preferred_candidate": "C1",
            "difference": f"+{d['robustness']['blur2_auc_diff']:.6f}",
            "interpretation": "Largest per-condition AUC advantage for C1",
            "evidence_source": "results/rq4_bootstrap_uncertainty_v1.json",
        },
        {
            "metric/domain": "ScreenshotStrong unseen AUC",
            "C0": f"{d['robustness']['screenshot_auc_c0']:.6f}",
            "C1": f"{d['robustness']['screenshot_auc_c1']:.6f}",
            "preferred_candidate": "C0",
            "difference": f"{d['robustness']['screenshot_auc_diff']:.6f}",
            "interpretation": "C1 not improved; 95% CI includes zero",
            "evidence_source": "results/rq4_bootstrap_uncertainty_v1.json",
        },
        {
            "metric/domain": "Calibrated unseen-original NLL",
            "C0": f"{d['calibration']['nll_c0']:.6f}",
            "C1": f"{d['calibration']['nll_c1']:.6f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['calibration']['nll_c1'] - d['calibration']['nll_c0']:.6f}",
            "interpretation": "C0 lower NLL (better calibrated probabilities)",
            "evidence_source": "results/rq5_calibration_test_metrics_v1.csv",
        },
        {
            "metric/domain": "Calibrated unseen-original ECE-15",
            "C0": f"{d['calibration']['ece_c0']:.6f}",
            "C1": f"{d['calibration']['ece_c1']:.6f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['calibration']['ece_c1'] - d['calibration']['ece_c0']:.6f}",
            "interpretation": "C0 lower ECE on unseen original",
            "evidence_source": "results/rq5_calibration_test_metrics_v1.csv",
        },
        {
            "metric/domain": "80% selective risk (unseen original)",
            "C0": f"{d['selective']['risk80_c0']:.6f}",
            "C1": f"{d['selective']['risk80_c1']:.6f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['selective']['risk80_diff']:.6f}",
            "interpretation": "C1 higher retained risk; 95% CI excludes zero",
            "evidence_source": "results/rq5_selective_test_metrics_v1.csv; rq5_bootstrap_uncertainty_v1.json",
        },
        {
            "metric/domain": "80% actual coverage (unseen original)",
            "C0": f"{d['selective']['cov80_c0']:.4f}",
            "C1": f"{d['selective']['cov80_c1']:.4f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['selective']['cov80_c1'] - d['selective']['cov80_c0']:.4f}",
            "interpretation": "C1 drifts above nominal 80% target",
            "evidence_source": "results/rq5_selective_test_metrics_v1.csv",
        },
        {
            "metric/domain": "AURC (unseen original)",
            "C0": f"{d['selective']['aurc_c0']:.6f}",
            "C1": f"{d['selective']['aurc_c1']:.6f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['selective']['aurc_c1'] - d['selective']['aurc_c0']:.6f}",
            "interpretation": "C0 better risk–coverage ordering on unseen original",
            "evidence_source": "results/rq5_risk_coverage_v1.csv",
        },
        {
            "metric/domain": "Parameters",
            "C0": str(d["resource"]["params_c0"]),
            "C1": str(d["resource"]["params_c1"]),
            "preferred_candidate": "C0",
            "difference": f"+{d['resource']['param_overhead_pct']:.1f}%",
            "interpretation": "C1 adds 393,186 parameters",
            "evidence_source": "results/resource_model_size_v1.csv",
        },
        {
            "metric/domain": "Deployable state dict (MiB)",
            "C0": f"{d['resource']['state_c0']:.2f}",
            "C1": f"{d['resource']['state_c1']:.2f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['resource']['state_overhead_pct']:.1f}%",
            "interpretation": "C1 larger deployable weight state",
            "evidence_source": "results/resource_model_size_v1.csv",
        },
        {
            "metric/domain": "CPU end-to-end latency median (ms)",
            "C0": f"{d['resource']['cpu_e2e_c0']:.2f}",
            "C1": f"{d['resource']['cpu_e2e_c1']:.2f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['resource']['cpu_e2e_overhead_pct']:.1f}%",
            "interpretation": "C1 slower including preprocessing",
            "evidence_source": "results/resource_efficiency_summary_v1.csv",
        },
        {
            "metric/domain": "MPS end-to-end latency median (ms)",
            "C0": f"{d['resource']['mps_e2e_c0']:.2f}",
            "C1": f"{d['resource']['mps_e2e_c1']:.2f}",
            "preferred_candidate": "C0",
            "difference": f"+{d['resource']['mps_e2e_overhead_pct']:.1f}%",
            "interpretation": "Larger relative overhead on Apple MPS",
            "evidence_source": "results/resource_efficiency_summary_v1.csv",
        },
        {
            "metric/domain": "MPS batch-32 throughput (images/sec)",
            "C0": f"{d['resource']['mps_tp_c0']:.1f}",
            "C1": f"{d['resource']['mps_tp_c1']:.1f}",
            "preferred_candidate": "C0",
            "difference": f"{d['resource']['mps_tp_delta_pct']:.1f}%",
            "interpretation": "C1 substantially lower batched forward throughput",
            "evidence_source": "results/resource_efficiency_summary_v1.csv",
        },
        {
            "metric/domain": "Deployment complexity",
            "C0": "RGB only; ImageNet norm",
            "C1": "RGB + FrequencyTransformV1 FFT branch",
            "preferred_candidate": "C0",
            "difference": "FFT preprocessing required",
            "interpretation": "C1 adds frequency pipeline and dual-branch inference",
            "evidence_source": "results/rq4_F2_frozen_config_v1.json; Stage 26A",
        },
    ]
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rq4_metrics = list(csv.DictReader((RESULTS / "rq4_test_metrics_v1.csv").open()))
    rq4_boot = load_json(RESULTS / "rq4_bootstrap_uncertainty_v1.json")
    cal_rows = list(csv.DictReader((RESULTS / "rq5_calibration_test_metrics_v1.csv").open()))
    sel_rows = list(csv.DictReader((RESULTS / "rq5_selective_test_metrics_v1.csv").open()))
    rq5_boot = load_json(RESULTS / "rq5_bootstrap_uncertainty_v1.json")
    res_summary = list(csv.DictReader((RESULTS / "resource_efficiency_summary_v1.csv").open()))
    res_size = list(csv.DictReader((RESULTS / "resource_model_size_v1.csv").open()))
    c0_temp = load_json(RESULTS / "rq5_C0_temperature_scaling_v1.json")
    c1_temp = load_json(RESULTS / "rq5_C1_temperature_scaling_v1.json")
    c0_sel = load_json(RESULTS / "rq5_C0_selective_policy_v1.json")
    c1_sel = load_json(RESULTS / "rq5_C1_selective_policy_v1.json")
    c0_frozen = load_json(RESULTS / "rq3_A2_frozen_config_v1.json")
    c1_frozen = load_json(RESULTS / "rq4_F2_frozen_config_v1.json")

    f0_orig = get_rq4_row(rq4_metrics, "F0", "unseen_test", "original")
    f2_orig = get_rq4_row(rq4_metrics, "F2", "unseen_test", "original")
    f0_known = get_rq4_row(rq4_metrics, "F0", "known_test", "original")
    f2_known = get_rq4_row(rq4_metrics, "F2", "known_test", "original")

    unseen_boot = rq4_boot["unseen_test"]["F2_vs_F0"]
    strong_boot = unseen_boot["StrongRobustTest"]["auc_diff"]
    orig_boot = unseen_boot["original"]["auc_diff"]
    ap_boot = unseen_boot["original"]["ap_diff"]
    blur_boot = unseen_boot["blur_sigma2"]["auc_diff"]
    ss_boot = unseen_boot["screenshot_strong"]["auc_diff"]

    cal_c0 = get_cal_row(cal_rows, "C0", "unseen_test", "original")
    cal_c1 = get_cal_row(cal_rows, "C1", "unseen_test", "original")
    sel_c0 = get_sel_row(sel_rows, "C0", "unseen_test", "original", 80)
    sel_c1 = get_sel_row(sel_rows, "C1", "unseen_test", "original", 80)

    risk80_boot = next(
        r
        for r in rq5_boot
        if r["analysis_type"] == "c1_vs_c0_selective"
        and r["split"] == "unseen_test"
        and r["condition"] == "original"
        and r["metric"] == "selective_risk_80"
    )

    res_c0_cpu = next(r for r in res_summary if r["model"] == "C0" and r["device"] == "cpu")
    res_c1_cpu = next(r for r in res_summary if r["model"] == "C1" and r["device"] == "cpu")
    res_c0_mps = next(r for r in res_summary if r["model"] == "C0" and r["device"] == "mps")
    res_c1_mps = next(r for r in res_summary if r["model"] == "C1" and r["device"] == "mps")
    size_c0 = next(r for r in res_size if r["model"] == "C0")
    size_c1 = next(r for r in res_size if r["model"] == "C1")

    evidence = {
        "domains": {
            "discrimination": {
                "unseen_original_auc_c0": float(f0_orig["roc_auc"]),
                "unseen_original_auc_c1": float(f2_orig["roc_auc"]),
                "unseen_original_auc_diff": float(orig_boot["observed"]),
                "unseen_original_auc_ci": [orig_boot["ci_95_low"], orig_boot["ci_95_high"]],
                "unseen_original_ap_c0": float(f0_orig["average_precision"]),
                "unseen_original_ap_c1": float(f2_orig["average_precision"]),
                "unseen_original_ap_diff": float(ap_boot["observed"]),
                "unseen_original_ap_ci_includes_zero": bool(ap_boot["includes_zero"]),
            },
            "generalisation": {
                "known_original_auc_c0": float(f0_known["roc_auc"]),
                "known_original_auc_c1": float(f2_known["roc_auc"]),
                "auc_gap_c0": float(f0_orig["generalisation_gap_auc"]),
                "auc_gap_c1": float(f2_orig["generalisation_gap_auc"]),
            },
            "robustness": {
                "strong_robust_auc_c0": float(rq4_boot["unseen_test"]["strong_robust"]["F0"]["strong_robust_test_auc_observed"]),
                "strong_robust_auc_c1": float(rq4_boot["unseen_test"]["strong_robust"]["F2"]["strong_robust_test_auc_observed"]),
                "strong_robust_auc_diff": float(strong_boot["observed"]),
                "strong_robust_auc_ci": [strong_boot["ci_95_low"], strong_boot["ci_95_high"]],
                "blur2_auc_c0": float(get_rq4_row(rq4_metrics, "F0", "unseen_test", "blur_sigma2")["roc_auc"]),
                "blur2_auc_c1": float(get_rq4_row(rq4_metrics, "F2", "unseen_test", "blur_sigma2")["roc_auc"]),
                "blur2_auc_diff": float(blur_boot["observed"]),
                "screenshot_auc_c0": float(get_rq4_row(rq4_metrics, "F0", "unseen_test", "screenshot_strong")["roc_auc"]),
                "screenshot_auc_c1": float(get_rq4_row(rq4_metrics, "F2", "unseen_test", "screenshot_strong")["roc_auc"]),
                "screenshot_auc_diff": float(ss_boot["observed"]),
            },
            "calibration": {
                "nll_c0": float(cal_c0["calibrated_nll"]),
                "nll_c1": float(cal_c1["calibrated_nll"]),
                "ece_c0": float(cal_c0["calibrated_ece15"]),
                "ece_c1": float(cal_c1["calibrated_ece15"]),
            },
            "selective": {
                "risk80_c0": float(sel_c0["selective_risk"]),
                "risk80_c1": float(sel_c1["selective_risk"]),
                "risk80_diff": float(risk80_boot["observed_difference"]),
                "risk80_ci": [risk80_boot["ci_low"], risk80_boot["ci_high"]],
                "cov80_c0": float(sel_c0["achieved_coverage"]),
                "cov80_c1": float(sel_c1["achieved_coverage"]),
                "aurc_c0": get_aurc(list(csv.DictReader((RESULTS / "rq5_risk_coverage_v1.csv").open())), "C0", "unseen_test", "original"),
                "aurc_c1": get_aurc(list(csv.DictReader((RESULTS / "rq5_risk_coverage_v1.csv").open())), "C1", "unseen_test", "original"),
            },
            "resource": {
                "params_c0": int(size_c0["parameters"]),
                "params_c1": int(size_c1["parameters"]),
                "param_overhead_pct": pct_overhead(int(size_c1["parameters"]), int(size_c0["parameters"])),
                "state_c0": float(size_c0["state_dict_size_mib"]),
                "state_c1": float(size_c1["state_dict_size_mib"]),
                "state_overhead_pct": pct_overhead(float(size_c1["state_dict_size_mib"]), float(size_c0["state_dict_size_mib"])),
                "cpu_e2e_c0": float(res_c0_cpu["end_to_end_median_ms"]),
                "cpu_e2e_c1": float(res_c1_cpu["end_to_end_median_ms"]),
                "cpu_e2e_overhead_pct": pct_overhead(float(res_c1_cpu["end_to_end_median_ms"]), float(res_c0_cpu["end_to_end_median_ms"])),
                "mps_e2e_c0": float(res_c0_mps["end_to_end_median_ms"]),
                "mps_e2e_c1": float(res_c1_mps["end_to_end_median_ms"]),
                "mps_e2e_overhead_pct": pct_overhead(float(res_c1_mps["end_to_end_median_ms"]), float(res_c0_mps["end_to_end_median_ms"])),
                "mps_tp_c0": float(res_c0_mps["throughput_batch32_ips"]),
                "mps_tp_c1": float(res_c1_mps["throughput_batch32_ips"]),
                "mps_tp_delta_pct": pct_delta(float(res_c1_mps["throughput_batch32_ips"]), float(res_c0_mps["throughput_batch32_ips"])),
            },
        }
    }

    # Pareto: neither dominates — C1 better discrimination/robustness; C0 better trust/resource
    final_model_id = "FINAL_RESEARCH_MODEL_V1"
    selected = "C0"
    selection_rationale = (
        "C0 (RQ3 A2 robust RGB MobileNet) is selected because discrimination and transformation "
        "robustness remain strong and close to C1, while C0 provides better calibrated probability "
        "quality (lower unseen-original NLL/ECE), lower 80%-policy selective risk, lower AURC, and "
        "substantially lower deployment cost (26% fewer parameters, 28% lower MPS end-to-end latency, "
        "32% lower MPS batch-32 throughput, no FFT branch). C1's incremental unseen AUC (+0.004) and "
        "StrongRobustTestAUC (+0.006) gains are modest and do not justify the combined trustworthiness "
        "and resource penalties under the project's resource-aware + trustworthy objective."
    )
    rejected_rationale = (
        "C1 (RGB+frequency fusion) is not selected despite modest statistically separated gains on "
        "unseen original AUC (+0.004) and StrongRobustTestAUC (+0.006), primarily under blur and resize. "
        "C1 exhibits worse calibrated unseen-original NLL (+0.018), higher 80% selective risk "
        "(+0.027 [+0.017, +0.037]), higher AURC (+0.009), requires larger temperature correction "
        "(T=3.386 vs 1.904), adds FFT preprocessing complexity, and incurs ~26% parameter/state overhead "
        "with materially lower throughput. Neither model is fully trustworthy under severe blur."
    )

    final_config = {
        "final_model_id": final_model_id,
        "selected_candidate": selected,
        "historical_model_id": "rq5_C0_rgb_a2",
        "historical_aliases": ["A2", "F0", "resource_C0_rgb"],
        "checkpoint": c0_frozen["checkpoint"],
        "architecture": c0_frozen["model"],
        "parameter_count": c0_frozen["total_parameters"],
        "preprocessing": "controlled_v1 RGB 224x224; ToTensor; ImageNet normalization",
        "frequency_branch_used": False,
        "temperature_config": "results/rq5_C0_temperature_scaling_v1.json",
        "selective_policy_config": "results/rq5_C0_selective_policy_v1.json",
        "frozen_config": "results/rq3_A2_frozen_config_v1.json",
        "binary_historical_threshold": c0_frozen["threshold"],
        "calibrated_temperature": c0_temp["temperature"],
        "primary_selective_coverage": 0.8,
        "real_ai_uncertain_bounds": {
            "real_if_p_leq": c0_sel["lower80"],
            "ai_generated_if_p_geq": c0_sel["upper80"],
            "uncertain_otherwise": True,
            "gamma80": c0_sel["gamma80"],
        },
        "selection_date": str(date.today()),
        "selection_stage": "26B",
        "selection_rationale": selection_rationale,
        "rejected_candidate": "C1",
        "rejected_rationale": rejected_rationale,
        "evidence_summary": evidence["domains"],
        "pareto_analysis": {
            "c0_strictly_dominates_c1": False,
            "c1_strictly_dominates_c0": False,
            "neither_strictly_dominates": True,
            "explanation": (
                "C1 is preferred on modest discrimination/robustness metrics; C0 is preferred on "
                "calibration, selective trustworthiness, resource efficiency, and deployment simplicity. "
                "Neither candidate is no-worse across all major domains."
            ),
        },
        "known_limitations": [
            "Sequential pilot benchmark only; not independent external confirmation",
            "Limited generator holdout; source independence not fully guaranteed",
            "Temperature and selective policies fit on clean validation only",
            "Severe blur causes saturated overconfidence; abstention fails for both models",
            "Physical screen recapture not tested",
            "Resource measurements from one Apple M1 environment",
            "Research prototype; not production-certified",
        ],
        "external_confirmation_status": "PENDING",
    }

    pointer = {
        "final_model_id": final_model_id,
        "selected_candidate": selected,
        "checkpoint": c0_frozen["checkpoint"],
        "frozen_config": "results/rq3_A2_frozen_config_v1.json",
        "temperature_config": "results/rq5_C0_temperature_scaling_v1.json",
        "selective_policy_config": "results/rq5_C0_selective_policy_v1.json",
        "selection_config": "results/final_model_selection_v1.json",
    }

    evidence_rows = build_evidence_table({"domains": evidence["domains"]})
    write_csv(PAPER_TABLES / "final_model_selection_evidence.csv", evidence_rows)

    (RESULTS / "final_model_selection_v1.json").write_text(json.dumps(final_config, indent=2) + "\n")
    (MODELS / "FINAL_MODEL_V1.json").write_text(json.dumps(pointer, indent=2) + "\n")

    d = evidence["domains"]
    report = f"""Stage 26B — Final Model Selection Report
============================================================

1. PURPOSE
   Select one final research/deployment candidate from frozen C0 and C1 using
   accumulated RQ1–RQ5 and Stage 26A evidence. Analysis only; no new inference.

2. CANDIDATES
   C0: Robust RGB MobileNetV3-Small (RQ3 A2 / F0)
   C1: RGB + Frequency Fusion (RQ4 F2)

3. SELECTION PRINCIPLE
   Multi-criteria evidence comparison across discrimination, robustness,
   calibration/selective trustworthiness, and resource efficiency.
   No arbitrary weighted composite score.

4. DISCRIMINATION EVIDENCE
   Unseen original AUC: C0 {d['discrimination']['unseen_original_auc_c0']:.6f};
   C1 {d['discrimination']['unseen_original_auc_c1']:.6f};
   diff +{d['discrimination']['unseen_original_auc_diff']:.6f}
   95% CI [{d['discrimination']['unseen_original_auc_ci'][0]:.6f}, {d['discrimination']['unseen_original_auc_ci'][1]:.6f}]

5. GENERALISATION EVIDENCE
   Known original AUC: C0 {d['generalisation']['known_original_auc_c0']:.6f};
   C1 {d['generalisation']['known_original_auc_c1']:.6f}
   AUC gap (unseen−known): C0 {d['generalisation']['auc_gap_c0']:.6f};
   C1 {d['generalisation']['auc_gap_c1']:.6f}

6. ROBUSTNESS EVIDENCE
   StrongRobustTestAUC: C0 {d['robustness']['strong_robust_auc_c0']:.6f};
   C1 {d['robustness']['strong_robust_auc_c1']:.6f};
   diff +{d['robustness']['strong_robust_auc_diff']:.6f}
   95% CI [{d['robustness']['strong_robust_auc_ci'][0]:.6f}, {d['robustness']['strong_robust_auc_ci'][1]:.6f}]
   Largest C1 absolute gain: blur2 AUC +{d['robustness']['blur2_auc_diff']:.6f}
   C1 not improved: screenshot_strong AUC {d['robustness']['screenshot_auc_diff']:.6f}

7. CALIBRATION EVIDENCE
   Calibrated unseen-original NLL: C0 {d['calibration']['nll_c0']:.6f};
   C1 {d['calibration']['nll_c1']:.6f}
   ECE-15: C0 {d['calibration']['ece_c0']:.6f}; C1 {d['calibration']['ece_c1']:.6f}

8. SELECTIVE-PREDICTION EVIDENCE
   80% policy selective risk (unseen original): C0 {d['selective']['risk80_c0']:.6f};
   C1 {d['selective']['risk80_c1']:.6f};
   C1−C0 +{d['selective']['risk80_diff']:.6f}
   95% CI [{d['selective']['risk80_ci'][0]:.6f}, {d['selective']['risk80_ci'][1]:.6f}]
   AURC: C0 {d['selective']['aurc_c0']:.6f}; C1 {d['selective']['aurc_c1']:.6f}

9. RESOURCE EVIDENCE
   Parameters: C0 {d['resource']['params_c0']:,}; C1 {d['resource']['params_c1']:,}
   (+{d['resource']['param_overhead_pct']:.1f}%)
   State dict: C0 {d['resource']['state_c0']:.2f} MiB; C1 {d['resource']['state_c1']:.2f} MiB
   CPU E2E: C0 {d['resource']['cpu_e2e_c0']:.2f} ms; C1 {d['resource']['cpu_e2e_c1']:.2f} ms
   MPS E2E: C0 {d['resource']['mps_e2e_c0']:.2f} ms; C1 {d['resource']['mps_e2e_c1']:.2f} ms
   MPS batch-32 throughput: C0 {d['resource']['mps_tp_c0']:.1f}; C1 {d['resource']['mps_tp_c1']:.1f} ips

10. PARETO ANALYSIS
    C0 strictly dominates C1: NO
    C1 strictly dominates C0: NO
    Neither strictly dominates across all domains.

11. PRACTICAL TRADE-OFF
    C1 gains +{d['discrimination']['unseen_original_auc_diff']:.4f} unseen AUC and
    +{d['robustness']['strong_robust_auc_diff']:.4f} StrongRobustTestAUC in exchange for
    +{d['resource']['param_overhead_pct']:.1f}% parameters, +{d['resource']['mps_e2e_overhead_pct']:.1f}%
    MPS end-to-end latency, {d['resource']['mps_tp_delta_pct']:.1f}% MPS batch-32 throughput,
    and worse calibrated selective behaviour (+{d['selective']['risk80_diff']:.4f} selective risk).

12. FINAL DECISION
    {final_model_id}: {selected}
    Checkpoint: {c0_frozen['checkpoint']}

13. WHY THE OTHER MODEL WAS NOT SELECTED
    {rejected_rationale}

14. WHAT THE SELECTED MODEL DOES WELL
    Strong unseen-generator discrimination; best calibration/selective metrics among finalists;
    simpler RGB-only deployment; lower latency and higher throughput.

15. WHAT THE SELECTED MODEL STILL FAILS ON
    Severe blur (near-random threshold behaviour, useless abstention); generator holdout not
    externally confirmed; moderate miscalibration under shift; physical recapture untested.

16. EXTERNAL-CONFIRMATION REQUIREMENT
    Status: PENDING — independent modern-generator evaluation required before deployment claims.

17. DEPLOYMENT IMPLICATIONS
    Use C0 with frozen temperature T={c0_temp['temperature']:.6f} and 80% selective policy
    (REAL p≤{c0_sel['lower80']:.6f}; AI p≥{c0_sel['upper80']:.6f}; else UNCERTAIN).
    Historical binary Youden threshold {c0_frozen['threshold']:.6f} retained for RQ1–RQ4 reporting.

18. SCIENTIFIC INTEGRITY
    New training: NO | New inference: NO | Weights changed: NO | Composite score: NO
    Final model selected: YES | External confirmation: NO
"""
    (RESULTS / "final_model_selection_report_v1.txt").write_text(report)

    print("=" * 50)
    print("STAGE 26B — FINAL MODEL SELECTION COMPLETE")
    print("=" * 50)
    print(f"\nFINAL DECISION: {final_model_id} = {selected}")
    print(f"Checkpoint: {c0_frozen['checkpoint']}")
    print(f"Temperature: {c0_temp['temperature']:.6f}")
    print(f"REAL: p <= {c0_sel['lower80']:.6f}")
    print(f"AI-GENERATED: p >= {c0_sel['upper80']:.6f}")
    print("\nSee results/final_model_selection_report_v1.txt for full summary.")


if __name__ == "__main__":
    main()
