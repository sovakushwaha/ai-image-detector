#!/usr/bin/env python3
"""
Stage 27A V2 — public-dataset external evaluation for FINAL_RESEARCH_MODEL_V1.

Runs ONLY after external_v2_data_readiness_v1.json gate passes.
Does NOT use fal.ai, retrain, refit T, or change selective bounds.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchvision import transforms
from tqdm import tqdm

from external_v2_common import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    FIGURES,
    META,
    PAPER_TABLES,
    RESULTS,
    ROOT,
    STRONG_ROBUST_CONDITIONS,
    TRANSFORM_FNS,
    controlled_preprocess,
    ece_15,
    load_frozen_config,
    nll_binary,
    selective_label,
)
from mobilenet_v3_small_binary_v1 import MobileNetV3SmallBinaryV1

DATA = ROOT / "data" / "external_v2"
READINESS = RESULTS / "external_v2_data_readiness_v1.json"
COND_MAP = {
    "original": "Original",
    "jpeg_q50": "JPEG50",
    "resize_112": "Resize112",
    "blur_sigma2": "Blur2",
    "screenshot_strong": "ScreenshotStrong",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_gate() -> dict:
    if not READINESS.exists():
        raise SystemExit("STOP: missing external_v2_data_readiness_v1.json")
    ready = json.loads(READINESS.read_text())
    if not ready.get("gate_pass"):
        raise SystemExit("STOP: Stage 27A V2 readiness gate failed")
    if ready.get("fal_api_used"):
        raise SystemExit("STOP: fal API must not be used in V2")
    return ready


def load_manifests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overlap = pd.read_csv(RESULTS / "external_overlap_audit_v2.csv")
    excl = set(overlap.loc[overlap["exclude_before_inference"], "image_id"])
    mllm = pd.read_csv(META / "external_mllm_manifest_v2.csv")
    mllm = mllm[~mllm["image_id"].isin(excl)].reset_index(drop=True)
    qwen = pd.read_csv(META / "external_qwen_image_bench_manifest_v2.csv")
    qwen = qwen[~qwen["image_id"].isin(excl)].reset_index(drop=True)
    coco = pd.read_csv(META / "external_coco_stress_manifest_v2.csv")
    return mllm, qwen, coco


def load_final_model(device: torch.device):
    cfg = load_frozen_config()
    ckpt = ROOT / cfg["pointer"]["checkpoint"]
    model = MobileNetV3SmallBinaryV1()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval().to(device)
    return model, cfg


def tf_controlled():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


@torch.inference_mode()
def predict_manifest(model, manifest: pd.DataFrame, device, transform, condition: str) -> tuple[np.ndarray, np.ndarray]:
    logits = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc=f"infer-{condition}", leave=False):
        img = controlled_preprocess(ROOT / r["native_path"])
        if condition != "original":
            img = TRANSFORM_FNS[condition](img)
        tensor = transform(img).unsqueeze(0).to(device)
        logits.append(model(tensor).squeeze().float().cpu().item())
    logits = np.asarray(logits, dtype=np.float64)
    raw_p = 1.0 / (1.0 + np.exp(-logits))
    return logits, raw_p


def write_transform_manifest(manifest: pd.DataFrame, tag: str) -> None:
    rows = []
    for _, r in manifest.iterrows():
        for cond in ["original"] + list(TRANSFORM_FNS.keys()):
            rows.append(
                {
                    "source_id": r["image_id"],
                    "condition": cond,
                    "label": r["label"],
                    "generator": r.get("generator", r.get("class_name", "")),
                    "source_path": r["native_path"],
                    "processed_path": "on_the_fly_controlled_preprocess",
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS / f"external_v2_{tag}_transform_manifest_v1.csv", index=False)


def binary_metrics(y, raw_p, cal_p, hist_thr, lower, upper) -> dict:
    pred = (raw_p >= hist_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sel = np.array([selective_label(p, lower, upper) for p in cal_p])
    accepted = sel != "UNCERTAIN"
    ai_mask = y == 1
    real_mask = y == 0
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "ai_recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "fnr": float(fn / (tp + fn)) if (tp + fn) else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "selective_coverage": float(accepted.mean()),
        "uncertain_rate": float((sel == "UNCERTAIN").mean()),
        "real_uncertain_rate": float((sel[real_mask] == "UNCERTAIN").mean()) if real_mask.any() else float("nan"),
        "ai_uncertain_rate": float((sel[ai_mask] == "UNCERTAIN").mean()) if ai_mask.any() else float("nan"),
    }
    if accepted.any():
        acc_pred = np.array([1 if s == "AI-GENERATED" else 0 for s in sel[accepted]])
        acc_y = y[accepted]
        out["selective_accuracy"] = float(np.mean(acc_pred == acc_y))
        out["selective_risk"] = 1.0 - out["selective_accuracy"]
        out["accepted_n"] = int(accepted.sum())
        if (sel[accepted] == "REAL").any():
            real_acc = sel[accepted] == "REAL"
            out["accepted_real_accuracy"] = float(np.mean(y[accepted][real_acc] == 0))
        else:
            out["accepted_real_accuracy"] = float("nan")
        if (sel[accepted] == "AI-GENERATED").any():
            ai_acc = sel[accepted] == "AI-GENERATED"
            out["accepted_ai_recall"] = float(np.mean(y[accepted][ai_acc] == 1))
        else:
            out["accepted_ai_recall"] = float("nan")
    else:
        out.update(
            {
                "selective_accuracy": float("nan"),
                "selective_risk": float("nan"),
                "accepted_n": 0,
                "accepted_real_accuracy": float("nan"),
                "accepted_ai_recall": float("nan"),
            }
        )
    if len(np.unique(y)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, cal_p))
        out["average_precision"] = float(average_precision_score(y, cal_p))
    return out


def run_inference_track(
    model,
    device,
    manifest: pd.DataFrame,
    T: float,
    hist_thr: float,
    lower: float,
    upper: float,
    source_dataset: str,
    *,
    with_transforms: bool,
) -> pd.DataFrame:
    conditions = ["original"] + (list(TRANSFORM_FNS.keys()) if with_transforms else [])
    rows = []
    for cond in conditions:
        logits, raw_p = predict_manifest(model, manifest, device, tf_controlled(), cond)
        cal_p = 1.0 / (1.0 + np.exp(-logits / T))
        for idx, (_, r) in enumerate(manifest.iterrows()):
            sel = selective_label(cal_p[idx], lower, upper)
            rows.append(
                {
                    "image_id": r["image_id"],
                    "source_dataset": source_dataset,
                    "generator": r.get("generator", r.get("class_name", "")),
                    "label": int(r["label"]),
                    "condition": cond,
                    "raw_logit": float(logits[idx]),
                    "raw_probability": float(raw_p[idx]),
                    "calibrated_probability": float(cal_p[idx]),
                    "binary_prediction": int(raw_p[idx] >= hist_thr),
                    "selective_prediction": sel,
                }
            )
    return pd.DataFrame(rows)


def reliability_figure(probs, labels, title, out_path):
    bins = np.linspace(0, 1, 16)
    centers, accs, confs = [], [], []
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & ((probs <= hi) if i == 14 else (probs < hi))
        if not mask.any():
            continue
        centers.append((lo + hi) / 2)
        accs.append(float(np.mean(labels[mask])))
        confs.append(float(np.mean(probs[mask])))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfect")
    ax.plot(confs, accs, "o-", lw=2, label="Model")
    ax.set_xlabel("Mean predicted P(AI)")
    ax.set_ylabel("Empirical AI rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def bootstrap_stratified(orig: pd.DataFrame, hist_thr, lower, upper) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    keys = ["roc_auc", "average_precision", "selective_coverage", "selective_risk"]
    store = {k: [] for k in keys}
    labels = orig["label"].to_numpy()
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    for _ in range(BOOTSTRAP_N):
        samp = np.concatenate([idx0[rng.integers(0, len(idx0), len(idx0))], idx1[rng.integers(0, len(idx1), len(idx1))]])
        s = orig.iloc[samp]
        m = binary_metrics(
            s["label"].to_numpy(),
            s["raw_probability"].to_numpy(),
            s["calibrated_probability"].to_numpy(),
            hist_thr,
            lower,
            upper,
        )
        for k in keys:
            store[k].append(m[k])
    return pd.DataFrame(
        [{"metric": k, "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)), "mean": float(np.mean(v))} for k, v in store.items()]
    )


def bootstrap_paired_auc(orig_df: pd.DataFrame, full_df: pd.DataFrame, hist_thr, lower, upper) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ids = orig_df["image_id"].tolist()
    rows = []
    for cond in STRONG_ROBUST_CONDITIONS + ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]:
        if cond not in full_df["condition"].unique():
            continue
    base = full_df[full_df["condition"] == "original"].set_index("image_id")
    store = {c: [] for c in STRONG_ROBUST_CONDITIONS}
    for _ in range(BOOTSTRAP_N):
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        y = base.loc[samp_ids, "label"].to_numpy()
        p0 = base.loc[samp_ids, "calibrated_probability"].to_numpy()
        auc0 = roc_auc_score(y, p0)
        for cond in STRONG_ROBUST_CONDITIONS:
            sub = full_df[full_df["condition"] == cond].set_index("image_id").loc[samp_ids]
            auc1 = roc_auc_score(y, sub["calibrated_probability"].to_numpy())
            store[cond].append(auc1 - auc0)
    out = []
    for cond in STRONG_ROBUST_CONDITIONS:
        v = store[cond]
        out.append({"metric": f"delta_auc_{cond}", "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)), "mean": float(np.mean(v))})
    # ExternalStrongRobustAUC bootstrap
    strong = []
    for _ in range(BOOTSTRAP_N):
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        y = base.loc[samp_ids, "label"].to_numpy()
        aucs = []
        for cond in STRONG_ROBUST_CONDITIONS:
            sub = full_df[full_df["condition"] == cond].set_index("image_id").loc[samp_ids]
            aucs.append(roc_auc_score(y, sub["calibrated_probability"].to_numpy()))
        strong.append(float(np.mean(aucs)))
    out.append({"metric": "ExternalStrongRobustAUC", "ci_low": float(np.percentile(strong, 2.5)), "ci_high": float(np.percentile(strong, 97.5)), "mean": float(np.mean(strong))})
    return pd.DataFrame(out)


def generator_metrics(df: pd.DataFrame, generators: list[str]) -> pd.DataFrame:
    rows = []
    for gen in generators:
        s = df[df["generator"] == gen]
        if len(s) == 0:
            continue
        y = s["label"].to_numpy()
        cal = s["calibrated_probability"].to_numpy()
        sel = s["selective_prediction"]
        rows.append(
            {
                "generator": gen,
                "n": len(s),
                "ai_recall": float((s["binary_prediction"] & (y == 1)).sum() / max((y == 1).sum(), 1)),
                "false_negative_rate": float(((s["binary_prediction"] == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                "mean_calibrated_p_ai": float(cal.mean()),
                "median_calibrated_p_ai": float(np.median(cal)),
                "uncertain_rate": float((sel == "UNCERTAIN").mean()),
                "ai_classification_rate": float((s["binary_prediction"] == 1).mean()),
                "accepted_ai_recall": float(
                    ((sel == "AI-GENERATED") & (y == 1)).sum() / max((sel == "AI-GENERATED").sum(), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    assert_gate()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)

    mllm, qwen, coco = load_manifests()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    write_transform_manifest(mllm, "mllm")

    model, cfg = load_final_model(device)
    T = cfg["temperature"]
    hist_thr = cfg["hist_thr"]
    lower = cfg["lower80"]
    upper = cfg["upper80"]
    print(f"Frozen T={T:.6f} hist_thr={hist_thr:.9f} selective=[{lower:.6f}, {upper:.6f}]")

    mllm_pred = run_inference_track(
        model, device, mllm, T, hist_thr, lower, upper, "MLLM", with_transforms=True
    )
    qwen_pred = run_inference_track(
        model, device, qwen, T, hist_thr, lower, upper, "Qwen-Image-Bench", with_transforms=False
    )
    coco_pred = run_inference_track(
        model, device, coco, T, hist_thr, lower, upper, "COCO_val2017", with_transforms=False
    )

    mllm_pred.to_csv(RESULTS / "external_v2_mllm_predictions_v1.csv", index=False)
    qwen_pred.to_csv(RESULTS / "external_v2_qwen_predictions_v1.csv", index=False)
    coco_pred.to_csv(RESULTS / "external_v2_coco_predictions_v1.csv", index=False)

    orig = mllm_pred[mllm_pred["condition"] == "original"]
    y = orig["label"].to_numpy()
    raw_p = orig["raw_probability"].to_numpy()
    cal_p = orig["calibrated_probability"].to_numpy()

    primary = binary_metrics(y, raw_p, cal_p, hist_thr, lower, upper)
    primary.update(
        {
            "raw_nll": nll_binary(raw_p, y),
            "raw_brier": float(brier_score_loss(y, raw_p)),
            "raw_ece15": ece_15(raw_p, y),
            "calibrated_nll": nll_binary(cal_p, y),
            "calibrated_brier": float(brier_score_loss(y, cal_p)),
            "calibrated_ece15": ece_15(cal_p, y),
        }
    )

    boot = bootstrap_stratified(orig, hist_thr, lower, upper)
    boot_delta = bootstrap_paired_auc(orig, mllm_pred, hist_thr, lower, upper)
    boot.to_csv(RESULTS / "external_v2_bootstrap_primary_v1.csv", index=False)
    boot_delta.to_csv(RESULTS / "external_v2_bootstrap_deltas_v1.csv", index=False)

    # Generator breakdown MLLM
    mllm_gen = generator_metrics(orig, ["GPT Image 2", "Nano Banana 2", "Real"])
    mllm_gen.to_csv(RESULTS / "external_v2_mllm_generator_metrics_v1.csv", index=False)

    # Qwen generators
    qorig = qwen_pred[qwen_pred["condition"] == "original"]
    qgen = generator_metrics(qorig, sorted(qorig["generator"].unique()))
    qgen = qgen.sort_values("ai_recall")
    qgen.to_csv(RESULTS / "external_v2_qwen_generator_metrics_v1.csv", index=False)

    # Robustness
    rob_rows = []
    orig_auc = primary["roc_auc"]
    orig_ap = primary["average_precision"]
    for cond in ["original"] + STRONG_ROBUST_CONDITIONS:
        s = mllm_pred[mllm_pred["condition"] == cond]
        m = binary_metrics(s["label"].to_numpy(), s["raw_probability"].to_numpy(), s["calibrated_probability"].to_numpy(), hist_thr, lower, upper)
        m.update(
            {
                "condition": COND_MAP.get(cond, cond),
                "delta_auc_vs_original": m.get("roc_auc", float("nan")) - orig_auc if cond != "original" else 0.0,
                "delta_ap_vs_original": m.get("average_precision", float("nan")) - orig_ap if cond != "original" else 0.0,
            }
        )
        rob_rows.append(m)
    rob_df = pd.DataFrame(rob_rows)
    strong_aucs = [rob_df.loc[rob_df["condition"] == COND_MAP[c], "roc_auc"].iloc[0] for c in STRONG_ROBUST_CONDITIONS]
    external_strong_robust_auc = float(np.mean(strong_aucs))

    # COCO stress
    corig = coco_pred[coco_pred["condition"] == "original"]
    coco_metrics = {
        "n": 400,
        "binary_fpr": float(corig["binary_prediction"].mean()),
        "binary_specificity": float(1 - corig["binary_prediction"].mean()),
        "mean_calibrated_p_ai": float(corig["calibrated_probability"].mean()),
        "median_calibrated_p_ai": float(np.median(corig["calibrated_probability"].to_numpy())),
        "selective_real_rate": float((corig["selective_prediction"] == "REAL").mean()),
        "selective_ai_rate": float((corig["selective_prediction"] == "AI-GENERATED").mean()),
        "selective_uncertain_rate": float((corig["selective_prediction"] == "UNCERTAIN").mean()),
        "accepted_real_accuracy": float(
            ((corig["selective_prediction"] == "REAL")).sum() and 1.0 or float("nan")
        ),
    }
    (RESULTS / "external_v2_coco_metrics_v1.json").write_text(json.dumps(coco_metrics, indent=2) + "\n")

    # Pilot comparison
    pilot = pd.read_csv(RESULTS / "rq5_calibration_test_metrics_v1.csv")
    pilot_o = pilot[(pilot["model"] == "C0") & (pilot["split"] == "unseen_test") & (pilot["condition"] == "original")].iloc[0]
    pilot_res = pd.read_csv(RESULTS / "resource_performance_context_v1.csv")
    pilot_r = pilot_res[pilot_res["model"] == "C0"].iloc[0]
    cmp_rows = [
        {"metric": "unseen_original_auc", "pilot": pilot_o["calibrated_roc_auc"], "external": primary["roc_auc"]},
        {"metric": "unseen_original_ap", "pilot": pilot_o["calibrated_ap"], "external": primary["average_precision"]},
        {"metric": "StrongRobustTestAUC", "pilot": pilot_r["unseen_strong_robust_test_auc"], "external": external_strong_robust_auc},
        {"metric": "calibrated_nll", "pilot": pilot_o["calibrated_nll"], "external": primary["calibrated_nll"]},
        {"metric": "selective_risk", "pilot": pilot_r["unseen_original_80pct_selective_risk"], "external": primary["selective_risk"]},
        {"metric": "selective_coverage", "pilot": float("nan"), "external": primary["selective_coverage"]},
    ]
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(RESULTS / "external_v2_pilot_comparison_v1.csv", index=False)

    # Paper tables
    pd.DataFrame([primary]).to_csv(PAPER_TABLES / "external_v2_primary_metrics.csv", index=False)
    mllm_gen.to_csv(PAPER_TABLES / "external_v2_generator_metrics.csv", index=False)
    qgen.to_csv(PAPER_TABLES / "external_v2_qwen_generator_metrics.csv", index=False)
    rob_df.to_csv(PAPER_TABLES / "external_v2_robustness.csv", index=False)
    pd.DataFrame(
        [
            {"type": "raw", **{k: primary[k] for k in ["raw_nll", "raw_brier", "raw_ece15"]}},
            {"type": "calibrated", **{k: primary[k] for k in ["calibrated_nll", "calibrated_brier", "calibrated_ece15"]}},
            {"type": "selective", "coverage": primary["selective_coverage"], "selective_risk": primary["selective_risk"], "uncertain_rate": primary["uncertain_rate"]},
        ]
    ).to_csv(PAPER_TABLES / "external_v2_calibration_selective.csv", index=False)
    cmp_df.to_csv(PAPER_TABLES / "external_v2_pilot_comparison.csv", index=False)

    # Figures
    fpr, tpr, _ = roc_curve(y, cal_p)
    prec, rec, _ = precision_recall_curve(y, cal_p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC={primary['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#888")
    axes[0].set_title("ROC — MLLM primary")
    axes[0].legend()
    axes[1].plot(rec, prec, lw=2, label=f"AP={primary['average_precision']:.3f}")
    axes[1].set_title("PR — MLLM primary")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "external_v2_mllm_roc_pr_v1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ai_gen = mllm_gen[mllm_gen["generator"].isin(["GPT Image 2", "Nano Banana 2"])]
    ax.bar(ai_gen["generator"], ai_gen["ai_recall"], color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("AI recall (frozen threshold)")
    ax.set_title("MLLM generator AI recall")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_v2_generator_recall_v1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(qgen["generator"], qgen["ai_recall"], color="#55A868")
    ax.set_xlabel("AI recall")
    ax.set_title("Qwen Image Bench — per-generator AI recall")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_v2_qwen_generator_recall_v1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rob_df["condition"], rob_df["roc_auc"], color="#8172B3")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("External robustness — ROC-AUC")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_v2_robustness_auc_v1.png", dpi=160)
    plt.close(fig)

    ddf = rob_df[rob_df["condition"] != "Original"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(ddf["condition"], ddf["delta_auc_vs_original"], color="#C44E52")
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylabel("ΔAUC vs Original")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_v2_robustness_delta_v1.png", dpi=160)
    plt.close(fig)

    reliability_figure(raw_p, y, "Raw reliability — MLLM", FIGURES / "external_v2_reliability_raw_v1.png")
    reliability_figure(cal_p, y, "Calibrated reliability — MLLM", FIGURES / "external_v2_reliability_calibrated_v1.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sel_counts = orig["selective_prediction"].value_counts()
    ax.bar(sel_counts.index, sel_counts.values, color=["#4C72B0", "#C44E52", "#CCB974"])
    ax.set_title("Selective prediction distribution — MLLM original")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_v2_selective_distribution_v1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cmp_df["metric"], cmp_df["external"] - cmp_df["pilot"], color="#8172B3")
    ax.axhline(0, color="#888")
    ax.set_title("External minus pilot (descriptive)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES / "external_v2_pilot_vs_external_v1.png", dpi=160)
    plt.close(fig)

    # Final summary JSON
    auc_ci = boot[boot["metric"] == "roc_auc"].iloc[0]
    ap_ci = boot[boot["metric"] == "average_precision"].iloc[0]
    strong_ci = boot_delta[boot_delta["metric"] == "ExternalStrongRobustAUC"].iloc[0]
    blur_row = rob_df[rob_df["condition"] == "Blur2"].iloc[0]

    summary = {
        "protocol": "Stage 27A V2 Public Dataset External Evaluation",
        "timestamp": utc_now(),
        "datasets": {
            "mllm_n": len(mllm),
            "qwen_n": len(qwen),
            "coco_n": len(coco),
        },
        "overlap_audit": {"development_exclusions": int(json.loads(READINESS.read_text())["development_overlap_exclusions"])},
        "primary_metrics": {
            **primary,
            "roc_auc_ci95": [auc_ci["ci_low"], auc_ci["ci_high"]],
            "ap_ci95": [ap_ci["ci_low"], ap_ci["ci_high"]],
        },
        "generator_metrics": mllm_gen.to_dict(orient="records"),
        "qwen_metrics": {"macro_ai_recall": float(qgen["ai_recall"].mean()), "generators": qgen.to_dict(orient="records")},
        "calibration": {
            "raw_nll": primary["raw_nll"],
            "calibrated_nll": primary["calibrated_nll"],
            "temperature_helped_nll": primary["calibrated_nll"] < primary["raw_nll"],
        },
        "selective_prediction": {k: primary[k] for k in ["selective_coverage", "selective_risk", "uncertain_rate", "accepted_ai_recall", "accepted_real_accuracy"]},
        "robustness": {
            "ExternalStrongRobustAUC": external_strong_robust_auc,
            "ExternalStrongRobustAUC_ci95": [strong_ci["ci_low"], strong_ci["ci_high"]],
            "by_condition": rob_df.to_dict(orient="records"),
        },
        "coco_real_stress": coco_metrics,
        "bootstrap": {"primary": boot.to_dict(orient="records"), "deltas": boot_delta.to_dict(orient="records")},
        "pilot_comparison": cmp_df.to_dict(orient="records"),
        "blur_failure_check": {
            "auc": float(blur_row["roc_auc"]),
            "specificity": float(blur_row["specificity"]),
            "ai_recall": float(blur_row["ai_recall"]),
            "pattern_high_ai_recall_collapsed_specificity": float(blur_row["ai_recall"]) > 0.8 and float(blur_row["specificity"]) < 0.7,
        },
        "integrity": {
            "training": False,
            "external_tuning": False,
            "temperature_refit": False,
            "policy_change": False,
            "threshold_change": False,
            "fal_usage": False,
            "historical_fal_images_evaluated": False,
        },
    }
    (RESULTS / "stage27a_v2_external_validation_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "=" * 50,
        "STAGE 27A V2 — EXTERNAL VALIDATION COMPLETE",
        "=" * 50,
        f"MLLM n={len(mllm)} | Qwen n={len(qwen)} | COCO n={len(coco)}",
        f"Primary ROC-AUC={primary['roc_auc']:.4f} [{auc_ci['ci_low']:.4f}, {auc_ci['ci_high']:.4f}]",
        f"Primary AP={primary['average_precision']:.4f} [{ap_ci['ci_low']:.4f}, {ap_ci['ci_high']:.4f}]",
        f"ExternalStrongRobustAUC={external_strong_robust_auc:.4f}",
        f"COCO FPR={coco_metrics['binary_fpr']:.4f}",
        "Fal images used: 0",
        "STATUS: COMPLETE",
    ]
    (RESULTS / "stage27a_v2_external_validation_report_v1.txt").write_text("\n".join(lines) + "\n")

    ready = json.loads(READINESS.read_text())
    ready["external_detector_predictions"] = len(mllm_pred) + len(qwen_pred) + len(coco_pred)
    ready["model_inference"] = "COMPLETE"
    (RESULTS / "external_v2_data_readiness_v1.json").write_text(json.dumps(ready, indent=2) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
