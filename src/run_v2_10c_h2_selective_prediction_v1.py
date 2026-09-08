#!/usr/bin/env python3
"""V2-10C — H2 Confidence Calibration + Selective Prediction.

C0 = raw H2 probability
C1 = temperature-scaled H2 probability (T fitted on V2-9C calib only)

Selective policies S95/S90/S80 from calibration confidence quantiles.
Primary policy = S90. Binary threshold remains p=0.5.

NO LoRA/CLIP/H2 retraining. NO Platt. NO threshold tuning.
FINAL_V2_MODEL NOT SELECTED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_v2_10a_representation_geometry_v1 import (  # noqa: E402
    ClipLoRAModel,
    PathDataset,
    resolve_path,
)
from rq5_calibration_utils_v1 import (  # noqa: E402
    apply_temperature,
    compute_brier,
    compute_ece15,
    compute_nll,
    fit_temperature,
    reliability_curve,
    sigmoid,
)

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
CALIB_IDS = OUT / "v2_9c_candidate_calibration_ids_v1.csv"
EVAL_PRED = OUT / "v2_10b_head_predictions_v1.csv"
CKPT_DIR = MODELS

SEED = 42
EPS = 1e-6
THR = 0.5
EXPECTED_CALIB = {1: 2123, 2: 2323, 3: 2223, 4: 2223}
EXPECTED_EVAL = {1: 2821, 2: 2619, 3: 1994, 4: 1709}
HARD_GENERATORS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]
REAL_DOMAINS = ["Tiny", "MLLM", "COCO", "Smartphone"]
# predeclared coverage targets; S90 PRIMARY
POLICIES = [("S95", 0.95), ("S90", 0.90), ("S80", 0.80)]
PRIMARY = "S90"

JSON_OUT = OUT / "v2_10c_selective_analysis_v1.json"
REPORT_OUT = OUT / "v2_10c_selective_report_v1.txt"
PRED_OUT = OUT / "v2_10c_selective_predictions_v1.csv"
METRICS_OUT = OUT / "v2_10c_selective_metrics_v1.csv"
TEMP_OUT = OUT / "v2_10c_temperature_params_v1.csv"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def to_logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def confidence(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.maximum(p, 1.0 - p)


def ece_bins(probs: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, 16)
    out = []
    n = len(labels)
    for i in range(15):
        lo, hi = bins[i], bins[i + 1]
        if i < 14:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        row = {"bin": i, "lo": float(lo), "hi": float(hi), "count": count}
        if count:
            row["mean_conf"] = float(probs[mask].mean())
            row["mean_acc"] = float(labels[mask].mean())
        else:
            row["mean_conf"] = None
            row["mean_acc"] = None
        row["weight"] = count / n if n else 0.0
        out.append(row)
    return out


def train_membership_ids(man: pd.DataFrame, fold: int) -> set[str]:
    CAP = 300
    col = f"fold_{fold}_role"
    base = man[man["fold_1_role"] != "EXCLUDED_DUPLICATE"]
    ai = base[(base["binary_label"] == 1) & (base[col] == "TRAIN")].copy()
    keep = []
    for _, gdf in ai.groupby("canonical_generator_id"):
        gdf = gdf.sort_values("image_id")
        keep.append(gdf.iloc[:CAP] if len(gdf) > CAP else gdf)
    ai_keep = pd.concat(keep) if keep else ai.iloc[:0]
    real = base[(base["binary_label"] == 0) & (base[col] == "TRAIN")]
    return set(pd.concat([ai_keep, real])["image_id"].astype(str))


def load_primary_calib(man: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_csv(CALIB_IDS)
    c["image_id"] = c["image_id"].astype(str)
    p = c[c["in_primary_proposal"] == True].copy()  # noqa: E712
    if p["overlap_train"].any() or p["overlap_eval"].any():
        stop("V2-9C primary calib has overlap flags True")
    for fold, exp in EXPECTED_CALIB.items():
        pf = p[p["fold"] == fold]
        if len(pf) != exp:
            stop(f"fold {fold} primary calib n={len(pf)} != {exp}")
        if pf["image_id"].duplicated().any():
            stop(f"fold {fold} duplicate calib IDs")
        # roles
        n_ih = int((pf["original_role"] == "REAL_INTERNAL_HOLDOUT").sum())
        n_pb = int((pf["original_role"] == "PROMPT_BLOCKED").sum())
        if n_ih != 923:
            stop(f"fold {fold} IH n={n_ih} != 923")
        if n_ih + n_pb != exp:
            stop(f"fold {fold} role sum mismatch")
        # overlap with train/eval
        tr = train_membership_ids(man, fold)
        ev = set(
            pd.read_csv(EVAL_PRED)
            .query(f"fold == {fold}")["image_id"]
            .astype(str)
        )
        cal = set(pf["image_id"])
        if cal & tr:
            stop(f"fold {fold} calib overlaps train n={len(cal & tr)}")
        if cal & ev:
            stop(f"fold {fold} calib overlaps eval n={len(cal & ev)}")
        # match V2-9C score IDs
        sc = pd.read_csv(OUT / f"v2_9c_calibration_scores_fold{fold}_v1.csv")
        if set(sc["image_id"].astype(str)) != cal:
            stop(f"fold {fold} primary IDs != V2-9C score IDs")
    return p.reset_index(drop=True)


def extract_calib_r1(man: pd.DataFrame, calib: pd.DataFrame) -> dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta: dict[str, Any] = {"performed": False, "device": str(device), "folds": {}}
    man_ix = man.set_index("image_id", drop=False)

    for fold in [1, 2, 3, 4]:
        out_npz = OUT / f"v2_10c_calibration_r1_fold{fold}_v1.npz"
        pf = calib[calib["fold"] == fold].sort_values("image_id").reset_index(drop=True)
        if out_npz.is_file():
            z = np.load(out_npz, allow_pickle=False)
            ids = z["image_ids"].astype(str)
            if set(ids) != set(pf["image_id"]) or len(ids) != len(pf):
                stop(f"existing calib R1 fold {fold} ID mismatch — delete and re-extract")
            if z["r1"].shape[1] != 512:
                stop(f"fold {fold} calib R1 D != 512")
            meta["folds"][f"fold_{fold}"] = {
                "reused": True,
                "checkpoint": f"models/v2/clip_lora_fold{fold}_best_v1.pt",
                "n": int(len(ids)),
                "D_r1": 512,
                "path": str(out_npz.relative_to(PROJECT_ROOT)),
            }
            continue

        meta["performed"] = True
        rows = []
        for iid in pf["image_id"]:
            r = man_ix.loc[iid]
            path = resolve_path(r)
            if not path.is_file():
                stop(f"missing calib image {iid}: {path}")
            rows.append({"image_id": iid, "path": str(path)})

        ckpt_path = CKPT_DIR / f"clip_lora_fold{fold}_best_v1.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        cfg.setdefault("clip", {})
        cfg["clip"].setdefault("l2_normalize", True)
        if isinstance(cfg.get("head"), str):
            cfg["head"] = {"dropout": 0.2}
        cfg.setdefault("head", {"dropout": 0.2})

        print(f"[V2-10C] Extracting CALIB R1 fold {fold} on {device} n={len(rows)} ...", flush=True)
        model = ClipLoRAModel(cfg, device)
        incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
        crit = [k for k in incompat.missing_keys if "lora_" in k or k.startswith("head.")]
        if crit:
            stop(f"fold {fold} critical missing keys {crit[:10]}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        ds = PathDataset(rows, model.preprocess)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        r1_list, id_list = [], []
        with torch.no_grad():
            for xb, ids in loader:
                xb = xb.to(device)
                r1 = model.encode(xb)
                r1_list.append(r1.detach().cpu().numpy().astype(np.float32))
                id_list.extend(list(ids))
        r1 = np.concatenate(r1_list, axis=0)
        ids_arr = np.asarray(id_list, dtype=str)
        pos = {i: j for j, i in enumerate(ids_arr)}
        order = [pos[i] for i in pf["image_id"]]
        r1 = r1[order]
        ids_arr = ids_arr[order]
        y = pf["binary_label"].to_numpy(dtype=np.int64)
        if not np.isfinite(r1).all() or r1.shape != (len(pf), 512):
            stop(f"fold {fold} bad calib R1")
        if list(ids_arr) != list(pf["image_id"]):
            stop(f"fold {fold} calib ID order mismatch")

        np.savez_compressed(
            out_npz,
            image_ids=ids_arr,
            r1=r1,
            y=y,
            fold=np.array([fold]),
            layer_r1="lora_visual_embedding_l2_normalized",
            membership="V2_9C_PRIMARY_CALIBRATION",
            checkpoint=np.array([str(ckpt_path.relative_to(PROJECT_ROOT))]),
        )
        meta["folds"][f"fold_{fold}"] = {
            "reused": False,
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "n": int(len(ids_arr)),
            "D_r1": 512,
            "path": str(out_npz.relative_to(PROJECT_ROOT)),
            "finite": True,
        }
        print(f"[V2-10C] Wrote {out_npz.name}", flush=True)

    if any(not v.get("reused") for v in meta["folds"].values()):
        meta["performed"] = True
    return meta


def load_calib_r1(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(OUT / f"v2_10c_calibration_r1_fold{fold}_v1.npz", allow_pickle=False)
    return z["image_ids"].astype(str), z["r1"].astype(np.float32), z["y"].astype(np.int64)


def cutoff_for_coverage(conf: np.ndarray, target: float) -> float:
    """Lowest confidence among top-target fraction (deterministic quantile).

    cutoff = quantile(conf, 1 - target) with numpy default linear interpolation.
    Covered iff conf >= cutoff. Ties: all equal-confidence samples share fate.
    """
    conf = np.asarray(conf, dtype=np.float64)
    q = 1.0 - float(target)
    return float(np.quantile(conf, q, method="linear"))


def selective_metrics(
    y: np.ndarray,
    p: np.ndarray,
    covered: np.ndarray,
    baseline_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    covered = np.asarray(covered).astype(bool)
    pred = (p >= THR).astype(int)
    if baseline_pred is None:
        baseline_pred = pred
    baseline_pred = np.asarray(baseline_pred).astype(int)
    baseline_err = baseline_pred != y

    n = len(y)
    n_cov = int(covered.sum())
    cov_rate = n_cov / n if n else float("nan")
    abstain = 1.0 - cov_rate

    out: dict[str, Any] = {
        "n": n,
        "n_covered": n_cov,
        "coverage": float(cov_rate),
        "abstention_rate": float(abstain),
    }

    if n_cov:
        yc, pc, predc = y[covered], p[covered], pred[covered]
        err_c = predc != yc
        out["selective_accuracy"] = float((predc == yc).mean())
        out["selective_error_rate"] = float(err_c.mean())
        # covered balanced accuracy
        ai_c = yc == 1
        real_c = yc == 0
        ai_rec = float((predc[ai_c] == 1).mean()) if ai_c.any() else float("nan")
        real_spec = float((predc[real_c] == 0).mean()) if real_c.any() else float("nan")
        out["covered_ai_recall"] = ai_rec
        out["covered_real_specificity"] = real_spec
        out["covered_balanced_accuracy"] = float(0.5 * (ai_rec + real_spec))
    else:
        out["selective_accuracy"] = float("nan")
        out["selective_error_rate"] = float("nan")
        out["covered_ai_recall"] = float("nan")
        out["covered_real_specificity"] = float("nan")
        out["covered_balanced_accuracy"] = float("nan")

    # class-wise coverage
    for lab, name in [(1, "ai"), (0, "real")]:
        m = y == lab
        nm = int(m.sum())
        nc = int((m & covered).sum())
        out[f"{name}_n"] = nm
        out[f"{name}_coverage"] = float(nc / nm) if nm else float("nan")
        out[f"{name}_uncertainty_rate"] = float(1.0 - nc / nm) if nm else float("nan")

    # whole-dataset resolved fractions
    # TP / total AI among covered correct AI
    ai = y == 1
    real = y == 0
    tp_cov = int(((pred == 1) & ai & covered).sum())
    tn_cov = int(((pred == 0) & real & covered).sum())
    out["tp_over_total_ai"] = float(tp_cov / ai.sum()) if ai.any() else float("nan")
    out["tn_over_total_real"] = float(tn_cov / real.sum()) if real.any() else float("nan")

    # error capture
    n_err = int(baseline_err.sum())
    err_abstained = int((baseline_err & ~covered).sum())
    correct_abstained = int((~baseline_err & ~covered).sum())
    out["baseline_errors"] = n_err
    out["errors_to_uncertain"] = err_abstained
    out["error_capture_rate"] = float(err_abstained / n_err) if n_err else float("nan")
    out["correct_to_uncertain"] = correct_abstained
    out["error_enrichment_ratio"] = (
        float(out["error_capture_rate"] / abstain) if abstain > 1e-12 else float("nan")
    )

    # AI / Real errors separately
    for lab, name in [(1, "ai"), (0, "real")]:
        e = baseline_err & (y == lab)
        ne = int(e.sum())
        ea = int((e & ~covered).sum())
        out[f"{name}_baseline_errors"] = ne
        out[f"{name}_errors_to_uncertain"] = ea
        out[f"{name}_error_capture_rate"] = float(ea / ne) if ne else float("nan")

    # baseline full error rate for comparison
    out["baseline_error_rate"] = float(baseline_err.mean())
    return out


def aurc(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Area Under Risk-Coverage: rank by calibrated confidence descending.

    coverage_k = k/n; risk_k = error rate among top-k confidence samples.
    AURC = trapezoidal integral of risk vs coverage over k=1..n (plus 0).
    Diagnostic only — not used to choose cutoffs.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    conf = confidence(p)
    pred = (p >= THR).astype(int)
    # deterministic tie-break by image index order already in arrays: stable sort
    order = np.argsort(-conf, kind="mergesort")
    err = (pred != y)[order].astype(float)
    n = len(y)
    if n == 0:
        return {"aurc": float("nan"), "definition": "trapz(risk_vs_coverage)"}
    cum_err = np.cumsum(err)
    ks = np.arange(1, n + 1)
    coverage = ks / n
    risk = cum_err / ks
    # include (0, risk_at_1) or (0,0): standard often starts at first point
    cov = np.concatenate([[0.0], coverage])
    risk0 = np.concatenate([[risk[0]], risk])
    return {
        "aurc": float(np.trapezoid(risk0, cov)),
        "definition": "trapz(selective_error_rate vs coverage); rank by conf=max(p,1-p); mergesort ties",
        "n": int(n),
    }


def decide(payload_bits: dict[str, Any]) -> tuple[str, list[str]]:
    """Predeclared S90 gate — do not alter after seeing results."""
    evidence = []
    s90 = payload_bits["s90_macro"]
    cal_imp = payload_bits["calib_improve_count"]
    class_changes = payload_bits["total_class_changes"]
    fold_ok = payload_bits["folds_benefiting"]
    hard = payload_bits["hard_s90"]
    real = payload_bits["real_s90"]

    phone = next(r for r in real if r["real_domain"] == "Smartphone")
    mllm = next(r for r in real if r["real_domain"] == "MLLM")

    # hard gen: not nearly every sample abstained
    hard_covs = [h["coverage"] for h in hard]
    mean_hard_cov = float(np.mean(hard_covs)) if hard_covs else 0.0
    not_all_abstain = mean_hard_cov >= 0.25 and sum(1 for c in hard_covs if c < 0.05) <= 2

    err_drop = s90["baseline_error_rate"] - s90["selective_error_rate"]
    material_err_drop = err_drop >= 0.02 or (
        s90["selective_error_rate"] <= 0.8 * s90["baseline_error_rate"]
    )

    flags = {
        "leakage_safe": True,
        "temp_improves_ge2": cal_imp >= 2,
        "p05_unchanged": class_changes == 0,
        "eval_coverage_ge_080": s90["coverage"] >= 0.80,
        "selective_error_down": material_err_drop,
        "enrichment_gt_1_5": s90["error_enrichment_ratio"] > 1.5,
        "covered_real_spec_ge_097": s90["covered_real_specificity"] >= 0.97,
        "phone_ge_099": phone["covered_specificity"] >= 0.99,
        "mllm_ge_085": mllm["covered_specificity"] >= 0.85,
        "folds_ge_3": fold_ok >= 3,
        "hard_not_all_abstain": not_all_abstain,
    }
    evidence.append(f"flags={flags}")
    evidence.append(
        f"S90 macro: cov={s90['coverage']:.3f} sel_err={s90['selective_error_rate']:.4f} "
        f"base_err={s90['baseline_error_rate']:.4f} enrich={s90['error_enrichment_ratio']:.3f} "
        f"covRealSpec={s90['covered_real_specificity']:.3f}"
    )
    evidence.append(
        f"Phone covSpec={phone['covered_specificity']:.3f} MLLM covSpec={mllm['covered_specificity']:.3f} "
        f"mean_hard_cov={mean_hard_cov:.3f} folds_ok={fold_ok} temp_metrics_improved={cal_imp}"
    )

    if all(flags.values()):
        return "SELECTIVE_PREDICTION_PROMISING", evidence

    # mixed: some useful capture but incomplete
    useful = (
        flags["enrichment_gt_1_5"]
        or flags["selective_error_down"]
        or (s90["error_capture_rate"] >= 0.25)
    )
    problematic = (
        not flags["eval_coverage_ge_080"]
        or not flags["covered_real_spec_ge_097"]
        or not flags["phone_ge_099"]
        or not flags["mllm_ge_085"]
        or not flags["folds_ge_3"]
        or not flags["hard_not_all_abstain"]
        or not flags["temp_improves_ge2"]
    )
    if useful and problematic:
        return "SELECTIVE_PREDICTION_MIXED", evidence
    if useful and not all(flags.values()):
        return "SELECTIVE_PREDICTION_MIXED", evidence
    return "SELECTIVE_PREDICTION_NOT_HELPFUL", evidence


def make_figures(
    fold_cal: dict[int, dict],
    fold_eval: dict[int, dict],
    risk_cov: dict[str, Any],
) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    created = []

    # reliability calib C0 vs C1 (macro overlay per fold mean)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(
        axes,
        ["p_raw", "p_temp"],
        ["C0 raw H2 (calib)", "C1 temperature (calib)"],
    ):
        for fold in [1, 2, 3, 4]:
            p = fold_cal[fold][key]
            y = fold_cal[fold]["y"]
            conf_m, acc_m = reliability_curve(p, y)
            ax.plot(conf_m, acc_m, marker="o", ms=3, label=f"F{fold}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    pth = FIG / "v2_10c_reliability_calib_c0_c1_v1.png"
    fig.savefig(pth, dpi=140)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))

    # reliability eval
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in zip(
        axes,
        ["p_raw", "p_temp"],
        ["C0 raw H2 (eval)", "C1 temperature (eval)"],
    ):
        for fold in [1, 2, 3, 4]:
            p = fold_eval[fold][key]
            y = fold_eval[fold]["y"]
            conf_m, acc_m = reliability_curve(p, y)
            ax.plot(conf_m, acc_m, marker="o", ms=3, label=f"F{fold}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    pth = FIG / "v2_10c_reliability_eval_c0_c1_v1.png"
    fig.savefig(pth, dpi=140)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))

    # risk-coverage diagnostic
    fig, ax = plt.subplots(figsize=(6, 4))
    for fold in [1, 2, 3, 4]:
        rc = risk_cov["folds"][f"fold_{fold}"]
        ax.plot(rc["coverage"], rc["risk"], label=f"F{fold} AURC={rc['aurc']:.3f}")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective error rate")
    ax.set_title("Risk–coverage (diagnostic; not for cutoff selection)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    pth = FIG / "v2_10c_risk_coverage_v1.png"
    fig.savefig(pth, dpi=140)
    plt.close(fig)
    created.append(str(pth.relative_to(PROJECT_ROOT)))
    return created


def write_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("V2-10C — H2 Confidence Calibration + Selective Prediction")
    lines.append(f"Status: COMPLETE — decision {payload['decision']}")
    lines.append("Temperature-only on frozen H2. Binary thr=0.5 unchanged. S90 primary.")
    lines.append("No Platt. No H2 retrain. FINAL_V2_MODEL NOT SELECTED.")
    lines.append("")
    lines.append("Provenance audit")
    for k, v in payload["provenance"].items():
        lines.append(f"  {k}: {v}")
    inf = payload["calib_feature_inference"]
    lines.append(
        f"NEW_CALIBRATION_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}"
    )
    for k, v in sorted(inf.get("folds", {}).items()):
        lines.append(
            f"  {k}: ckpt={v['checkpoint']} n={v['n']} D={v['D_r1']} "
            f"reused={v.get('reused')} path={v['path']}"
        )
    lines.append("")
    lines.append("Temperature params (fit on calib only)")
    for row in payload["temperature"]:
        lines.append(
            f"  Fold {row['fold']}: T={row['T']:.6f} raw_NLL={row['raw_nll']:.4f} "
            f"temp_NLL={row['temp_nll']:.4f} status={row['status']}"
        )
    lines.append("")
    lines.append(f"p=0.5 classification changes (eval): {payload['class_changes_total']} / {payload['eval_n_total']}")
    lines.append("")
    lines.append("Calibration quality (macro over folds)")
    for split in ["calib", "eval"]:
        lines.append(f"  {split}:")
        for c in ["C0", "C1"]:
            m = payload["cal_metrics_macro"][split][c]
            lines.append(
                f"    {c}: NLL={m['nll']:.4f} Brier={m['brier']:.4f} ECE={m['ece']:.4f} "
                f"AUC={m['auc']:.4f} AP={m['ap']:.4f} AI_rec={m['ai_recall']:.4f} "
                f"RealSpec={m['real_specificity']:.4f}"
            )
    lines.append("")
    lines.append("Selective policies (eval macro)")
    for pol in ["S95", "S90", "S80"]:
        m = payload["selective_macro"][pol]
        lines.append(
            f"  {pol}: cov={m['coverage']:.3f} abstain={m['abstention_rate']:.3f} "
            f"sel_acc={m['selective_accuracy']:.4f} sel_err={m['selective_error_rate']:.4f} "
            f"covBalAcc={m['covered_balanced_accuracy']:.4f} "
            f"err_capture={m['error_capture_rate']:.3f} enrich={m['error_enrichment_ratio']:.3f}"
        )
    lines.append("")
    lines.append("PRIMARY S90 details")
    m = payload["selective_macro"]["S90"]
    lines.append(
        f"  coverage={m['coverage']:.3f} AI_cov={m['ai_coverage']:.3f} Real_cov={m['real_coverage']:.3f}"
    )
    lines.append(
        f"  covered AI recall={m['covered_ai_recall']:.4f} covered RealSpec={m['covered_real_specificity']:.4f}"
    )
    lines.append(
        f"  TP/totalAI={m['tp_over_total_ai']:.4f} TN/totalReal={m['tn_over_total_real']:.4f}"
    )
    lines.append(
        f"  err_capture={m['error_capture_rate']:.3f} enrich={m['error_enrichment_ratio']:.3f} "
        f"(AI capture={m['ai_error_capture_rate']:.3f}, Real capture={m['real_error_capture_rate']:.3f})"
    )
    lines.append("")
    lines.append("S90 per-fold")
    for fm in payload["s90_per_fold"]:
        lines.append(
            f"  Fold {fm['fold']}: cov={fm['coverage']:.3f} sel_acc={fm['selective_accuracy']:.4f} "
            f"covBalAcc={fm['covered_balanced_accuracy']:.4f} "
            f"AI_cov={fm['ai_coverage']:.3f} covAI_rec={fm['covered_ai_recall']:.4f} "
            f"Real_cov={fm['real_coverage']:.3f} covRealSpec={fm['covered_real_specificity']:.4f} "
            f"err_cap={fm['error_capture_rate']:.3f} enrich={fm['error_enrichment_ratio']:.3f}"
        )
    lines.append("")
    lines.append("Hard generators under S90")
    for h in payload["hard_s90"]:
        lines.append(
            f"  {h['generator_id']}: n={h['n']} base_rec={h['baseline_recall']:.3f} "
            f"cov={h['coverage']:.3f} unc={h['uncertainty_rate']:.3f} "
            f"cov_rec={h['covered_recall']:.3f} TP/total={h['tp_over_total']:.3f} "
            f"err_cap={h['errors_captured']:.0f}/{h['baseline_errors']:.0f} "
            f"mode={h['interpretation']}"
        )
    lines.append("")
    lines.append("Real domains under S90")
    for r in payload["real_s90"]:
        lines.append(
            f"  {r['real_domain']}: n={r['n']} cov={r['coverage']:.3f} unc={r['uncertainty_rate']:.3f} "
            f"covSpec={r['covered_specificity']:.3f} covFP={r['covered_fp']} "
            f"FP_to_unc={r['fp_to_uncertain']}"
        )
    if payload.get("fold3_mllm_s90"):
        f3 = payload["fold3_mllm_s90"]
        lines.append(
            f"  Fold3 MLLM: n={f3['n']} cov={f3['coverage']:.3f} covSpec={f3['covered_specificity']:.3f} "
            f"FP_to_unc={f3['fp_to_uncertain']}"
        )
    lines.append("")
    lines.append(f"Risk-coverage AURC (macro mean folds): {payload['risk_coverage']['macro_aurc']:.4f}")
    lines.append("")
    lines.append("Decision evidence")
    for e in payload["decision_evidence"]:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append(f"DECISION: {payload['decision']}")
    lines.append("")
    lines.append("Interpretation")
    for k, v in payload["interpretation"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"Development candidate: {payload['development_candidate']}")
    lines.append(f"Next-stage recommendation ONLY: {payload['next_stage_recommendation']}")
    lines.append("")
    lines.append("Integrity statement")
    for k, v in payload["integrity_statement"].items():
        lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("FINAL_V2_MODEL_SELECTED = NO")
    lines.append("NEXT_STAGE_STARTED = NO")
    return "\n".join(lines) + "\n"


def append_research_log(payload: dict[str, Any]) -> None:
    log_path = PROJECT_ROOT / "paper" / "research_log.md"
    text = log_path.read_text()
    if "## Stage V2-10C" in text:
        print("[V2-10C] research_log already contains V2-10C; not rewriting.", flush=True)
        return
    inf = payload["calib_feature_inference"]
    m = payload["selective_macro"]["S90"]
    cal = payload["cal_metrics_macro"]
    temps = ", ".join(f"F{r['fold']} T={r['T']:.4f}" for r in payload["temperature"])
    folds_line = "; ".join(
        f"F{k.split('_')[1]} n={v['n']} D={v['D_r1']} reused={v.get('reused')}"
        for k, v in sorted(inf.get("folds", {}).items())
    )
    entry = f"""
## Stage V2-10C — H2 Confidence Calibration + Selective Prediction

**Date:** 2026-09-07  
**Status:** **COMPLETE — {payload['decision']}**  
**Mode:** Temperature-only calibration + predeclared selective prediction on frozen H2. No LoRA/CLIP/H2 retrain. No Platt. Binary threshold remains 0.5. Final V2 model NOT SELECTED. Next stage NOT STARTED.

**Motivation:** V2-10B HEAD_RESCUE_PROMISING — H2 is primary V2 development candidate; assess whether calibrated confidence supports REAL / AI / UNCERTAIN without moving the binary boundary.

**Calibration pool:** V2-9C primary (REAL_INTERNAL_HOLDOUT 923 + Fold-F PROMPT_BLOCKED). Overlap train=0, eval=0.  
**NEW_CALIBRATION_FEATURE_INFERENCE_PERFORMED = {'YES' if inf.get('performed') else 'NO'}**  
{folds_line}

**Method:** C0=raw H2; C1=z/T temperature (fit per fold on calib labels only). Selective: confidence=max(p_temp,1-p_temp); cutoffs from calib quantiles for S95/S90/S80; **S90 primary**. Cutoffs never use eval labels.

**Temperature:** {temps}. Eval p=0.5 class changes = {payload['class_changes_total']}.

**Calib macro NLL/Brier/ECE:** C0 {cal['calib']['C0']['nll']:.4f}/{cal['calib']['C0']['brier']:.4f}/{cal['calib']['C0']['ece']:.4f} → C1 {cal['calib']['C1']['nll']:.4f}/{cal['calib']['C1']['brier']:.4f}/{cal['calib']['C1']['ece']:.4f}.

**Eval S90 macro:** coverage={m['coverage']:.3f}; sel_err={m['selective_error_rate']:.4f} (base={m['baseline_error_rate']:.4f}); enrich={m['error_enrichment_ratio']:.3f}; covRealSpec={m['covered_real_specificity']:.3f}; covAI_rec={m['covered_ai_recall']:.4f}.

**Decision:** **{payload['decision']}**  
**Development candidate:** {payload['development_candidate']} (not FINAL).

**Outputs:** `src/run_v2_10c_h2_selective_prediction_v1.py`; `results/v2/v2_10c_*`; figures `figures/v2/v2_10c_*.png`.

**Integrity:** NTIRE=NO; fal=NO; folds unchanged; LoRA/CLIP/H2 not updated; no Platt; no thr change; calib∩train=0; calib∩eval=0; eval labels unused for T/cutoffs; FINAL_V2_MODEL_SELECTED=NO; NEXT_STAGE_STARTED=NO.

**Next (recommendation only):** {payload['next_stage_recommendation']}

---
"""
    log_path.write_text(text.rstrip() + "\n" + entry)
    print("[V2-10C] Appended research_log.md", flush=True)


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("[V2-10C] Provenance audit ...", flush=True)
    man = pd.read_csv(MANIFEST)
    man["image_id"] = man["image_id"].astype(str)
    calib = load_primary_calib(man)

    eval_df = pd.read_csv(EVAL_PRED)
    eval_df["image_id"] = eval_df["image_id"].astype(str)
    for fold, exp in EXPECTED_EVAL.items():
        ef = eval_df[eval_df["fold"] == fold]
        if len(ef) != exp:
            stop(f"eval fold {fold} n={len(ef)} != {exp}")
        if ef["image_id"].duplicated().any():
            stop(f"eval fold {fold} duplicate IDs")
        # R1 exists
        zpath = OUT / f"v2_10a_lora_features_fold{fold}_v1.npz"
        if not zpath.is_file():
            stop(f"missing eval R1 {zpath}")
        z = np.load(zpath, allow_pickle=False)
        if set(z["image_ids"].astype(str)) != set(ef["image_id"]):
            stop(f"fold {fold} eval R1 IDs != V2-10B preds")
        h2p = MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib"
        if not h2p.is_file():
            stop(f"missing H2 model {h2p}")

    provenance = {
        "v2_9c_primary_calib_ok": True,
        "calib_train_overlap_all_folds": 0,
        "calib_eval_overlap_all_folds": 0,
        "eval_matches_v2_10b": True,
        "h2_models_present": True,
        "eval_r1_present": True,
    }
    print("[V2-10C] Provenance OK", flush=True)

    feat_meta = extract_calib_r1(man, calib)

    # Apply H2 to calib + fit T + apply to eval
    temp_rows = []
    fold_cal_store: dict[int, dict] = {}
    fold_eval_store: dict[int, dict] = {}
    all_pred_rows = []
    class_changes_total = 0
    cal_metrics_folds = {"calib": {"C0": [], "C1": []}, "eval": {"C0": [], "C1": []}}
    selective_fold_results: dict[str, list] = {name: [] for name, _ in POLICIES}
    cutoffs: dict[str, dict[int, float]] = {name: {} for name, _ in POLICIES}
    risk_folds: dict[str, Any] = {}

    for fold in [1, 2, 3, 4]:
        print(f"[V2-10C] Fold {fold}: H2 calib scores + temperature ...", flush=True)
        ids, X, y = load_calib_r1(fold)
        pf = calib[calib["fold"] == fold].set_index("image_id")
        # align meta
        y_check = pf.loc[ids, "binary_label"].to_numpy(dtype=int)
        if not np.array_equal(y, y_check):
            stop(f"fold {fold} calib label mismatch")

        h2 = joblib.load(MODELS / f"v2_10b_h2_balanced_logreg_fold{fold}_v1.joblib")
        p_raw = h2.predict_proba(X)[:, list(h2.classes_).index(1)].astype(np.float64)
        z = to_logit(p_raw)

        # fit temperature — may raise SystemExit if NLL worsens
        T, tmeta = fit_temperature(z, y.astype(np.float64))
        p_temp = apply_temperature(z, T)
        # verify p=0.5 unchanged on calib
        ch_cal = int(((p_raw >= THR) != (p_temp >= THR)).sum())
        if ch_cal > 0:
            # numerical edge only allowed if p extremely close to 0.5
            edge = np.isclose(p_raw, 0.5, atol=1e-8) | np.isclose(p_temp, 0.5, atol=1e-8)
            if int(((p_raw >= THR) != (p_temp >= THR) & ~edge).sum()) > 0:
                stop(f"fold {fold} calib temperature changed p=0.5 classes n={ch_cal}")

        temp_rows.append(
            {
                "fold": fold,
                "T": float(T),
                "raw_nll": float(tmeta["raw_nll"]),
                "temp_nll": float(tmeta["calibrated_nll"]),
                "raw_brier": float(tmeta["raw_brier"]),
                "temp_brier": float(tmeta["calibrated_brier"]),
                "raw_ece": float(tmeta["raw_ece15"]),
                "temp_ece": float(tmeta["calibrated_ece15"]),
                "n_calib": int(len(y)),
                "status": "ok",
                "calib_class_changes_at_05": ch_cal,
            }
        )

        fold_cal_store[fold] = {
            "y": y,
            "p_raw": p_raw,
            "p_temp": p_temp,
            "ids": ids,
            "generator": pf.loc[ids, "canonical_generator_id"].astype(str).to_numpy(),
            "real_domain": pf.loc[ids, "real_domain"].fillna("").astype(str).to_numpy(),
            "original_role": pf.loc[ids, "original_role"].astype(str).to_numpy(),
        }

        # calib metrics
        for tag, pp, zz in [("C0", p_raw, z), ("C1", p_temp, z / T)]:
            pred = (pp >= THR).astype(int)
            cal_metrics_folds["calib"][tag].append(
                {
                    "nll": compute_nll(zz, y),
                    "brier": compute_brier(pp, y),
                    "ece": compute_ece15(pp, y),
                    "auc": float(roc_auc_score(y, pp)),
                    "ap": float(average_precision_score(y, pp)),
                    "ai_recall": float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
                    "real_specificity": float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())),
                    "ece_bins": ece_bins(pp, y),
                }
            )

        # cutoffs from calib confidence only
        conf_cal = confidence(p_temp)
        for name, tgt in POLICIES:
            cutoffs[name][fold] = cutoff_for_coverage(conf_cal, tgt)

        # eval: existing H2 probs
        ef = eval_df[eval_df["fold"] == fold].copy()
        if len(ef) != EXPECTED_EVAL[fold]:
            stop(f"fold {fold} eval size")
        y_e = ef["y"].to_numpy(dtype=int)
        p_e_raw = ef["p_H2"].to_numpy(dtype=np.float64)
        z_e = to_logit(p_e_raw)
        p_e_temp = apply_temperature(z_e, T)
        ch = int(((p_e_raw >= THR) != (p_e_temp >= THR)).sum())
        class_changes_total += ch
        if ch > 0:
            stop(f"fold {fold} eval temperature changed {ch} classifications at p=0.5")

        fold_eval_store[fold] = {"y": y_e, "p_raw": p_e_raw, "p_temp": p_e_temp}

        for tag, pp, zz in [("C0", p_e_raw, z_e), ("C1", p_e_temp, z_e / T)]:
            pred = (pp >= THR).astype(int)
            cal_metrics_folds["eval"][tag].append(
                {
                    "nll": compute_nll(zz, y_e),
                    "brier": compute_brier(pp, y_e),
                    "ece": compute_ece15(pp, y_e),
                    "auc": float(roc_auc_score(y_e, pp)),
                    "ap": float(average_precision_score(y_e, pp)),
                    "ai_recall": float(((pred == 1) & (y_e == 1)).sum() / max(1, (y_e == 1).sum())),
                    "real_specificity": float(
                        ((pred == 0) & (y_e == 0)).sum() / max(1, (y_e == 0).sum())
                    ),
                    "ece_bins": ece_bins(pp, y_e),
                }
            )

        conf_e = confidence(p_e_temp)
        baseline_pred = (p_e_raw >= THR).astype(int)  # same as temp at 0.5
        for name, _tgt in POLICIES:
            cov = conf_e >= cutoffs[name][fold]
            sm = selective_metrics(y_e, p_e_temp, cov, baseline_pred=baseline_pred)
            sm["fold"] = fold
            sm["policy"] = name
            sm["cutoff"] = cutoffs[name][fold]
            selective_fold_results[name].append(sm)

        # risk-coverage diagnostic
        # build curve points for figure
        conf = conf_e
        pred = (p_e_temp >= THR).astype(int)
        order = np.argsort(-conf, kind="mergesort")
        err = (pred != y_e)[order].astype(float)
        n = len(y_e)
        cum = np.cumsum(err)
        ks = np.arange(1, n + 1)
        coverage = ks / n
        risk = cum / ks
        a = aurc(y_e, p_e_temp)
        risk_folds[f"fold_{fold}"] = {
            "coverage": coverage.tolist()[:: max(1, n // 200)],  # downsample for JSON
            "risk": risk.tolist()[:: max(1, n // 200)],
            "aurc": a["aurc"],
            "definition": a["definition"],
        }

        # predictions rows
        for i, row in ef.iterrows():
            idx = i  # not used
        for j in range(len(ef)):
            row = ef.iloc[j]
            pe = float(p_e_temp[j])
            conf_j = float(conf_e[j])
            pred_j = int(pe >= THR)
            decisions = {}
            for name, _ in POLICIES:
                covered = bool(conf_j >= cutoffs[name][fold])
                if covered:
                    decisions[name] = "AI" if pred_j == 1 else "REAL"
                else:
                    decisions[name] = "UNCERTAIN"
            all_pred_rows.append(
                {
                    "fold": fold,
                    "image_id": row["image_id"],
                    "y": int(row["y"]),
                    "p_h2_raw": float(p_e_raw[j]),
                    "p_h2_temp": pe,
                    "confidence": conf_j,
                    "pred_05": pred_j,
                    "generator_id": row["generator_id"] if pd.notna(row["generator_id"]) else "",
                    "real_domain": row["real_domain"] if pd.notna(row["real_domain"]) else "",
                    "S95": decisions["S95"],
                    "S90": decisions["S90"],
                    "S80": decisions["S80"],
                }
            )

    if class_changes_total != 0:
        stop(f"unexpected total class changes {class_changes_total}")

    pred_out = pd.DataFrame(all_pred_rows)
    pred_out.to_csv(PRED_OUT, index=False)
    pd.DataFrame(temp_rows).to_csv(TEMP_OUT, index=False)

    def macro_cal(split: str, tag: str) -> dict[str, float]:
        rows = cal_metrics_folds[split][tag]
        keys = ["nll", "brier", "ece", "auc", "ap", "ai_recall", "real_specificity"]
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    cal_metrics_macro = {
        "calib": {"C0": macro_cal("calib", "C0"), "C1": macro_cal("calib", "C1")},
        "eval": {"C0": macro_cal("eval", "C0"), "C1": macro_cal("eval", "C1")},
    }

    # how many of NLL/Brier/ECE improved on calib
    c0, c1 = cal_metrics_macro["calib"]["C0"], cal_metrics_macro["calib"]["C1"]
    cal_imp = sum(
        [
            c1["nll"] < c0["nll"] - 1e-6,
            c1["brier"] < c0["brier"] - 1e-6,
            c1["ece"] < c0["ece"] - 1e-6,
        ]
    )

    def macro_sel(name: str) -> dict[str, float]:
        rows = selective_fold_results[name]
        keys = [
            "coverage",
            "abstention_rate",
            "selective_accuracy",
            "selective_error_rate",
            "covered_balanced_accuracy",
            "covered_ai_recall",
            "covered_real_specificity",
            "ai_coverage",
            "real_coverage",
            "tp_over_total_ai",
            "tn_over_total_real",
            "error_capture_rate",
            "error_enrichment_ratio",
            "baseline_error_rate",
            "ai_error_capture_rate",
            "real_error_capture_rate",
        ]
        # map names
        out = {}
        for k in keys:
            src = k
            if k == "ai_error_capture_rate":
                src = "ai_error_capture_rate"
                vals = [r["ai_error_capture_rate"] for r in rows]
            elif k == "real_error_capture_rate":
                vals = [r["real_error_capture_rate"] for r in rows]
            else:
                vals = [r[k] for r in rows]
            out[k] = float(np.nanmean(vals))
        return out

    selective_macro = {name: macro_sel(name) for name, _ in POLICIES}

    # metrics CSV
    metric_rows = []
    for name, _ in POLICIES:
        for r in selective_fold_results[name]:
            metric_rows.append(r)
        metric_rows.append({"fold": "macro", "policy": name, **selective_macro[name]})
    pd.DataFrame(metric_rows).to_csv(METRICS_OUT, index=False)

    # S90 hard generators
    hard_s90 = []
    for gid in HARD_GENERATORS:
        base_recs, covs, uncs, cov_recs, tp_tot, err_caps, base_errs, means, meds, ns = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for fold in [1, 2, 3, 4]:
            sub = pred_out[(pred_out["fold"] == fold) & (pred_out["generator_id"] == gid)]
            if len(sub) == 0:
                continue
            ns.append(len(sub))
            y = sub["y"].to_numpy()
            p = sub["p_h2_temp"].to_numpy()
            pred = (p >= THR).astype(int)
            base_recs.append(float((pred == 1).mean()))
            covered = (sub["S90"] != "UNCERTAIN").to_numpy()
            covs.append(float(covered.mean()))
            uncs.append(float(1.0 - covered.mean()))
            if covered.any():
                cov_recs.append(float((pred[covered] == 1).mean()))
            else:
                cov_recs.append(float("nan"))
            tp_tot.append(float(((pred == 1) & covered).sum() / len(sub)))
            base_err = pred != y  # all AI so err = pred==0
            # actually y should all be 1
            base_errs.append(int((pred == 0).sum()))
            err_caps.append(int(((pred == 0) & ~covered).sum()))
            means.append(float(sub["confidence"].mean()))
            meds.append(float(sub["confidence"].median()))
        cov_m = float(np.nanmean(covs)) if covs else float("nan")
        cov_rec_m = float(np.nanmean(cov_recs)) if cov_recs else float("nan")
        unc_m = float(np.nanmean(uncs)) if uncs else float("nan")
        # interpretation
        if cov_m < 0.25 and unc_m >= 0.75:
            interp = "mostly_abstains"
        elif cov_rec_m >= 0.7 and cov_m >= 0.3:
            interp = "confidently_detects_many"
        elif cov_m >= 0.3 and cov_rec_m < 0.5:
            interp = "still_confidently_misclassifies"
        else:
            interp = "mixed_detect_and_abstain"
        hard_s90.append(
            {
                "generator_id": gid,
                "n": int(sum(ns)),
                "baseline_recall": float(np.mean(base_recs)) if base_recs else float("nan"),
                "coverage": cov_m,
                "uncertainty_rate": unc_m,
                "covered_recall": cov_rec_m,
                "tp_over_total": float(np.mean(tp_tot)) if tp_tot else float("nan"),
                "baseline_errors": float(np.sum(base_errs)),
                "errors_captured": float(np.sum(err_caps)),
                "mean_confidence": float(np.mean(means)) if means else float("nan"),
                "median_confidence": float(np.mean(meds)) if meds else float("nan"),
                "interpretation": interp,
            }
        )

    # Real domains S90
    real_s90 = []
    for dom in REAL_DOMAINS:
        covs, uncs, specs, fps, fp_unc, ns = [], [], [], [], [], []
        for fold in [1, 2, 3, 4]:
            sub = pred_out[
                (pred_out["fold"] == fold)
                & (pred_out["y"] == 0)
                & (pred_out["real_domain"] == dom)
            ]
            if len(sub) == 0:
                continue
            ns.append(len(sub))
            covered = (sub["S90"] != "UNCERTAIN").to_numpy()
            p = sub["p_h2_temp"].to_numpy()
            pred = (p >= THR).astype(int)
            covs.append(float(covered.mean()))
            uncs.append(float(1.0 - covered.mean()))
            if covered.any():
                specs.append(float((pred[covered] == 0).mean()))
                fps.append(int(((pred == 1) & covered).sum()))
            else:
                specs.append(float("nan"))
                fps.append(0)
            # FP among all (covered or not) that were FP at 0.5, sent to uncertain
            fp_all = pred == 1
            fp_unc.append(int((fp_all & ~covered).sum()))
        real_s90.append(
            {
                "real_domain": dom,
                "n": int(sum(ns)),
                "coverage": float(np.mean(covs)) if covs else float("nan"),
                "uncertainty_rate": float(np.mean(uncs)) if uncs else float("nan"),
                "covered_specificity": float(np.nanmean(specs)) if specs else float("nan"),
                "covered_fp": int(sum(fps)),
                "fp_to_uncertain": int(sum(fp_unc)),
            }
        )

    # Fold3 MLLM
    sub = pred_out[(pred_out["fold"] == 3) & (pred_out["y"] == 0) & (pred_out["real_domain"] == "MLLM")]
    if len(sub):
        covered = (sub["S90"] != "UNCERTAIN").to_numpy()
        p = sub["p_h2_temp"].to_numpy()
        pred = (p >= THR).astype(int)
        fold3_mllm = {
            "n": int(len(sub)),
            "coverage": float(covered.mean()),
            "uncertainty_rate": float(1.0 - covered.mean()),
            "covered_specificity": float((pred[covered] == 0).mean()) if covered.any() else float("nan"),
            "covered_fp": int(((pred == 1) & covered).sum()),
            "fp_to_uncertain": int(((pred == 1) & ~covered).sum()),
        }
    else:
        fold3_mllm = None

    # folds benefiting under S90: enrichment>1.2 and sel_err < baseline
    folds_benefiting = 0
    for r in selective_fold_results["S90"]:
        if (
            r["error_enrichment_ratio"] > 1.2
            and r["selective_error_rate"] < r["baseline_error_rate"] - 1e-6
        ):
            folds_benefiting += 1

    s90_per_fold = []
    for r in selective_fold_results["S90"]:
        s90_per_fold.append(
            {
                "fold": r["fold"],
                "coverage": r["coverage"],
                "selective_accuracy": r["selective_accuracy"],
                "covered_balanced_accuracy": r["covered_balanced_accuracy"],
                "ai_coverage": r["ai_coverage"],
                "covered_ai_recall": r["covered_ai_recall"],
                "real_coverage": r["real_coverage"],
                "covered_real_specificity": r["covered_real_specificity"],
                "error_capture_rate": r["error_capture_rate"],
                "error_enrichment_ratio": r["error_enrichment_ratio"],
                "cutoff": r["cutoff"],
            }
        )

    risk_coverage = {
        "folds": risk_folds,
        "macro_aurc": float(np.mean([risk_folds[f"fold_{f}"]["aurc"] for f in [1, 2, 3, 4]])),
        "note": "Diagnostic only; S90 cutoff not chosen from this curve.",
    }

    figures = make_figures(fold_cal_store, fold_eval_store, risk_coverage)

    decision, evidence = decide(
        {
            "s90_macro": selective_macro["S90"],
            "calib_improve_count": cal_imp,
            "total_class_changes": class_changes_total,
            "folds_benefiting": folds_benefiting,
            "hard_s90": hard_s90,
            "real_s90": real_s90,
        }
    )

    s90 = selective_macro["S90"]
    # remaining mistakes: confident wrong among covered errors vs abstained
    # from pred_out
    base_err = pred_out["pred_05"] != pred_out["y"]
    covered_s90 = pred_out["S90"] != "UNCERTAIN"
    n_conf_wrong = int((base_err & covered_s90).sum())
    n_unc_err = int((base_err & ~covered_s90).sum())
    n_err = int(base_err.sum())

    if decision == "SELECTIVE_PREDICTION_PROMISING":
        dev = "H2 + temperature + S90 selective policy"
        next_rec = (
            "Authorize a locked V2 development packaging / robustness or paper-facing "
            "evaluation plan for H2+T+S90 as DEVELOPMENT candidate — not FINAL selection, "
            "not LoRA retraining. Do not auto-start."
        )
    elif decision == "SELECTIVE_PREDICTION_MIXED":
        dev = "H2 (+ optional temperature); selective policy conditional"
        next_rec = (
            "Keep H2 as primary binary development candidate; treat selective S90 as "
            "optional risk-reduction layer with documented limitations (coverage/hard-gen). "
            "Next: targeted residual-error analysis or robustness — not Platt, not LoRA redo. "
            "Do not auto-start."
        )
    else:
        dev = "H2 (binary) without selective policy emphasis"
        next_rec = (
            "Do not adopt selective prediction as primary V2 story; keep H2 binary operating "
            "point. Next: residual hard-generator analysis or robustness on frozen H2. "
            "Do not auto-start."
        )

    interpretation = {
        "q1_h2_miscalibrated": (
            f"Calib C0 NLL/Brier/ECE={c0['nll']:.4f}/{c0['brier']:.4f}/{c0['ece']:.4f}; "
            f"C1={c1['nll']:.4f}/{c1['brier']:.4f}/{c1['ece']:.4f}; "
            f"metrics improved={cal_imp}/3."
        ),
        "q2_temp_helps": f"Temperature improved {cal_imp}/3 of NLL/Brier/ECE on calib; AUC unchanged by construction.",
        "q3_p05_changes": f"{class_changes_total} classification changes at p=0.5 on eval.",
        "q4_s90_coverage": f"S90 macro eval coverage={s90['coverage']:.3f}.",
        "q5_error_capture": f"S90 error capture rate={s90['error_capture_rate']:.3f}.",
        "q6_enrichment": f"S90 enrichment={s90['error_enrichment_ratio']:.3f} (>1 means preferential error abstention).",
        "q7_covered_ai": (
            f"Covered AI recall={s90['covered_ai_recall']:.4f}; "
            f"TP/totalAI={s90['tp_over_total_ai']:.4f} (not full-dataset recall)."
        ),
        "q8_real_safety": (
            f"Covered RealSpec={s90['covered_real_specificity']:.4f}; "
            f"Phone={next(r for r in real_s90 if r['real_domain']=='Smartphone')['covered_specificity']:.3f}; "
            f"MLLM={next(r for r in real_s90 if r['real_domain']=='MLLM')['covered_specificity']:.3f}."
        ),
        "q9_gpt": next(h for h in hard_s90 if "GPT_Image_2" in h["generator_id"])["interpretation"]
        + f" (cov={next(h for h in hard_s90 if 'GPT_Image_2' in h['generator_id'])['coverage']:.3f})",
        "q10_nano": next(h for h in hard_s90 if "Nano_Banana" in h["generator_id"])["interpretation"]
        + f" (cov={next(h for h in hard_s90 if 'Nano_Banana' in h['generator_id'])['coverage']:.3f})",
        "q11_flux": next(h for h in hard_s90 if "FLUX" in h["generator_id"])["interpretation"]
        + f" (cov={next(h for h in hard_s90 if 'FLUX' in h['generator_id'])['coverage']:.3f})",
        "q12_seedream": next(h for h in hard_s90 if "Seedream" in h["generator_id"])["interpretation"]
        + f" (cov={next(h for h in hard_s90 if 'Seedream' in h['generator_id'])['coverage']:.3f})",
        "q13_mllm_real": (
            f"Fold3 MLLM covSpec={fold3_mllm['covered_specificity']:.3f} cov={fold3_mllm['coverage']:.3f} "
            f"FP_to_unc={fold3_mllm['fp_to_uncertain']}"
            if fold3_mllm
            else "N/A"
        ),
        "q14_remaining_mistakes": (
            f"Of {n_err} baseline errors: {n_unc_err} UNCERTAIN, {n_conf_wrong} still covered (confident wrong)."
        ),
        "q15_dev_candidate": dev,
        "q16_next": next_rec,
    }

    integrity_statement = {
        "NTIRE_ACCESSED": "NO",
        "FAL_USED": "NO",
        "V1_MODIFIED": "NO",
        "V2_3_FOLDS_CHANGED": "NO",
        "LORA_BACKBONE_TRAINED": "NO",
        "LORA_WEIGHTS_UPDATED": "NO",
        "CLIP_WEIGHTS_UPDATED": "NO",
        "H2_RETRAINED": "NO",
        "NEW_HEAD_TRAINED": "NO",
        "CALIBRATION_DATA_OVERLAPS_TRAIN": "NO",
        "CALIBRATION_DATA_OVERLAPS_EVAL": "NO",
        "EVAL_LABELS_USED_FOR_TEMPERATURE_FIT": "NO",
        "EVAL_LABELS_USED_FOR_SELECTIVE_CUTOFF": "NO",
        "PLATT_OR_AFFINE_CALIBRATION_USED": "NO",
        "BINARY_THRESHOLD_CHANGED": "NO",
        "FINAL_V2_MODEL_SELECTED": "NO",
        "NEXT_STAGE_STARTED": "NO",
        "NEW_DATA_ACQUIRED": "NO",
        "NEW_CALIBRATION_FEATURE_INFERENCE_PERFORMED": "YES" if feat_meta.get("performed") else "NO",
    }

    payload = {
        "stage": "V2-10C",
        "status": "COMPLETE",
        "decision": decision,
        "decision_evidence": evidence,
        "provenance": provenance,
        "calib_feature_inference": feat_meta,
        "temperature": temp_rows,
        "cutoffs": {k: {f"fold_{f}": v[f] for f in [1, 2, 3, 4]} for k, v in cutoffs.items()},
        "class_changes_total": class_changes_total,
        "eval_n_total": int(len(pred_out)),
        "cal_metrics_macro": cal_metrics_macro,
        "cal_metrics_folds": {
            split: {
                tag: [{k: v for k, v in r.items() if k != "ece_bins"} for r in rows]
                for tag, rows in tags.items()
            }
            for split, tags in cal_metrics_folds.items()
        },
        "selective_macro": selective_macro,
        "s90_per_fold": s90_per_fold,
        "hard_s90": hard_s90,
        "real_s90": real_s90,
        "fold3_mllm_s90": fold3_mllm,
        "risk_coverage": {
            "macro_aurc": risk_coverage["macro_aurc"],
            "note": risk_coverage["note"],
            "per_fold_aurc": {k: v["aurc"] for k, v in risk_folds.items()},
        },
        "figures": figures,
        "interpretation": interpretation,
        "development_candidate": dev,
        "next_stage_recommendation": next_rec,
        "integrity_statement": integrity_statement,
        "final_v2_model_selected": False,
        "next_stage_started": False,
        "primary_policy": PRIMARY,
        "remaining_mistakes": {
            "baseline_errors": n_err,
            "uncertain": n_unc_err,
            "confident_wrong_covered": n_conf_wrong,
        },
    }

    JSON_OUT.write_text(json.dumps(_clean(payload), indent=2))
    REPORT_OUT.write_text(write_report(payload))
    append_research_log(payload)
    print(REPORT_OUT.read_text())
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
