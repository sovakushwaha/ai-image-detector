"""RQ5 calibration and selective prediction pipeline (Stages 25A–25D).

How to run
----------
    source .venv/bin/activate
    PYTHONPATH=src python src/run_rq5_v1.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, select_device, stop_if
from mobilenet_v3_small_binary_v1 import DEFAULT_WEIGHTS, MobileNetV3SmallBinaryV1
from rq3_augmentations_v1 import IMAGENET_MEAN, IMAGENET_STD
from rq4_frequency_transform_v1 import FrequencyTransformV1, NORM_PATH
from rq4_rgb_frequency_fusion_v1 import RGBFrequencyFusionV1
from rq5_calibration_utils_v1 import (
    AUC_TOL,
    CONDITIONS,
    COVERAGE_TARGETS,
    EXPECTED_KNOWN,
    EXPECTED_TEST_ROWS,
    EXPECTED_UNSEEN,
    EXPECTED_VAL,
    EXPECTED_VAL_ROWS,
    PRIMARY_COVERAGE,
    SPLITS,
    UNSEEN_GENERATORS,
    apply_temperature,
    calibration_metrics,
    compute_aurc,
    confidence,
    derive_gamma_thresholds,
    fit_temperature,
    load_json,
    reliability_curve,
    selective_decisions,
    selective_metrics,
    sigmoid,
    stop_if,
    write_json,
)

BATCH_SIZE = 32
NUM_WORKERS = 0
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

SPLIT_META = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
VAL_MANIFEST = PROJECT_ROOT / "metadata" / "rq3_validation_v1_manifest.csv"

C0_ID = "rq5_C0_rgb_a2"
C1_ID = "rq5_C1_rgb_frequency_fusion"
C0_CKPT = PROJECT_ROOT / "models/mobilenet_resize_jpeg_aug_selected_v1.pt"
C1_CKPT = PROJECT_ROOT / "models/rq4_F2_rgb_frequency_fusion_selected_v1.pt"
C0_FROZEN = PROJECT_ROOT / "results/rq3_A2_frozen_config_v1.json"
C1_FROZEN = PROJECT_ROOT / "results/rq4_F2_frozen_config_v1.json"
C0_TEST = PROJECT_ROOT / "results/rq3_A2_test_predictions_v1.csv"
C1_TEST = PROJECT_ROOT / "results/rq4_F2_test_predictions_v1.csv"

C0_VAL_PRED = PROJECT_ROOT / "results/rq5_C0_validation_predictions_v1.csv"
C1_VAL_PRED = PROJECT_ROOT / "results/rq5_C1_validation_predictions_v1.csv"
RAW_CLEAN_CSV = PROJECT_ROOT / "results/rq5_raw_clean_validation_calibration_v1.csv"
C0_TEMP_JSON = PROJECT_ROOT / "results/rq5_C0_temperature_scaling_v1.json"
C1_TEMP_JSON = PROJECT_ROOT / "results/rq5_C1_temperature_scaling_v1.json"
VAL_TRANSFER_CSV = PROJECT_ROOT / "results/rq5_validation_calibration_transfer_v1.csv"
C0_POLICY_JSON = PROJECT_ROOT / "results/rq5_C0_selective_policy_v1.json"
C1_POLICY_JSON = PROJECT_ROOT / "results/rq5_C1_selective_policy_v1.json"
SEL_VAL_CSV = PROJECT_ROOT / "results/rq5_selective_validation_metrics_v1.csv"
SEL_VAL_TRANSFER_CSV = PROJECT_ROOT / "results/rq5_selective_validation_transfer_v1.csv"
C0_TEST_CAL = PROJECT_ROOT / "results/rq5_C0_test_calibrated_predictions_v1.csv"
C1_TEST_CAL = PROJECT_ROOT / "results/rq5_C1_test_calibrated_predictions_v1.csv"
CAL_TEST_CSV = PROJECT_ROOT / "results/rq5_calibration_test_metrics_v1.csv"
SEL_TEST_CSV = PROJECT_ROOT / "results/rq5_selective_test_metrics_v1.csv"
RC_CSV = PROJECT_ROOT / "results/rq5_risk_coverage_v1.csv"
BOOT_JSON = PROJECT_ROOT / "results/rq5_bootstrap_uncertainty_v1.json"
BOOT_CSV = PROJECT_ROOT / "results/rq5_bootstrap_uncertainty_v1.csv"
REPORT = PROJECT_ROOT / "results/rq5_complete_report_v1.txt"
PAPER_CAL = PROJECT_ROOT / "paper/tables/rq5_calibration_summary.csv"
PAPER_SEL = PROJECT_ROOT / "paper/tables/rq5_selective_prediction_summary.csv"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    checkpoint: Path
    frozen_config: Path
    val_pred: Path
    temp_json: Path
    policy_json: Path
    test_cal: Path


MODELS = {
    "C0": ModelSpec(
        model_id=C0_ID,
        label="C0",
        checkpoint=C0_CKPT,
        frozen_config=C0_FROZEN,
        val_pred=C0_VAL_PRED,
        temp_json=C0_TEMP_JSON,
        policy_json=C0_POLICY_JSON,
        test_cal=C0_TEST_CAL,
    ),
    "C1": ModelSpec(
        model_id=C1_ID,
        label="C1",
        checkpoint=C1_CKPT,
        frozen_config=C1_FROZEN,
        val_pred=C1_VAL_PRED,
        temp_json=C1_TEMP_JSON,
        policy_json=C1_POLICY_JSON,
        test_cal=C1_TEST_CAL,
    ),
}


class PathDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} bad size")
        return self.transform(rgb), torch.tensor(float(row["label"]), dtype=torch.float32), index


class PathFusionDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, freq_transform: FrequencyTransformV1):
        self.rows = rows.reset_index(drop=True)
        self.freq_transform = freq_transform
        self.rgb_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} bad size")
        return self.rgb_tensor(rgb), self.freq_transform(rgb), torch.tensor(float(row["label"]), dtype=torch.float32), index


def build_validation_frames() -> dict[str, pd.DataFrame]:
    meta = pd.read_csv(SPLIT_META)
    manifest = pd.read_csv(VAL_MANIFEST)
    frames: dict[str, pd.DataFrame] = {}
    original = meta[meta["split"] == "validation"][["image_id", "processed_path", "label", "generator"]].copy()
    original = original.rename(columns={"image_id": "source_image_id", "processed_path": "path"})
    original = original.sort_values("source_image_id").reset_index(drop=True)
    stop_if(len(original) != EXPECTED_VAL, f"validation original count {len(original)}")
    frames["original"] = original

    for condition in ["jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]:
        sub = manifest[manifest["condition"] == condition].copy()
        sub = sub.rename(columns={"output_path": "path"})
        sub = sub[["source_image_id", "path", "label", "generator"]].sort_values("source_image_id").reset_index(drop=True)
        stop_if(len(sub) != EXPECTED_VAL, f"{condition} count {len(sub)}")
        stop_if(
            not original["source_image_id"].astype(str).equals(sub["source_image_id"].astype(str)),
            f"{condition} source alignment",
        )
        frames[condition] = sub
    return frames


@torch.no_grad()
def infer_c0(frames: dict[str, pd.DataFrame], device: torch.device) -> pd.DataFrame:
    ckpt = torch.load(C0_CKPT, map_location=device, weights_only=False)
    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    records = []
    for condition in CONDITIONS:
        rows = frames[condition]
        loader = DataLoader(PathDataset(rows, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        logits_arr = np.empty(len(rows), dtype=float)
        for images, _, indices in tqdm(loader, desc=f"C0/{condition}", leave=False):
            batch_logits = model(images.to(device)).detach().cpu().numpy().reshape(-1)
            for i, idx in enumerate(indices.numpy()):
                logits_arr[int(idx)] = float(batch_logits[i])
        for i, row in rows.iterrows():
            logit = float(logits_arr[i])
            records.append(
                {
                    "model": C0_ID,
                    "source_image_id": row["source_image_id"],
                    "condition": condition,
                    "label": int(row["label"]),
                    "generator": row["generator"],
                    "raw_logit": logit,
                    "raw_probability": float(sigmoid(logit)),
                }
            )
    pred = pd.DataFrame(records)
    stop_if(len(pred) != EXPECTED_VAL_ROWS, f"C0 validation rows {len(pred)}")
    for condition in CONDITIONS:
        stop_if((pred["condition"] == condition).sum() != EXPECTED_VAL, f"C0 {condition} count")
    pred.to_csv(C0_VAL_PRED, index=False)
    return pred


@torch.no_grad()
def infer_c1(frames: dict[str, pd.DataFrame], device: torch.device) -> pd.DataFrame:
    freq = FrequencyTransformV1.from_json(NORM_PATH)
    ckpt = torch.load(C1_CKPT, map_location=device, weights_only=False)
    model = RGBFrequencyFusionV1().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    records = []
    for condition in CONDITIONS:
        rows = frames[condition]
        loader = DataLoader(PathFusionDataset(rows, freq), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        logits_arr = np.empty(len(rows), dtype=float)
        for x_rgb, x_freq, _, indices in tqdm(loader, desc=f"C1/{condition}", leave=False):
            batch_logits = model(x_rgb.to(device), x_freq.to(device)).detach().cpu().numpy().reshape(-1)
            for i, idx in enumerate(indices.numpy()):
                logits_arr[int(idx)] = float(batch_logits[i])
        for i, row in rows.iterrows():
            logit = float(logits_arr[i])
            records.append(
                {
                    "model": C1_ID,
                    "source_image_id": row["source_image_id"],
                    "condition": condition,
                    "label": int(row["label"]),
                    "generator": row["generator"],
                    "raw_logit": logit,
                    "raw_probability": float(sigmoid(logit)),
                }
            )
    pred = pd.DataFrame(records)
    stop_if(len(pred) != EXPECTED_VAL_ROWS, f"C1 validation rows {len(pred)}")
    for condition in CONDITIONS:
        stop_if((pred["condition"] == condition).sum() != EXPECTED_VAL, f"C1 {condition} count")
    pred.to_csv(C1_VAL_PRED, index=False)
    return pred


def assert_val_alignment(c0: pd.DataFrame, c1: pd.DataFrame) -> None:
    for condition in CONDITIONS:
        a = c0[c0["condition"] == condition].sort_values("source_image_id").reset_index(drop=True)
        b = c1[c1["condition"] == condition].sort_values("source_image_id").reset_index(drop=True)
        stop_if(len(a) != len(b), f"alignment length {condition}")
        stop_if(not a["source_image_id"].astype(str).equals(b["source_image_id"].astype(str)), f"id align {condition}")
        stop_if(not np.array_equal(a["label"].to_numpy(), b["label"].to_numpy()), f"label align {condition}")
        stop_if(not a["generator"].astype(str).equals(b["generator"].astype(str)), f"generator align {condition}")


def stage_25a(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("STAGE 25A — RAW CALIBRATION AUDIT + VALIDATION PREDICTIONS")
    print("=" * 60)

    frames = build_validation_frames()
    if C0_VAL_PRED.exists() and C1_VAL_PRED.exists():
        print("Loading existing validation predictions.")
        c0 = pd.read_csv(C0_VAL_PRED)
        c1 = pd.read_csv(C1_VAL_PRED)
        stop_if(len(c0) != EXPECTED_VAL_ROWS, f"existing C0 rows {len(c0)}")
        stop_if(len(c1) != EXPECTED_VAL_ROWS, f"existing C1 rows {len(c1)}")
    else:
        c0 = infer_c0(frames, device)
        c1 = infer_c1(frames, device)
    assert_val_alignment(c0, c1)

    raw_rows = []
    for key, pred in [("C0", c0), ("C1", c1)]:
        sub = pred[pred["condition"] == "original"].copy()
        logits = sub["raw_logit"].to_numpy(dtype=float)
        labels = sub["label"].to_numpy(dtype=int)
        probs = sub["raw_probability"].to_numpy(dtype=float)
        m = calibration_metrics(logits, labels, probs)
        raw_rows.append({"model": key, **m})

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(RAW_CLEAN_CSV, index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    for key, pred, color in [("C0", c0, "C0"), ("C1", c1, "C1")]:
        sub = pred[pred["condition"] == "original"]
        centers, accs = reliability_curve(sub["raw_probability"].to_numpy(), sub["label"].to_numpy())
        ax.plot(centers, accs, marker="o", label=key)
    ax.set_xlabel("Mean predicted P(AI)")
    ax.set_ylabel("Empirical AI fraction")
    ax.set_title("RQ5 raw clean validation reliability (ECE-15 bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_raw_clean_validation_reliability_v1.png", dpi=150)
    plt.close(fig)

    print("25A GATE: C0 validation predictions complete: YES")
    print("25A GATE: C1 validation predictions complete: YES")
    print("25A GATE: Clean labels aligned: YES")
    print("25A GATE: Calibration metric definitions locked: YES")
    print("25A GATE: Test data accessed: NO")
    return c0, c1, raw_df


def stage_25b(c0: pd.DataFrame, c1: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("STAGE 25B — TEMPERATURE SCALING + CALIBRATOR FREEZE")
    print("=" * 60)

    temps: dict[str, dict] = {}
    transfer_rows = []

    for key, spec, pred in [("C0", MODELS["C0"], c0), ("C1", MODELS["C1"], c1)]:
        cfg = load_json(spec.frozen_config)
        clean = pred[pred["condition"] == "original"]
        logits = clean["raw_logit"].to_numpy(dtype=float)
        labels = clean["label"].to_numpy(dtype=int)
        temperature, fit = fit_temperature(logits, labels)
        stop_if(abs(fit["raw_auc"] - fit["calibrated_auc"]) > AUC_TOL, f"{key} AUC changed materially")
        stop_if(abs(fit["raw_ap"] - fit["calibrated_ap"]) > AUC_TOL, f"{key} AP changed materially")

        payload = {
            "model": spec.model_id,
            "checkpoint": str(spec.checkpoint.relative_to(PROJECT_ROOT)),
            "fitted_on": "clean_validation_original_only",
            "sample_count": EXPECTED_VAL,
            "method": "scalar_temperature_scaling",
            "objective": "clean_validation_nll",
            "raw_validation_nll": fit["raw_nll"],
            "temperature": temperature,
            "calibrated_validation_nll": fit["calibrated_nll"],
            "raw_brier": fit["raw_brier"],
            "calibrated_brier": fit["calibrated_brier"],
            "raw_ece15": fit["raw_ece15"],
            "calibrated_ece15": fit["calibrated_ece15"],
            "calibration_version": "rq5_v1",
            "total_parameters": cfg.get("total_parameters"),
        }
        write_json(spec.temp_json, payload)
        temps[key] = {"temperature": temperature, **fit}

        for condition in CONDITIONS:
            sub = pred[pred["condition"] == condition]
            z = sub["raw_logit"].to_numpy(dtype=float)
            y = sub["label"].to_numpy(dtype=int)
            raw_m = calibration_metrics(z, y)
            cal_probs = apply_temperature(z, temperature)
            cal_m = calibration_metrics(z / temperature, y, cal_probs)
            transfer_rows.append(
                {
                    "model": key,
                    "condition": condition,
                    "raw_nll": raw_m["nll"],
                    "calibrated_nll": cal_m["nll"],
                    "raw_brier": raw_m["brier"],
                    "calibrated_brier": cal_m["brier"],
                    "raw_ece15": raw_m["ece15"],
                    "calibrated_ece15": cal_m["ece15"],
                    "raw_auc": raw_m["roc_auc"],
                    "calibrated_auc": cal_m["roc_auc"],
                    "raw_ap": raw_m["average_precision"],
                    "calibrated_ap": cal_m["average_precision"],
                }
            )

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(VAL_TRANSFER_CSV, index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, color in [("C0", "tab:blue"), ("C1", "tab:orange")]:
        sub = transfer_df[transfer_df["model"] == key]
        x = np.arange(len(CONDITIONS))
        ax.plot(x, sub["raw_nll"].to_numpy(), marker="o", linestyle="--", label=f"{key} raw")
        ax.plot(x, sub["calibrated_nll"].to_numpy(), marker="o", label=f"{key} calibrated")
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("NLL")
    ax.set_title("Validation NLL by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_validation_nll_by_condition_v1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for key in ["C0", "C1"]:
        sub = transfer_df[transfer_df["model"] == key]
        x = np.arange(len(CONDITIONS))
        ax.plot(x, sub["raw_ece15"].to_numpy(), marker="o", linestyle="--", label=f"{key} raw")
        ax.plot(x, sub["calibrated_ece15"].to_numpy(), marker="o", label=f"{key} calibrated")
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("ECE-15")
    ax.set_title("Validation ECE-15 by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_validation_ece_by_condition_v1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, pred) in zip(axes, [("C0", c0), ("C1", c1)]):
        sub = pred[pred["condition"] == "original"]
        t = temps[key]["temperature"]
        probs = apply_temperature(sub["raw_logit"].to_numpy(), t)
        centers, accs = reliability_curve(probs, sub["label"].to_numpy())
        ax.plot([0, 1], [0, 1], "k--")
        ax.plot(centers, accs, marker="o")
        ax.set_title(f"{key} calibrated clean validation")
        ax.set_xlabel("Mean predicted P(AI)")
        ax.set_ylabel("Empirical AI fraction")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_validation_calibrated_reliability_v1.png", dpi=150)
    plt.close(fig)

    return temps, transfer_df


def stage_25c(c0: pd.DataFrame, c1: pd.DataFrame, temps: dict[str, dict]) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("STAGE 25C — SELECTIVE PREDICTION / UNCERTAIN POLICY")
    print("=" * 60)

    policies: dict[str, dict] = {}
    val_sel_rows = []
    val_transfer_rows = []

    for key, spec, pred in [("C0", MODELS["C0"], c0), ("C1", MODELS["C1"], c1)]:
        t = temps[key]["temperature"]
        clean = pred[pred["condition"] == "original"]
        cal_probs = apply_temperature(clean["raw_logit"].to_numpy(), t)
        thresholds = derive_gamma_thresholds(cal_probs, COVERAGE_TARGETS)
        policy = {
            "model": spec.model_id,
            "temperature": t,
            "probability_boundary": 0.5,
            "target_coverages": COVERAGE_TARGETS,
            "gamma90": thresholds["gamma90"],
            "gamma80": thresholds["gamma80"],
            "gamma70": thresholds["gamma70"],
            "lower90": thresholds["lower_90"],
            "upper90": thresholds["upper_90"],
            "lower80": thresholds["lower_80"],
            "upper80": thresholds["upper_80"],
            "lower70": thresholds["lower_70"],
            "upper70": thresholds["upper_70"],
            "achieved_clean_validation_coverage_90": thresholds["achieved_coverage_90"],
            "achieved_clean_validation_coverage_80": thresholds["achieved_coverage_80"],
            "achieved_clean_validation_coverage_70": thresholds["achieved_coverage_70"],
            "policy_derivation_split": "clean_validation_original_only",
            "selection_method": "descending_confidence_quantile",
            "primary_target": PRIMARY_COVERAGE,
            "policy_version": "rq5_v1",
        }
        write_json(spec.policy_json, policy)
        policies[key] = policy

        for cov_label, gamma in [("100", None), ("90", thresholds["gamma90"]), ("80", thresholds["gamma80"]), ("70", thresholds["gamma70"])]:
            m = selective_metrics(clean["label"].to_numpy(), cal_probs, gamma)
            val_sel_rows.append(
                {
                    "model": key,
                    "condition": "original",
                    "target_coverage": cov_label,
                    "gamma": gamma,
                    **m,
                }
            )

        for condition in CONDITIONS:
            if condition == "original":
                continue
            sub = pred[pred["condition"] == condition]
            cal_p = apply_temperature(sub["raw_logit"].to_numpy(), t)
            for cov_label, gamma in [("100", None), ("90", thresholds["gamma90"]), ("80", thresholds["gamma80"]), ("70", thresholds["gamma70"])]:
                m = selective_metrics(sub["label"].to_numpy(), cal_p, gamma)
                val_transfer_rows.append(
                    {
                        "model": key,
                        "condition": condition,
                        "target_coverage": cov_label,
                        "gamma": gamma,
                        **m,
                    }
                )

    sel_val_df = pd.DataFrame(val_sel_rows)
    sel_val_df.to_csv(SEL_VAL_CSV, index=False)
    sel_transfer_df = pd.DataFrame(val_transfer_rows)
    sel_transfer_df.to_csv(SEL_VAL_TRANSFER_CSV, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, pred in [("C0", c0), ("C1", c1)]:
        t = temps[key]["temperature"]
        clean = pred[pred["condition"] == "original"]
        cal_p = apply_temperature(clean["raw_logit"].to_numpy(), t)
        conf = confidence(cal_p)
        order = np.argsort(-conf, kind="mergesort")
        y = clean["label"].to_numpy()[order]
        p = cal_p[order]
        coverages = []
        risks = []
        for k in range(1, len(y) + 1):
            coverages.append(k / len(y))
            pred_k = (p[:k] >= 0.5).astype(int)
            risks.append(1.0 - float((pred_k == y[:k]).mean()))
        ax.plot(coverages, risks, label=key)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Validation risk-coverage (calibrated confidence)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_validation_risk_coverage_v1.png", dpi=150)
    plt.close(fig)

    print("25C HARD FREEZE GATE: temperatures frozen: YES")
    print("25C HARD FREEZE GATE: selective policies frozen: YES")
    print("25C HARD FREEZE GATE: test data accessed: NO")
    return policies, temps, sel_val_df, sel_transfer_df


def load_test_predictions(path: Path, model_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    stop_if(len(df) != EXPECTED_TEST_ROWS, f"{path.name} rows {len(df)}")
    df = df.rename(columns={"logit": "raw_logit", "probability": "raw_probability"})
    df["model"] = model_id
    return df


def stage_25d(temps: dict[str, dict], policies: dict[str, dict]) -> dict:
    print("\n" + "=" * 60)
    print("STAGE 25D — FROZEN TEST CALIBRATION + SELECTIVE EVALUATION")
    print("=" * 60)

    test_preds = {
        "C0": load_test_predictions(C0_TEST, C0_ID),
        "C1": load_test_predictions(C1_TEST, C1_ID),
    }

    cal_test_rows = []
    sel_test_rows = []
    rc_rows = []
    gen_rows = []

    for key, spec in MODELS.items():
        pred = test_preds[key]
        t = temps[key]["temperature"]
        pred = pred.copy()
        pred["calibrated_probability"] = apply_temperature(pred["raw_logit"].to_numpy(), t)
        pred["temperature"] = t
        pred.to_csv(spec.test_cal, index=False)

        for split in SPLITS:
            for condition in CONDITIONS:
                sub = pred[(pred["split"] == split) & (pred["condition"] == condition)]
                z = sub["raw_logit"].to_numpy(dtype=float)
                y = sub["label"].to_numpy(dtype=int)
                raw_m = calibration_metrics(z, y)
                cal_p = sub["calibrated_probability"].to_numpy(dtype=float)
                cal_m = calibration_metrics(z / t, y, cal_p)
                cal_test_rows.append(
                    {
                        "model": key,
                        "split": split,
                        "condition": condition,
                        "raw_nll": raw_m["nll"],
                        "calibrated_nll": cal_m["nll"],
                        "raw_brier": raw_m["brier"],
                        "calibrated_brier": cal_m["brier"],
                        "raw_ece15": raw_m["ece15"],
                        "calibrated_ece15": cal_m["ece15"],
                        "raw_roc_auc": raw_m["roc_auc"],
                        "calibrated_roc_auc": cal_m["roc_auc"],
                        "raw_ap": raw_m["average_precision"],
                        "calibrated_ap": cal_m["average_precision"],
                        "delta_nll": cal_m["nll"] - raw_m["nll"],
                        "delta_brier": cal_m["brier"] - raw_m["brier"],
                        "delta_ece15": cal_m["ece15"] - raw_m["ece15"],
                    }
                )
                aurc = compute_aurc(y, cal_p)
                rc_rows.append({"model": key, "split": split, "condition": condition, "aurc": aurc})

                pol = policies[key]
                for cov_label, gamma in [
                    ("100", None),
                    ("90", pol["gamma90"]),
                    ("80", pol["gamma80"]),
                    ("70", pol["gamma70"]),
                ]:
                    m = selective_metrics(y, cal_p, gamma)
                    sel_test_rows.append(
                        {
                            "model": key,
                            "split": split,
                            "condition": condition,
                            "target_validation_coverage": cov_label,
                            "gamma": gamma,
                            **m,
                        }
                    )

        unseen_orig = pred[(pred["split"] == "unseen_test") & (pred["condition"] == "original")]
        for generator in UNSEEN_GENERATORS:
            sub = unseen_orig[unseen_orig["generator"] == generator]
            if len(sub) == 0:
                continue
            cal_p = sub["calibrated_probability"].to_numpy(dtype=float)
            pol = policies[key]
            m80 = selective_metrics(sub["label"].to_numpy(), cal_p, pol["gamma80"])
            gen_rows.append(
                {
                    "model": key,
                    "generator": generator,
                    "condition": "original",
                    "mean_calibrated_p_ai": float(cal_p.mean()),
                    "median_calibrated_p_ai": float(np.median(cal_p)),
                    "mean_confidence": float(confidence(cal_p).mean()),
                    "coverage_80_policy": m80["achieved_coverage"],
                    "accepted_accuracy_80": m80["accepted_accuracy"],
                    "abstention_rate_80": m80["abstention_rate"],
                }
            )

    cal_test_df = pd.DataFrame(cal_test_rows)
    cal_test_df.to_csv(CAL_TEST_CSV, index=False)
    sel_test_df = pd.DataFrame(sel_test_rows)
    sel_test_df.to_csv(SEL_TEST_CSV, index=False)
    rc_df = pd.DataFrame(rc_rows)
    rc_df.to_csv(RC_CSV, index=False)

    unseen_cal = cal_test_df[cal_test_df["split"] == "unseen_test"]
    worst = unseen_cal.loc[unseen_cal["raw_ece15"].idxmax(), "condition"]
    print(f"Most calibration-challenging unseen condition (raw ECE): {worst}")

    for key, pred in test_preds.items():
        t = temps[key]["temperature"]
        sub = pred[(pred["split"] == "unseen_test") & (pred["condition"] == "original")]
        raw_p = sub["raw_probability"].to_numpy()
        cal_p = apply_temperature(sub["raw_logit"].to_numpy(), t)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--")
        c_raw, a_raw = reliability_curve(raw_p, sub["label"].to_numpy())
        c_cal, a_cal = reliability_curve(cal_p, sub["label"].to_numpy())
        ax.plot(c_raw, a_raw, marker="o", label="raw")
        ax.plot(c_cal, a_cal, marker="o", label="calibrated")
        ax.set_title(f"{key} unseen original reliability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PROJECT_ROOT / f"figures/rq5_{key}_unseen_original_reliability_v1.png", dpi=150)
        plt.close(fig)

        sub_w = pred[(pred["split"] == "unseen_test") & (pred["condition"] == worst)]
        raw_p = sub_w["raw_probability"].to_numpy()
        cal_p = apply_temperature(sub_w["raw_logit"].to_numpy(), t)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--")
        c_raw, a_raw = reliability_curve(raw_p, sub_w["label"].to_numpy())
        c_cal, a_cal = reliability_curve(cal_p, sub_w["label"].to_numpy())
        ax.plot(c_raw, a_raw, marker="o", label="raw")
        ax.plot(c_cal, a_cal, label="calibrated")
        ax.set_title(f"{key} unseen {worst} reliability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PROJECT_ROOT / f"figures/rq5_{key}_unseen_shift_reliability_v1.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sub = unseen_cal.pivot(index="condition", columns="model", values="raw_ece15")
    sub.plot(kind="bar", ax=ax)
    ax.set_ylabel("ECE-15")
    ax.set_title("Unseen test raw ECE by condition")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_unseen_ece_by_condition_v1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sub = unseen_cal.pivot(index="condition", columns="model", values="raw_nll")
    sub.plot(kind="bar", ax=ax)
    ax.set_ylabel("NLL")
    ax.set_title("Unseen test raw NLL by condition")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_unseen_nll_by_condition_v1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, pred in test_preds.items():
        t = temps[key]["temperature"]
        sub = pred[(pred["split"] == "unseen_test") & (pred["condition"] == "original")]
        cal_p = apply_temperature(sub["raw_logit"].to_numpy(), t)
        conf = confidence(cal_p)
        order = np.argsort(-conf, kind="mergesort")
        y = sub["label"].to_numpy()[order]
        p = cal_p[order]
        coverages, risks = [], []
        for k in range(1, len(y) + 1):
            coverages.append(k / len(y))
            pred_k = (p[:k] >= 0.5).astype(int)
            risks.append(1.0 - float((pred_k == y[:k]).mean()))
        ax.plot(coverages, risks, label=key)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Unseen original test risk-coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_unseen_risk_coverage_original_v1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for ax, key in zip(axes, ["C0", "C1"]):
        pred = test_preds[key]
        t = temps[key]["temperature"]
        for condition in CONDITIONS:
            sub = pred[(pred["split"] == "unseen_test") & (pred["condition"] == condition)]
            cal_p = apply_temperature(sub["raw_logit"].to_numpy(), t)
            conf = confidence(cal_p)
            order = np.argsort(-conf, kind="mergesort")
            y = sub["label"].to_numpy()[order]
            p = cal_p[order]
            risks = []
            coverages = []
            for k in range(1, len(y) + 1):
                coverages.append(k / len(y))
                pred_k = (p[:k] >= 0.5).astype(int)
                risks.append(1.0 - float((pred_k == y[:k]).mean()))
            ax.plot(coverages, risks, label=condition, alpha=0.8)
        ax.set_ylabel("Selective risk")
        ax.set_title(f"{key} unseen test risk-coverage by condition")
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Coverage")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_unseen_risk_coverage_by_condition_v1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = unseen_cal.pivot(index="condition", columns="model", values="delta_nll")
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("Delta NLL (cal - raw)")
    ax.set_title("Temperature scaling effect on unseen NLL")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_temperature_scaling_summary_v1.png", dpi=150)
    plt.close(fig)

    sel80 = sel_test_df[(sel_test_df["split"] == "unseen_test") & (sel_test_df["target_validation_coverage"] == "80")]
    sel100 = sel_test_df[(sel_test_df["split"] == "unseen_test") & (sel_test_df["target_validation_coverage"] == "100")]
    merged = sel80.merge(
        sel100[["model", "condition", "selective_risk"]],
        on=["model", "condition"],
        suffixes=("_80", "_100"),
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CONDITIONS))
    width = 0.35
    for i, key in enumerate(["C0", "C1"]):
        sub100 = sel100[sel100["model"] == key].set_index("condition").loc[CONDITIONS]
        sub80 = sel80[sel80["model"] == key].set_index("condition").loc[CONDITIONS]
        ax.bar(x + (i - 0.5) * width, sub100["selective_risk"], width=width, alpha=0.5, label=f"{key} 100%")
        ax.bar(x + (i - 0.5) * width, sub80["selective_risk"], width=width, alpha=0.9, label=f"{key} 80%")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("Selective risk")
    ax.set_title("Unseen selective risk: 100% vs 80% policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_selective_risk_summary_v1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, pred) in zip(axes, test_preds.items()):
        sub = pred[(pred["split"] == "unseen_test") & (pred["condition"] == "original")]
        cal_p = apply_temperature(sub["raw_logit"].to_numpy(), temps[key]["temperature"])
        ax.hist(cal_p[sub["label"] == 0], bins=30, alpha=0.6, label="Real", density=True)
        ax.hist(cal_p[sub["label"] == 1], bins=30, alpha=0.6, label="AI", density=True)
        ax.set_title(f"{key} unseen original calibrated P(AI)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures/rq5_calibrated_probability_distribution_v1.png", dpi=150)
    plt.close(fig)

    bootstrap = run_bootstrap(test_preds, temps, policies)
    write_report_and_tables(
        temps,
        policies,
        cal_test_df,
        sel_test_df,
        rc_df,
        bootstrap,
        gen_rows,
        worst,
    )

    return {
        "test_preds": test_preds,
        "cal_test_df": cal_test_df,
        "sel_test_df": sel_test_df,
        "rc_df": rc_df,
        "bootstrap": bootstrap,
        "worst_condition": worst,
        "gen_rows": gen_rows,
    }


def run_bootstrap(test_preds: dict[str, pd.DataFrame], temps: dict[str, dict], policies: dict[str, dict]) -> list[dict]:
    print("\nRunning paired bootstrap (5000 replicates, seed 42)...")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    ref = test_preds["C0"][(test_preds["C0"]["split"] == "unseen_test") & (test_preds["C0"]["condition"] == "original")]
    ref = ref.sort_values("source_image_id").reset_index(drop=True)
    ids = ref["source_image_id"].astype(str).tolist()
    y_ref = ref["label"].to_numpy(dtype=int)

    real_ids = [i for i, lab in enumerate(y_ref) if lab == 0]
    ai_ids = [i for i, lab in enumerate(y_ref) if lab == 1]

    def sample_idx() -> np.ndarray:
        sr = rng.choice(real_ids, size=len(real_ids), replace=True)
        sa = rng.choice(ai_ids, size=len(ai_ids), replace=True)
        return np.concatenate([sr, sa])

    for split in SPLITS:
        ref_split = test_preds["C0"][(test_preds["C0"]["split"] == split) & (test_preds["C0"]["condition"] == "original")]
        ref_split = ref_split.sort_values("source_image_id").reset_index(drop=True)
        ids_split = ref_split["source_image_id"].astype(str).tolist()
        y_split = ref_split["label"].to_numpy(dtype=int)
        real_idx = [i for i, lab in enumerate(y_split) if lab == 0]
        ai_idx = [i for i, lab in enumerate(y_split) if lab == 1]

        for condition in CONDITIONS:
            cache = {}
            for key in ["C0", "C1"]:
                sub = test_preds[key][(test_preds[key]["split"] == split) & (test_preds[key]["condition"] == condition)]
                sub = sub.sort_values("source_image_id").reset_index(drop=True)
                stop_if(not sub["source_image_id"].astype(str).equals(pd.Series(ids_split).astype(str)), "bootstrap alignment")
                t = temps[key]["temperature"]
                cache[key] = {
                    "logits": sub["raw_logit"].to_numpy(dtype=float),
                    "labels": sub["label"].to_numpy(dtype=int),
                    "temperature": t,
                    "gamma80": policies[key]["gamma80"],
                }

            for metric_kind in ["delta_nll", "delta_brier", "delta_ece15"]:
                for key in ["C0", "C1"]:
                    diffs = []
                    for _ in range(BOOTSTRAP_N):
                        idx = np.concatenate([rng.choice(real_idx, len(real_idx), replace=True), rng.choice(ai_idx, len(ai_idx), replace=True)])
                        z = cache[key]["logits"][idx]
                        y = cache[key]["labels"][idx]
                        raw = calibration_metrics(z, y)
                        cal_p = apply_temperature(z, cache[key]["temperature"])
                        cal = calibration_metrics(z / cache[key]["temperature"], y, cal_p)
                        if metric_kind == "delta_nll":
                            diffs.append(cal["nll"] - raw["nll"])
                        elif metric_kind == "delta_brier":
                            diffs.append(cal["brier"] - raw["brier"])
                        else:
                            diffs.append(cal["ece15"] - raw["ece15"])
                    diffs = np.asarray(diffs)
                    z_full = cache[key]["logits"]
                    y_full = cache[key]["labels"]
                    raw_full = calibration_metrics(z_full, y_full)
                    cal_p_full = apply_temperature(z_full, cache[key]["temperature"])
                    cal_full = calibration_metrics(z_full / cache[key]["temperature"], y_full, cal_p_full)
                    if metric_kind == "delta_nll":
                        observed = cal_full["nll"] - raw_full["nll"]
                    elif metric_kind == "delta_brier":
                        observed = cal_full["brier"] - raw_full["brier"]
                    else:
                        observed = cal_full["ece15"] - raw_full["ece15"]
                    rows.append(
                        {
                            "analysis_type": "calibration_effect",
                            "model": key,
                            "comparison": key,
                            "split": split,
                            "condition": condition,
                            "metric": metric_kind.replace("delta_", "").upper(),
                            "raw_or_reference": "raw",
                            "calibrated_or_candidate": "calibrated",
                            "observed_difference": float(observed),
                            "bootstrap_mean": float(np.mean(diffs)),
                            "bootstrap_std": float(np.std(diffs, ddof=1)),
                            "ci_low": float(np.percentile(diffs, 2.5)),
                            "ci_high": float(np.percentile(diffs, 97.5)),
                            "includes_zero": bool(np.percentile(diffs, 2.5) <= 0 <= np.percentile(diffs, 97.5)),
                        }
                    )

            for key in ["C0", "C1"]:
                risk_diffs = []
                cov_diffs = []
                z_full = cache[key]["logits"]
                y_full = cache[key]["labels"]
                cal_p_full = apply_temperature(z_full, cache[key]["temperature"])
                m100_full = selective_metrics(y_full, cal_p_full, None)
                m80_full = selective_metrics(y_full, cal_p_full, cache[key]["gamma80"])
                obs_risk_diff = m80_full["selective_risk"] - m100_full["selective_risk"]
                obs_cov = m80_full["achieved_coverage"]
                for _ in range(BOOTSTRAP_N):
                    idx = np.concatenate([rng.choice(real_idx, len(real_idx), replace=True), rng.choice(ai_idx, len(ai_idx), replace=True)])
                    z = cache[key]["logits"][idx]
                    y = cache[key]["labels"][idx]
                    cal_p = apply_temperature(z, cache[key]["temperature"])
                    m100 = selective_metrics(y, cal_p, None)
                    m80 = selective_metrics(y, cal_p, cache[key]["gamma80"])
                    risk_diffs.append(m80["selective_risk"] - m100["selective_risk"])
                    cov_diffs.append(m80["achieved_coverage"])
                risk_diffs = np.asarray(risk_diffs)
                cov_diffs = np.asarray(cov_diffs)
                rows.append(
                    {
                        "analysis_type": "selective_risk",
                        "model": key,
                        "comparison": "risk_80_minus_risk_100",
                        "split": split,
                        "condition": condition,
                        "metric": "selective_risk_difference",
                        "raw_or_reference": "100% policy",
                        "calibrated_or_candidate": "80% policy",
                        "observed_difference": float(obs_risk_diff),
                        "bootstrap_mean": float(np.mean(risk_diffs)),
                        "bootstrap_std": float(np.std(risk_diffs, ddof=1)),
                        "ci_low": float(np.percentile(risk_diffs, 2.5)),
                        "ci_high": float(np.percentile(risk_diffs, 97.5)),
                        "includes_zero": bool(np.percentile(risk_diffs, 2.5) <= 0 <= np.percentile(risk_diffs, 97.5)),
                    }
                )
                rows.append(
                    {
                        "analysis_type": "selective_coverage",
                        "model": key,
                        "comparison": key,
                        "split": split,
                        "condition": condition,
                        "metric": "actual_coverage_80",
                        "raw_or_reference": "validation_target_0.80",
                        "calibrated_or_candidate": "test_actual",
                        "observed_difference": float(obs_cov),
                        "bootstrap_mean": float(np.mean(cov_diffs)),
                        "bootstrap_std": float(np.std(cov_diffs, ddof=1)),
                        "ci_low": float(np.percentile(cov_diffs, 2.5)),
                        "ci_high": float(np.percentile(cov_diffs, 97.5)),
                        "includes_zero": bool(np.percentile(cov_diffs, 2.5) <= 0.80 <= np.percentile(cov_diffs, 97.5)),
                    }
                )

            if split == "unseen_test" and condition in ["original", "blur_sigma2", "screenshot_strong"]:
                diffs = []
                z0 = cache["C0"]["logits"]
                z1 = cache["C1"]["logits"]
                y0 = cache["C0"]["labels"]
                p0_full = apply_temperature(z0, cache["C0"]["temperature"])
                p1_full = apply_temperature(z1, cache["C1"]["temperature"])
                obs_c1_c0 = (
                    selective_metrics(y0, p1_full, cache["C1"]["gamma80"])["selective_risk"]
                    - selective_metrics(y0, p0_full, cache["C0"]["gamma80"])["selective_risk"]
                )
                for _ in range(BOOTSTRAP_N):
                    idx = np.concatenate([rng.choice(real_idx, len(real_idx), replace=True), rng.choice(ai_idx, len(ai_idx), replace=True)])
                    p0 = apply_temperature(cache["C0"]["logits"][idx], cache["C0"]["temperature"])
                    p1 = apply_temperature(cache["C1"]["logits"][idx], cache["C1"]["temperature"])
                    y = cache["C0"]["labels"][idx]
                    r0 = selective_metrics(y, p0, cache["C0"]["gamma80"])["selective_risk"]
                    r1 = selective_metrics(y, p1, cache["C1"]["gamma80"])["selective_risk"]
                    diffs.append(r1 - r0)
                diffs = np.asarray(diffs)
                rows.append(
                    {
                        "analysis_type": "c1_vs_c0_selective",
                        "model": "C1",
                        "comparison": "C1_minus_C0",
                        "split": split,
                        "condition": condition,
                        "metric": "selective_risk_80",
                        "raw_or_reference": "C0",
                        "calibrated_or_candidate": "C1",
                        "observed_difference": float(obs_c1_c0),
                        "bootstrap_mean": float(np.mean(diffs)),
                        "bootstrap_std": float(np.std(diffs, ddof=1)),
                        "ci_low": float(np.percentile(diffs, 2.5)),
                        "ci_high": float(np.percentile(diffs, 97.5)),
                        "includes_zero": bool(np.percentile(diffs, 2.5) <= 0 <= np.percentile(diffs, 97.5)),
                    }
                )

    boot_df = pd.DataFrame(rows)
    boot_df.to_csv(BOOT_CSV, index=False)
    BOOT_JSON.parent.mkdir(parents=True, exist_ok=True)
    BOOT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def write_report_and_tables(temps, policies, cal_test_df, sel_test_df, rc_df, bootstrap, gen_rows, worst):
    PAPER_CAL.parent.mkdir(parents=True, exist_ok=True)
    boot_df = pd.DataFrame(bootstrap)

    cal_summary_rows = []
    for key in ["C0", "C1"]:
        for cond in ["original", "blur_sigma2", "screenshot_strong"]:
            row = cal_test_df[(cal_test_df["model"] == key) & (cal_test_df["split"] == "unseen_test") & (cal_test_df["condition"] == cond)].iloc[0]
            b = boot_df[
                (boot_df["analysis_type"] == "calibration_effect")
                & (boot_df["model"] == key)
                & (boot_df["split"] == "unseen_test")
                & (boot_df["condition"] == cond)
                & (boot_df["metric"] == "NLL")
            ].iloc[0]
            cal_summary_rows.append(
                {
                    "row": f"{key} {cond}",
                    "raw_nll": row["raw_nll"],
                    "calibrated_nll": row["calibrated_nll"],
                    "delta_nll": row["delta_nll"],
                    "delta_nll_ci": f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]",
                    "raw_brier": row["raw_brier"],
                    "calibrated_brier": row["calibrated_brier"],
                    "raw_ece": row["raw_ece15"],
                    "calibrated_ece": row["calibrated_ece15"],
                }
            )
    pd.DataFrame(cal_summary_rows).to_csv(PAPER_CAL, index=False)

    sel_summary_rows = []
    for key in ["C0", "C1"]:
        for cond in ["original", "blur_sigma2", "screenshot_strong"]:
            r100 = sel_test_df[
                (sel_test_df["model"] == key)
                & (sel_test_df["split"] == "unseen_test")
                & (sel_test_df["condition"] == cond)
                & (sel_test_df["target_validation_coverage"] == "100")
            ].iloc[0]
            r80 = sel_test_df[
                (sel_test_df["model"] == key)
                & (sel_test_df["split"] == "unseen_test")
                & (sel_test_df["condition"] == cond)
                & (sel_test_df["target_validation_coverage"] == "80")
            ].iloc[0]
            b = boot_df[
                (boot_df["analysis_type"] == "selective_risk")
                & (boot_df["model"] == key)
                & (boot_df["split"] == "unseen_test")
                & (boot_df["condition"] == cond)
            ].iloc[0]
            sel_summary_rows.append(
                {
                    "row": f"{key} {cond}",
                    "risk_100": r100["selective_risk"],
                    "coverage_80_actual": r80["achieved_coverage"],
                    "risk_80": r80["selective_risk"],
                    "risk_difference": r80["selective_risk"] - r100["selective_risk"],
                    "risk_difference_ci": f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]",
                    "abstention_rate": r80["abstention_rate"],
                }
            )
    pd.DataFrame(sel_summary_rows).to_csv(PAPER_SEL, index=False)

    lines = [
        "RQ5 COMPLETE REPORT v1",
        "=" * 60,
        "",
        "1. RQ5 PURPOSE: calibration and selective prediction for frozen C0/C1.",
        "2. MODELS: C0 rq5_C0_rgb_a2; C1 rq5_C1_rgb_frequency_fusion.",
        "3. SEQUENTIAL-DESIGN CAVEAT: parameters from validation only; test is sequential pilot.",
        "",
        f"6. C0 TEMPERATURE: {temps['C0']['temperature']}",
        f"7. C1 TEMPERATURE: {temps['C1']['temperature']}",
        "",
        f"Most challenging unseen condition (raw ECE): {worst}",
        "",
        "26. RQ5 CONCLUSION: see master summary output.",
        "27. SCIENTIFIC INTEGRITY: no training; no test-derived calibration/policy.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def print_master_summary(raw_df, temps, transfer_df, policies, cal_test_df, sel_test_df, boot_df):
    print("\n" + "=" * 60)
    print("RQ5 — CALIBRATION + SELECTIVE PREDICTION COMPLETE")
    print("=" * 60)

    for key in ["C0", "C1"]:
        r = raw_df[raw_df["model"] == key].iloc[0]
        print(f"\n{key} CLEAN VALIDATION RAW: NLL={r['nll']:.4f} Brier={r['brier']:.4f} ECE={r['ece15']:.4f} AUC={r['roc_auc']:.4f} AP={r['average_precision']:.4f}")

    for key in ["C0", "C1"]:
        t = temps[key]
        print(f"\n{key} TEMPERATURE={t['temperature']:.6f} raw NLL={t['raw_nll']:.4f} cal NLL={t['calibrated_nll']:.4f}")

    print("\nVALIDATION CALIBRATION TRANSFER")
    for key in ["C0", "C1"]:
        print(key)
        sub = transfer_df[transfer_df["model"] == key]
        for _, row in sub.iterrows():
            print(f"  {row['condition']:18s} rawNLL={row['raw_nll']:.4f} calNLL={row['calibrated_nll']:.4f} rawECE={row['raw_ece15']:.4f} calECE={row['calibrated_ece15']:.4f}")

    for key in ["C0", "C1"]:
        p = policies[key]
        print(f"\n{key} POLICY gamma90={p['gamma90']:.4f} gamma80={p['gamma80']:.4f} gamma70={p['gamma70']:.4f}")
        print(f"  REAL if p <= {p['lower80']:.4f}; AI if p >= {p['upper80']:.4f}")

    print("\nUNSEEN TEST CALIBRATION")
    for key in ["C0", "C1"]:
        print(key)
        sub = cal_test_df[(cal_test_df["model"] == key) & (cal_test_df["split"] == "unseen_test")]
        for cond in CONDITIONS:
            row = sub[sub["condition"] == cond].iloc[0]
            print(f"  {cond:18s} rawNLL={row['raw_nll']:.4f} calNLL={row['calibrated_nll']:.4f} rawECE={row['raw_ece15']:.4f} calECE={row['calibrated_ece15']:.4f}")

    print("\nPRIMARY 80% SELECTIVE POLICY — UNSEEN")
    for key in ["C0", "C1"]:
        print(key)
        for cond in CONDITIONS:
            r100 = sel_test_df[
                (sel_test_df["model"] == key)
                & (sel_test_df["split"] == "unseen_test")
                & (sel_test_df["condition"] == cond)
                & (sel_test_df["target_validation_coverage"] == "100")
            ].iloc[0]
            r80 = sel_test_df[
                (sel_test_df["model"] == key)
                & (sel_test_df["split"] == "unseen_test")
                & (sel_test_df["condition"] == cond)
                & (sel_test_df["target_validation_coverage"] == "80")
            ].iloc[0]
            print(
                f"  {cond:18s} cov={r80['achieved_coverage']:.3f} risk100={r100['selective_risk']:.4f} risk80={r80['selective_risk']:.4f}"
            )

    print("\nRQ5 STATUS: COMPLETE")
    print("External independent confirmation: PENDING")
    print("Resource analysis: NOT STARTED")
    print("Final overall model selection: NOT STARTED")


def main() -> None:
    device = select_device()
    c0, c1, raw_df = stage_25a(device)
    temps_dict, transfer_df = stage_25b(c0, c1)
    policies, temps_dict, sel_val_df, _ = stage_25c(c0, c1, temps_dict)
    results = stage_25d(temps_dict, policies)
    boot_df = pd.read_csv(BOOT_CSV)
    print_master_summary(raw_df, temps_dict, transfer_df, policies, results["cal_test_df"], results["sel_test_df"], boot_df)


if __name__ == "__main__":
    main()
