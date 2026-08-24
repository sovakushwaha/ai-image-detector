"""Shared calibration metrics and helpers for RQ5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_VAL = 456
EXPECTED_VAL_ROWS = 2280
EXPECTED_KNOWN = 456
EXPECTED_UNSEEN = 1712
EXPECTED_TEST_ROWS = 10840

CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2", "screenshot_strong"]
SPLITS = ["known_test", "unseen_test"]
UNSEEN_GENERATORS = ["Midjourney", "VQDM", "Wukong"]
COVERAGE_TARGETS = [0.90, 0.80, 0.70]
PRIMARY_COVERAGE = 0.80
ECE_BINS = np.linspace(0.0, 1.0, 16)
AUC_TOL = 1e-4


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    stop_if(temperature <= 0, f"temperature must be > 0, got {temperature}")
    return sigmoid(np.asarray(logits, dtype=float) / temperature)


def compute_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(np.asarray(logits, dtype=np.float64).copy())
    labels_t = torch.tensor(np.asarray(labels, dtype=np.float64).copy())
    return float(F.binary_cross_entropy_with_logits(logits_t, labels_t).item())


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def compute_ece15(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(labels)
    ece = 0.0
    for i in range(15):
        lo, hi = ECE_BINS[i], ECE_BINS[i + 1]
        if i < 14:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(probs[mask].mean())
        bin_acc = float(labels[mask].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
    return float(ece)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def calibration_metrics(logits: np.ndarray, labels: np.ndarray, probs: np.ndarray | None = None) -> dict:
    probs = np.asarray(probs if probs is not None else sigmoid(logits), dtype=float)
    labels = np.asarray(labels, dtype=int)
    return {
        "nll": compute_nll(logits, labels),
        "brier": compute_brier(probs, labels),
        "ece15": compute_ece15(probs, labels),
        "roc_auc": safe_auc(labels, probs),
        "average_precision": safe_ap(labels, probs),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    raw_nll = compute_nll(logits, labels)

    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    labels_t = torch.as_tensor(labels, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([log_t], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = F.binary_cross_entropy_with_logits(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).detach().cpu().item())
    stop_if(temperature <= 0, "fitted temperature <= 0")
    calibrated_nll = compute_nll(logits / temperature, labels)
    stop_if(calibrated_nll > raw_nll + 1e-6, "calibrated NLL worse than raw at T=1 baseline")

    raw_probs = sigmoid(logits)
    cal_probs = apply_temperature(logits, temperature)
    return temperature, {
        "raw_nll": raw_nll,
        "calibrated_nll": calibrated_nll,
        "raw_brier": compute_brier(raw_probs, labels),
        "calibrated_brier": compute_brier(cal_probs, labels),
        "raw_ece15": compute_ece15(raw_probs, labels),
        "calibrated_ece15": compute_ece15(cal_probs, labels),
        "raw_auc": safe_auc(labels, raw_probs),
        "calibrated_auc": safe_auc(labels, cal_probs),
        "raw_ap": safe_ap(labels, raw_probs),
        "calibrated_ap": safe_ap(labels, cal_probs),
    }


def confidence(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    return np.maximum(probs, 1.0 - probs)


def derive_gamma_thresholds(calibrated_probs: np.ndarray, coverage_targets: list[float]) -> dict:
    conf = confidence(calibrated_probs)
    order = np.argsort(-conf, kind="mergesort")
    sorted_conf = conf[order]
    n = len(conf)
    out = {}
    for target in coverage_targets:
        k = int(np.ceil(target * n))
        k = max(1, min(k, n))
        gamma = float(sorted_conf[k - 1])
        achieved = float((conf >= gamma).mean())
        out[f"gamma{int(target * 100)}"] = gamma
        out[f"achieved_coverage_{int(target * 100)}"] = achieved
        out[f"lower_{int(target * 100)}"] = 1.0 - gamma
        out[f"upper_{int(target * 100)}"] = gamma
    return out


def selective_decisions(probs: np.ndarray, gamma: float | None) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if gamma is None:
        return np.where(probs >= 0.5, "AI-GENERATED", "REAL")
    lower = 1.0 - gamma
    decisions = np.full(len(probs), "UNCERTAIN", dtype=object)
    decisions[probs <= lower] = "REAL"
    decisions[probs >= gamma] = "AI-GENERATED"
    return decisions


def selective_metrics(labels: np.ndarray, probs: np.ndarray, gamma: float | None) -> dict:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    decisions = selective_decisions(probs, gamma)
    accepted = decisions != "UNCERTAIN"
    n = len(labels)
    accepted_count = int(accepted.sum())
    abstention_rate = float(1.0 - accepted_count / n) if n else 0.0
    achieved_coverage = float(accepted_count / n) if n else 0.0

    if accepted_count == 0:
        return {
            "achieved_coverage": achieved_coverage,
            "abstention_rate": abstention_rate,
            "accepted_count": accepted_count,
            "accepted_accuracy": float("nan"),
            "accepted_balanced_accuracy": float("nan"),
            "selective_risk": float("nan"),
            "ai_class_coverage": float("nan"),
            "real_class_coverage": float("nan"),
        }

    y_acc = labels[accepted]
    pred_acc = np.where(probs[accepted] >= 0.5, 1, 0)
    acc = float((pred_acc == y_acc).mean())
    real_mask = labels == 0
    ai_mask = labels == 1
    real_cov = float(accepted[real_mask].mean()) if real_mask.any() else float("nan")
    ai_cov = float(accepted[ai_mask].mean()) if ai_mask.any() else float("nan")
    tpr = float((pred_acc[y_acc == 1] == 1).mean()) if (y_acc == 1).any() else float("nan")
    tnr = float((pred_acc[y_acc == 0] == 0).mean()) if (y_acc == 0).any() else float("nan")
    bal_acc = float(np.nanmean([tpr, tnr]))
    return {
        "achieved_coverage": achieved_coverage,
        "abstention_rate": abstention_rate,
        "accepted_count": accepted_count,
        "accepted_accuracy": acc,
        "accepted_balanced_accuracy": bal_acc,
        "selective_risk": float(1.0 - acc),
        "ai_class_coverage": ai_cov,
        "real_class_coverage": real_cov,
    }


def compute_aurc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    conf = confidence(probs)
    order = np.argsort(-conf, kind="mergesort")
    y_sorted = labels[order]
    risks = []
    for k in range(1, len(labels) + 1):
        pred = (probs[order][:k] >= 0.5).astype(int)
        risks.append(1.0 - float((pred == y_sorted[:k]).mean()))
    return float(np.mean(risks))


def reliability_curve(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    accuracies = []
    for i in range(15):
        lo, hi = ECE_BINS[i], ECE_BINS[i + 1]
        if i < 14:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        centers.append(float((lo + hi) / 2.0))
        accuracies.append(float(labels[mask].mean()))
    return np.asarray(centers), np.asarray(accuracies)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
