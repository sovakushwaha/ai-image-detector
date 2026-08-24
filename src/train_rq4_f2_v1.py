"""Train RQ4 F2 RGB+frequency fusion and freeze F1/F2 (Stage 24B).

Why this file exists
--------------------
Phase-1 fusion-head training (branches frozen) then Phase-2 joint fine-tuning.
Selects F2 by RobustValAUC, evaluates ScreenshotStrong once, selects clean-
validation Youden thresholds for F1/F2, and writes frozen configs.
No test access.

How to run
----------
    source .venv/bin/activate
    PYTHONPATH=src python src/train_rq4_f2_v1.py
"""

from __future__ import annotations

import json
import platform
import random
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, load_split_metadata, select_device, stop_if
from rq3_augmentations_v1 import IMAGENET_MEAN, IMAGENET_STD, REGIME_CONFIGS, RobustnessAwarePILTransform
from rq4_frequency_cnn_v1 import FrequencyOnlyCNNV1, count_parameters as count_freq_params
from rq4_frequency_transform_v1 import FrequencyTransformV1, NORM_PATH
from rq4_rgb_frequency_fusion_v1 import RGBFrequencyFusionV1
from train_rq4_f1_v1 import (
    FrequencyEvalDataset,
    SELECTION_CONDITIONS,
    SCREENSHOT_CONDITION,
    build_condition_frames,
    is_better_robust,
    predict_probs,
)

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_TRAIN = 1376
EXPECTED_VAL = 456
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20
WEIGHT_DECAY = 1e-4
PHASE1_LR = 1e-3
PHASE2_RGB_LR = 1e-5
PHASE2_FREQ_LR = 5e-5
PHASE2_HEAD_LR = 1e-4
DEFAULT_THRESHOLD = 0.50
YOUDEN_J_TIE_TOLERANCE = 1e-12

F0_ID = "rq4_F0_rgb_a2_reference"
F1_ID = "rq4_F1_frequency_only_v1"
F2_ID = "rq4_F2_rgb_frequency_fusion_v1"

A2_CKPT = PROJECT_ROOT / "models" / "mobilenet_resize_jpeg_aug_selected_v1.pt"
A2_FROZEN = PROJECT_ROOT / "results" / "rq3_A2_frozen_config_v1.json"
F1_CKPT = PROJECT_ROOT / "models" / "rq4_F1_frequency_only_selected_v1.pt"
RQ3_DEV_SUMMARY = PROJECT_ROOT / "results" / "rq3_development_summary_v1.csv"

SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "rq3_validation_v1_manifest.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

PHASE1_CKPT = MODELS_DIR / "rq4_F2_fusion_phase1_best_v1.pt"
PHASE2_CKPT = MODELS_DIR / "rq4_F2_fusion_phase2_best_v1.pt"
F2_SELECTED = MODELS_DIR / "rq4_F2_rgb_frequency_fusion_selected_v1.pt"
PHASE1_HIST = RESULTS_DIR / "rq4_F2_phase1_history_v1.csv"
PHASE2_HIST = RESULTS_DIR / "rq4_F2_phase2_history_v1.csv"
TRAINING_FIG = FIGURES_DIR / "rq4_F2_training_v1.png"
DEV_SUMMARY = RESULTS_DIR / "rq4_development_summary_v1.csv"
F1_FROZEN = RESULTS_DIR / "rq4_F1_frozen_config_v1.json"
F2_FROZEN = RESULTS_DIR / "rq4_F2_frozen_config_v1.json"
FROZEN_CSV = RESULTS_DIR / "rq4_frozen_models_v1.csv"
F1_VAL_PRED = RESULTS_DIR / "rq4_F1_clean_validation_predictions_v1.csv"
F2_VAL_PRED = RESULTS_DIR / "rq4_F2_clean_validation_predictions_v1.csv"
THRESH_REPORT = RESULTS_DIR / "rq4_threshold_selection_report_v1.txt"
GATE_PATH = RESULTS_DIR / "rq4_24b_gate_v1.json"


class FusionTrainDataset(Dataset):
    """One shared A2 augmentation → RGB ImageNet tensor + frequency spectrum."""

    def __init__(self, rows: pd.DataFrame, freq_transform: FrequencyTransformV1, seed: int):
        self.rows = rows.reset_index(drop=True)
        self.freq_transform = freq_transform
        self.pil_aug = RobustnessAwarePILTransform(REGIME_CONFIGS["A2"], rng=random.Random(seed))
        self.rgb_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row["processed_path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} size {rgb.size}")
        augmented = self.pil_aug(rgb)
        x_rgb = self.rgb_tensor(augmented)
        x_freq = self.freq_transform(augmented)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x_rgb, x_freq, label, index


class FusionEvalDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, freq_transform: FrequencyTransformV1, path_col: str = "path"):
        self.rows = rows.reset_index(drop=True)
        self.freq_transform = freq_transform
        self.path_col = path_col
        self.rgb_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row[self.path_col]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} size {rgb.size}")
        x_rgb = self.rgb_tensor(rgb)
        x_freq = self.freq_transform(rgb)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x_rgb, x_freq, label, index


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefer_phase1_on_tie(a: dict, b: dict) -> dict:
    if is_better_robust(a, b) and not is_better_robust(b, a):
        return a
    if is_better_robust(b, a) and not is_better_robust(a, b):
        return b
    return a if a["phase"] == 1 else b


@torch.no_grad()
def predict_fusion(
    model: RGBFrequencyFusionV1, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()
    for x_rgb, x_freq, labels, _ in loader:
        x_rgb = x_rgb.to(device)
        x_freq = x_freq.to(device)
        labels = labels.to(device)
        logits = model(x_rgb, x_freq)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    return labels_np, probs, total_loss / total_samples


@torch.no_grad()
def evaluate_fusion_robust(
    model: RGBFrequencyFusionV1,
    condition_frames: dict[str, pd.DataFrame],
    freq_transform: FrequencyTransformV1,
    device: torch.device,
) -> dict:
    metrics: dict[str, dict[str, float]] = {}
    for condition in SELECTION_CONDITIONS:
        loader = DataLoader(
            FusionEvalDataset(condition_frames[condition], freq_transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        labels, probs, loss = predict_fusion(model, loader, device)
        metrics[condition] = {
            "roc_auc": float(roc_auc_score(labels, probs)),
            "ap": float(average_precision_score(labels, probs)),
            "loss": float(loss),
        }
    robust_val_auc = float(np.mean([metrics[c]["roc_auc"] for c in SELECTION_CONDITIONS]))
    robust_val_ap = float(np.mean([metrics[c]["ap"] for c in SELECTION_CONDITIONS]))
    return {
        "conditions": metrics,
        "robust_val_auc": robust_val_auc,
        "robust_val_ap": robust_val_ap,
        "original_auc": metrics["original"]["roc_auc"],
        "original_ap": metrics["original"]["ap"],
        "original_loss": metrics["original"]["loss"],
    }


def snapshot_bn(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snaps = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            snaps.append((module.running_mean.detach().cpu().clone(), module.running_var.detach().cpu().clone()))
    return snaps


def bn_unchanged(model: nn.Module, before: list[tuple[torch.Tensor, torch.Tensor]]) -> bool:
    after = snapshot_bn(model)
    if len(before) != len(after):
        return False
    for (mb, vb), (ma, va) in zip(before, after):
        if not torch.equal(mb, ma) or not torch.equal(vb, va):
            return False
    return True


def train_phase1_epoch(
    model: RGBFrequencyFusionV1,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.set_branches_eval()
    bn_before = snapshot_bn(model)
    frozen = [p for p in list(model.rgb_branch.parameters()) + list(model.freq_branch.parameters())]
    frozen_before = [p.detach().cpu().clone() for p in frozen]
    total_loss = 0.0
    total_samples = 0
    for x_rgb, x_freq, labels, _ in loader:
        x_rgb = x_rgb.to(device)
        x_freq = x_freq.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(x_rgb, x_freq)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    stop_if(not bn_unchanged(model, bn_before), "Phase1: BN stats updated")
    for p, before in zip(frozen, frozen_before):
        stop_if(not torch.equal(p.detach().cpu(), before), "Phase1: frozen branch updated")
    return total_loss / total_samples


def train_phase2_epoch(
    model: RGBFrequencyFusionV1,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_rgb, x_freq, labels, _ in loader:
        x_rgb = x_rgb.to(device)
        x_freq = x_freq.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(x_rgb, x_freq)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def history_row(phase: int, epoch: int, train_loss: float, val: dict) -> dict:
    return {
        "phase": phase,
        "epoch": epoch,
        "train_loss": train_loss,
        "original_auc": val["conditions"]["original"]["roc_auc"],
        "original_ap": val["conditions"]["original"]["ap"],
        "jpeg50_auc": val["conditions"]["jpeg_q50"]["roc_auc"],
        "jpeg50_ap": val["conditions"]["jpeg_q50"]["ap"],
        "resize112_auc": val["conditions"]["resize_112"]["roc_auc"],
        "resize112_ap": val["conditions"]["resize_112"]["ap"],
        "blur2_auc": val["conditions"]["blur_sigma2"]["roc_auc"],
        "blur2_ap": val["conditions"]["blur_sigma2"]["ap"],
        "robust_val_auc": val["robust_val_auc"],
        "robust_val_ap": val["robust_val_ap"],
        "original_loss": val["original_loss"],
    }


def save_f2_checkpoint(
    path: Path,
    model: RGBFrequencyFusionV1,
    optimizer: torch.optim.Optimizer | None,
    phase: int,
    epoch: int,
    val: dict,
    param_counts: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": F2_ID,
        "phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "validation_metrics": val,
        "robust_val_auc": val["robust_val_auc"],
        "robust_val_ap": val["robust_val_ap"],
        "condition_aucs": {c: val["conditions"][c]["roc_auc"] for c in SELECTION_CONDITIONS},
        "condition_aps": {c: val["conditions"][c]["ap"] for c in SELECTION_CONDITIONS},
        "rgb_source_checkpoint": str(A2_CKPT.relative_to(PROJECT_ROOT)),
        "freq_source_checkpoint": str(F1_CKPT.relative_to(PROJECT_ROOT)),
        "frequency_normalization": str(NORM_PATH.relative_to(PROJECT_ROOT)),
        "seed": RANDOM_SEED,
        "parameter_counts": param_counts,
        "primary_rq4_intervention": True,
        "representation": "controlled_v1",
        "split_protocol": "generator_protocol_v1",
    }
    torch.save(payload, path)


def confusion_parts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_parts(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(fpr),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def select_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # sklearn may return thresholds with first value = inf for perfect scores; drop non-finite
    mask = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[mask], tpr[mask], thresholds[mask]
    youden = tpr - fpr
    best_j = float(np.max(youden))
    candidates = np.where(np.abs(youden - best_j) <= YOUDEN_J_TIE_TOLERANCE)[0]
    best_idx = candidates[0]
    best_thr = float(thresholds[best_idx])
    for idx in candidates[1:]:
        thr = float(thresholds[idx])
        if abs(thr - DEFAULT_THRESHOLD) < abs(best_thr - DEFAULT_THRESHOLD) - 1e-15:
            best_thr = thr
            best_idx = idx
        elif abs(abs(thr - DEFAULT_THRESHOLD) - abs(best_thr - DEFAULT_THRESHOLD)) <= 1e-15 and thr < best_thr:
            best_thr = thr
            best_idx = idx
    return best_thr, best_j


def plot_f2_training(p1: pd.DataFrame, p2: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(p1["epoch"], p1["train_loss"], "o-", label="Phase1")
    axes[0].plot(p2["epoch"], p2["train_loss"], "s--", label="Phase2")
    axes[0].set_xlabel("Epoch within phase")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("F2 training loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(p1["epoch"], p1["robust_val_auc"], "o-", label="Phase1 RobustValAUC")
    axes[1].plot(p2["epoch"], p2["robust_val_auc"], "s--", label="Phase2 RobustValAUC")
    axes[1].set_xlabel("Epoch within phase")
    axes[1].set_ylabel("RobustValAUC")
    axes[1].set_title("F2 robust validation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    TRAINING_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAINING_FIG, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=== Stage 24B — F2 fusion development + freeze ===")
    stop_if(not NORM_PATH.exists(), "missing frequency norm")
    stop_if(not F1_CKPT.exists(), "missing F1 checkpoint — complete 24A first")
    stop_if(not A2_CKPT.exists(), "missing A2 checkpoint")
    set_seed(RANDOM_SEED)
    device = select_device()
    print(f"Device: {device}")

    freq_transform = FrequencyTransformV1.from_json(NORM_PATH)
    train_meta = load_split_metadata("train")
    val_meta = load_split_metadata("validation")
    stop_if(len(train_meta) != EXPECTED_TRAIN, "train count")
    stop_if(len(val_meta) != EXPECTED_VAL, "val count")
    manifest = pd.read_csv(MANIFEST_PATH)
    condition_frames = build_condition_frames(val_meta, manifest)

    train_loader = DataLoader(
        FusionTrainDataset(train_meta, freq_transform, seed=RANDOM_SEED),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    model = RGBFrequencyFusionV1().to(device)
    model.load_rgb_from_a2(A2_CKPT)
    model.load_freq_from_f1(F1_CKPT)
    param_counts = model.count_component_params()
    print(f"F2 params: {param_counts}")

    criterion = nn.BCEWithLogitsLoss()

    # ---- Phase 1 ----
    model.freeze_branches()
    optimizer = torch.optim.AdamW(model.fusion_head.parameters(), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    p1_rows: list[dict] = []
    best_p1: dict | None = None
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_phase1_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate_fusion_robust(model, condition_frames, freq_transform, device)
        p1_rows.append(history_row(1, epoch, train_loss, val))
        print(f"P1 {epoch}/{PHASE1_EPOCHS} loss={train_loss:.4f} RobustValAUC={val['robust_val_auc']:.4f}")
        cand = {
            "phase": 1,
            "epoch": epoch,
            "robust_val_auc": val["robust_val_auc"],
            "robust_val_ap": val["robust_val_ap"],
            "original_auc": val["original_auc"],
            "full_val": val,
        }
        if is_better_robust(cand, best_p1):
            best_p1 = cand
            save_f2_checkpoint(PHASE1_CKPT, model, optimizer, 1, epoch, val, param_counts)
            print(f"  -> Phase1 best @ epoch {epoch}")

    stop_if(best_p1 is None, "no phase1 checkpoint")
    p1_df = pd.DataFrame(p1_rows)
    p1_df.to_csv(PHASE1_HIST, index=False)

    # Load best phase1 for phase2 start
    ckpt1 = torch.load(PHASE1_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt1["model_state_dict"])
    model.unfreeze_all()
    optimizer2 = torch.optim.AdamW(
        model.parameter_groups(PHASE2_RGB_LR, PHASE2_FREQ_LR, PHASE2_HEAD_LR, WEIGHT_DECAY)
    )

    p2_rows: list[dict] = []
    best_p2: dict | None = None
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_phase2_epoch(model, train_loader, criterion, optimizer2, device)
        val = evaluate_fusion_robust(model, condition_frames, freq_transform, device)
        p2_rows.append(history_row(2, epoch, train_loss, val))
        print(f"P2 {epoch}/{PHASE2_EPOCHS} loss={train_loss:.4f} RobustValAUC={val['robust_val_auc']:.4f}")
        cand = {
            "phase": 2,
            "epoch": epoch,
            "robust_val_auc": val["robust_val_auc"],
            "robust_val_ap": val["robust_val_ap"],
            "original_auc": val["original_auc"],
            "full_val": val,
        }
        if is_better_robust(cand, best_p2):
            best_p2 = cand
            save_f2_checkpoint(PHASE2_CKPT, model, optimizer2, 2, epoch, val, param_counts)
            print(f"  -> Phase2 best @ epoch {epoch}")

    stop_if(best_p2 is None, "no phase2 checkpoint")
    p2_df = pd.DataFrame(p2_rows)
    p2_df.to_csv(PHASE2_HIST, index=False)
    plot_f2_training(p1_df, p2_df)

    selected = prefer_phase1_on_tie(best_p1, best_p2)
    src_ckpt = PHASE1_CKPT if selected["phase"] == 1 else PHASE2_CKPT
    ckpt_sel = torch.load(src_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_sel["model_state_dict"])
    save_f2_checkpoint(
        F2_SELECTED,
        model,
        None,
        selected["phase"],
        selected["epoch"],
        selected["full_val"],
        param_counts,
    )
    # annotate final selection
    final = torch.load(F2_SELECTED, map_location="cpu", weights_only=False)
    final["final_selection"] = {"phase": selected["phase"], "epoch": selected["epoch"]}
    torch.save(final, F2_SELECTED)
    print(f"Selected F2: Phase {selected['phase']} epoch {selected['epoch']} RobustValAUC={selected['robust_val_auc']:.6f}")

    # Screenshot once
    screen_loader = DataLoader(
        FusionEvalDataset(condition_frames[SCREENSHOT_CONDITION], freq_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    labels_s, probs_s, _ = predict_fusion(model, screen_loader, device)
    f2_screenshot = {
        "roc_auc": float(roc_auc_score(labels_s, probs_s)),
        "ap": float(average_precision_score(labels_s, probs_s)),
    }
    final = torch.load(F2_SELECTED, map_location="cpu", weights_only=False)
    final["screenshot_strong_auc"] = f2_screenshot["roc_auc"]
    final["screenshot_strong_ap"] = f2_screenshot["ap"]
    torch.save(final, F2_SELECTED)

    # Load F1 screenshot from checkpoint
    f1_ckpt = torch.load(F1_CKPT, map_location="cpu", weights_only=False)
    f1_screenshot = {
        "roc_auc": float(f1_ckpt.get("screenshot_strong_auc", float("nan"))),
        "ap": float(f1_ckpt.get("screenshot_strong_ap", float("nan"))),
    }

    # Development summary
    rq3 = pd.read_csv(RQ3_DEV_SUMMARY)
    a2 = rq3[rq3["regime"] == "A2"].iloc[0]
    with open(A2_FROZEN) as f:
        a2_frozen = json.load(f)

    f1_val = f1_ckpt["validation_metrics"]
    f2_val = selected["full_val"]

    summary_rows = [
        {
            "regime": "F0",
            "label": "RGB A2",
            "parameters": int(a2_frozen["total_parameters"]),
            "original_val_auc": float(a2["original_auc"]),
            "jpeg50_val_auc": float(a2["jpeg50_auc"]),
            "resize112_val_auc": float(a2["resize112_auc"]),
            "blur2_val_auc": float(a2["blur2_auc"]),
            "RobustValAUC": float(a2["robust_val_auc"]),
            "RobustValAP": float(a2["robust_val_ap"]),
            "screenshotStrong_val_auc": float(a2["screenshot_strong_auc"]),
            "screenshotStrong_val_ap": float(a2["screenshot_strong_ap"]),
            "selected_phase": a2["selected_phase"],
            "selected_epoch": a2["selected_epoch"],
        },
        {
            "regime": "F1",
            "label": "Frequency-only",
            "parameters": int(f1_ckpt["total_parameters"]),
            "original_val_auc": float(f1_val["conditions"]["original"]["roc_auc"]),
            "jpeg50_val_auc": float(f1_val["conditions"]["jpeg_q50"]["roc_auc"]),
            "resize112_val_auc": float(f1_val["conditions"]["resize_112"]["roc_auc"]),
            "blur2_val_auc": float(f1_val["conditions"]["blur_sigma2"]["roc_auc"]),
            "RobustValAUC": float(f1_val["robust_val_auc"]),
            "RobustValAP": float(f1_val["robust_val_ap"]),
            "screenshotStrong_val_auc": f1_screenshot["roc_auc"],
            "screenshotStrong_val_ap": f1_screenshot["ap"],
            "selected_phase": "single",
            "selected_epoch": int(f1_ckpt["epoch"]),
        },
        {
            "regime": "F2",
            "label": "RGB+Frequency Fusion",
            "parameters": int(param_counts["total"]),
            "original_val_auc": float(f2_val["conditions"]["original"]["roc_auc"]),
            "jpeg50_val_auc": float(f2_val["conditions"]["jpeg_q50"]["roc_auc"]),
            "resize112_val_auc": float(f2_val["conditions"]["resize_112"]["roc_auc"]),
            "blur2_val_auc": float(f2_val["conditions"]["blur_sigma2"]["roc_auc"]),
            "RobustValAUC": float(f2_val["robust_val_auc"]),
            "RobustValAP": float(f2_val["robust_val_ap"]),
            "screenshotStrong_val_auc": f2_screenshot["roc_auc"],
            "screenshotStrong_val_ap": f2_screenshot["ap"],
            "selected_phase": selected["phase"],
            "selected_epoch": selected["epoch"],
        },
    ]
    pd.DataFrame(summary_rows).to_csv(DEV_SUMMARY, index=False)

    # ---- Threshold selection (clean original validation only) ----
    print("Selecting Youden thresholds on clean validation...")
    # F1
    f1_model = FrequencyOnlyCNNV1().to(device)
    f1_model.load_state_dict(f1_ckpt["model_state_dict"])
    f1_loader = DataLoader(
        FrequencyEvalDataset(condition_frames["original"], freq_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    y1, p1, _ = predict_probs(f1_model, f1_loader, device)
    stop_if(len(y1) != EXPECTED_VAL, "F1 val count")
    f1_auc = float(roc_auc_score(y1, p1))
    f1_ap = float(average_precision_score(y1, p1))
    f1_thr, f1_j = select_youden_threshold(y1, p1)
    f1_default = threshold_metrics(y1, p1, DEFAULT_THRESHOLD)
    f1_youden = threshold_metrics(y1, p1, f1_thr)
    pd.DataFrame(
        {
            "source_image_id": condition_frames["original"]["source_image_id"].values,
            "label": y1,
            "probability": p1,
            "generator": condition_frames["original"]["generator"].values,
        }
    ).to_csv(F1_VAL_PRED, index=False)

    # F2
    f2_loader = DataLoader(
        FusionEvalDataset(condition_frames["original"], freq_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    y2, p2probs, _ = predict_fusion(model, f2_loader, device)
    stop_if(len(y2) != EXPECTED_VAL, "F2 val count")
    f2_auc = float(roc_auc_score(y2, p2probs))
    f2_ap = float(average_precision_score(y2, p2probs))
    f2_thr, f2_j = select_youden_threshold(y2, p2probs)
    f2_default = threshold_metrics(y2, p2probs, DEFAULT_THRESHOLD)
    f2_youden = threshold_metrics(y2, p2probs, f2_thr)
    pd.DataFrame(
        {
            "source_image_id": condition_frames["original"]["source_image_id"].values,
            "label": y2,
            "probability": p2probs,
            "generator": condition_frames["original"]["generator"].values,
        }
    ).to_csv(F2_VAL_PRED, index=False)

    f1_frozen = {
        "model_id": F1_ID,
        "architecture": "FrequencyOnlyCNNV1",
        "frequency_representation": "FrequencyTransformV1 log1p |FFT| luminance",
        "normalization_stats_file": str(NORM_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint": str(F1_CKPT.relative_to(PROJECT_ROOT)),
        "selected_epoch": int(f1_ckpt["epoch"]),
        "seed": RANDOM_SEED,
        "training_augmentation": "RQ3 A2 Resize+JPEG then FrequencyTransformV1",
        "validation_metrics": {
            "original_auc": float(f1_val["conditions"]["original"]["roc_auc"]),
            "jpeg50_auc": float(f1_val["conditions"]["jpeg_q50"]["roc_auc"]),
            "resize112_auc": float(f1_val["conditions"]["resize_112"]["roc_auc"]),
            "blur2_auc": float(f1_val["conditions"]["blur_sigma2"]["roc_auc"]),
            "screenshot_strong_auc": f1_screenshot["roc_auc"],
            "screenshot_strong_ap": f1_screenshot["ap"],
        },
        "RobustValAUC": float(f1_val["robust_val_auc"]),
        "RobustValAP": float(f1_val["robust_val_ap"]),
        "threshold_method": "clean_validation_youden_j",
        "threshold": float(f1_thr),
        "validation_youden_j": float(f1_j),
        "clean_validation_auc": f1_auc,
        "clean_validation_ap": f1_ap,
        "default_threshold_metrics": f1_default,
        "youden_threshold_metrics": f1_youden,
        "total_parameters": int(f1_ckpt["total_parameters"]),
        "primary_rq4_intervention": False,
        "known_test_accessed": False,
        "unseen_test_accessed": False,
    }
    with open(F1_FROZEN, "w") as f:
        json.dump(f1_frozen, f, indent=2)
        f.write("\n")

    f2_frozen = {
        "model_id": F2_ID,
        "architecture": "RGBFrequencyFusionV1",
        "rgb_source_checkpoint": str(A2_CKPT.relative_to(PROJECT_ROOT)),
        "frequency_source_checkpoint": str(F1_CKPT.relative_to(PROJECT_ROOT)),
        "frequency_representation": "FrequencyTransformV1 log1p |FFT| luminance",
        "normalization_stats_file": str(NORM_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint": str(F2_SELECTED.relative_to(PROJECT_ROOT)),
        "selected_phase": int(selected["phase"]),
        "selected_epoch": int(selected["epoch"]),
        "seed": RANDOM_SEED,
        "training_augmentation": "shared RQ3 A2 Resize+JPEG for RGB and frequency branches",
        "validation_metrics": {
            "original_auc": float(f2_val["conditions"]["original"]["roc_auc"]),
            "jpeg50_auc": float(f2_val["conditions"]["jpeg_q50"]["roc_auc"]),
            "resize112_auc": float(f2_val["conditions"]["resize_112"]["roc_auc"]),
            "blur2_auc": float(f2_val["conditions"]["blur_sigma2"]["roc_auc"]),
            "screenshot_strong_auc": f2_screenshot["roc_auc"],
            "screenshot_strong_ap": f2_screenshot["ap"],
        },
        "RobustValAUC": float(f2_val["robust_val_auc"]),
        "RobustValAP": float(f2_val["robust_val_ap"]),
        "threshold_method": "clean_validation_youden_j",
        "threshold": float(f2_thr),
        "validation_youden_j": float(f2_j),
        "clean_validation_auc": f2_auc,
        "clean_validation_ap": f2_ap,
        "default_threshold_metrics": f2_default,
        "youden_threshold_metrics": f2_youden,
        "parameter_counts": param_counts,
        "total_parameters": int(param_counts["total"]),
        "primary_rq4_intervention": True,
        "known_test_accessed": False,
        "unseen_test_accessed": False,
    }
    with open(F2_FROZEN, "w") as f:
        json.dump(f2_frozen, f, indent=2)
        f.write("\n")

    frozen_rows = [
        {
            "regime": "F0",
            "model_id": F0_ID,
            "checkpoint": str(A2_CKPT.relative_to(PROJECT_ROOT)),
            "threshold": float(a2_frozen["threshold"]),
            "parameters": int(a2_frozen["total_parameters"]),
            "primary": False,
            "source": "RQ3 A2 frozen",
        },
        {
            "regime": "F1",
            "model_id": F1_ID,
            "checkpoint": str(F1_CKPT.relative_to(PROJECT_ROOT)),
            "threshold": float(f1_thr),
            "parameters": int(f1_ckpt["total_parameters"]),
            "primary": False,
            "source": "RQ4 F1",
        },
        {
            "regime": "F2",
            "model_id": F2_ID,
            "checkpoint": str(F2_SELECTED.relative_to(PROJECT_ROOT)),
            "threshold": float(f2_thr),
            "parameters": int(param_counts["total"]),
            "primary": True,
            "source": "RQ4 F2",
        },
    ]
    pd.DataFrame(frozen_rows).to_csv(FROZEN_CSV, index=False)

    report = [
        "RQ4 Stage 24B — Threshold Selection + Freeze Report",
        "=" * 60,
        "",
        f"F1 threshold (Youden): {f1_thr:.12f}  J={f1_j:.6f}  clean AUC={f1_auc:.6f} AP={f1_ap:.6f}",
        f"F2 threshold (Youden): {f2_thr:.12f}  J={f2_j:.6f}  clean AUC={f2_auc:.6f} AP={f2_ap:.6f}",
        f"F0 threshold (from A2): {a2_frozen['threshold']}",
        "",
        f"F2 selected phase/epoch: {selected['phase']}/{selected['epoch']}",
        f"F2 RobustValAUC: {selected['robust_val_auc']:.6f}",
        f"Platform: {platform.platform()}",
        f"Device: {device}",
        "",
        "Test accessed: NO",
    ]
    THRESH_REPORT.write_text("\n".join(report) + "\n")

    gate = {
        "F0_frozen": True,
        "F1_checkpoint_frozen": True,
        "F1_threshold_frozen": True,
        "F2_checkpoint_frozen": True,
        "F2_threshold_frozen": True,
        "F2_marked_primary": True,
        "test_accessed": False,
        "representation_changed": False,
        "architecture_search": False,
        "training_budgets_extended": False,
        "f1_threshold": f1_thr,
        "f2_threshold": f2_thr,
        "f2_phase": selected["phase"],
        "f2_epoch": selected["epoch"],
    }
    with open(GATE_PATH, "w") as f:
        json.dump(gate, f, indent=2)
        f.write("\n")

    print("\n=== Stage 24B HARD FREEZE GATE ===")
    for k, v in gate.items():
        print(f"  {k}: {v}")
    print("Stage 24B COMPLETE. Authorised to begin 24C.")


if __name__ == "__main__":
    main()
