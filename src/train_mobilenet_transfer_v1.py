"""Train MobileNetV3-Small transfer learning (Stage 18B).

Experiment ID: mobilenet_v3_small_transfer_v1

PHASE 1 — Fixed pretrained feature extraction (10 epochs)
    Freeze backbone.features; train classifier only; AdamW lr=1e-3

PHASE 2 — Full-network fine-tuning (20 epochs)
    Start from best Phase-1 checkpoint; unfreeze all; AdamW lr=1e-4

Train + validation only. known_test / unseen_test remain CLOSED.
No augmentation, scheduler, early stopping, or threshold optimisation.

How to run
----------
    source .venv/bin/activate
    python src/train_mobilenet_transfer_v1.py
"""

from __future__ import annotations

import copy
import platform
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn_dataset_v1 import ControlledV1Dataset, load_split_metadata, select_device
from mobilenet_v3_small_binary_v1 import (
    DEFAULT_WEIGHTS,
    MobileNetV3SmallBinaryV1,
    count_parameters,
)

# --- experiment constants (FIXED — do not change after seeing validation) ---
EXPERIMENT_ID = "mobilenet_v3_small_transfer_v1"
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_TRAIN = 1376
EXPECTED_VAL = 456

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
DIAGNOSTIC_THRESHOLD = 0.50

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REPRESENTATION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"
WEIGHTS_ENUM = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

PHASE1_CKPT = MODELS_DIR / "mobilenet_v3_small_phase1_best_v1.pt"
PHASE2_CKPT = MODELS_DIR / "mobilenet_v3_small_phase2_best_v1.pt"
SELECTED_CKPT = MODELS_DIR / "mobilenet_v3_small_selected_v1.pt"
PHASE1_HISTORY = RESULTS_DIR / "mobilenet_v3_small_phase1_history_v1.csv"
PHASE2_HISTORY = RESULTS_DIR / "mobilenet_v3_small_phase2_history_v1.csv"
REPORT_PATH = RESULTS_DIR / "mobilenet_v3_small_training_report_v1.txt"
LOSS_FIG = FIGURES_DIR / "mobilenet_v3_small_transfer_loss_v1.png"
AUC_FIG = FIGURES_DIR / "mobilenet_v3_small_transfer_auc_v1.png"
AP_FIG = FIGURES_DIR / "mobilenet_v3_small_transfer_ap_v1.png"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_imagenet_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DIAGNOSTIC_THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fpr": float(fpr),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def is_better_checkpoint(candidate: dict, current_best: dict | None) -> bool:
    """Selection: higher AUC → higher AP → lower loss → earlier epoch."""
    if current_best is None:
        return True
    if candidate["val_roc_auc"] > current_best["val_roc_auc"]:
        return True
    if candidate["val_roc_auc"] < current_best["val_roc_auc"]:
        return False
    if candidate["val_ap"] > current_best["val_ap"]:
        return True
    if candidate["val_ap"] < current_best["val_ap"]:
        return False
    if candidate["val_loss"] < current_best["val_loss"]:
        return True
    if candidate["val_loss"] > current_best["val_loss"]:
        return False
    return candidate["epoch"] < current_best["epoch"]


def prefer_phase1_on_tie(a: dict, b: dict) -> dict:
    """Final selection between Phase-1 and Phase-2 bests; prefer Phase 1 on full tie."""
    if a["val_roc_auc"] != b["val_roc_auc"]:
        return a if a["val_roc_auc"] > b["val_roc_auc"] else b
    if a["val_ap"] != b["val_ap"]:
        return a if a["val_ap"] > b["val_ap"] else b
    if a["val_loss"] != b["val_loss"]:
        return a if a["val_loss"] < b["val_loss"] else b
    # Prefer Phase 1 (less adaptation) on exact remaining tie.
    return a if a["phase"] == 1 else b


def snapshot_bn_running_stats(features: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snaps: list[tuple[torch.Tensor, torch.Tensor]] = []
    for module in features.modules():
        if isinstance(module, nn.BatchNorm2d):
            snaps.append(
                (
                    module.running_mean.detach().cpu().clone(),
                    module.running_var.detach().cpu().clone(),
                )
            )
    return snaps


def bn_stats_unchanged(
    features: nn.Module,
    before: list[tuple[torch.Tensor, torch.Tensor]],
) -> bool:
    after = snapshot_bn_running_stats(features)
    stop_if(len(before) != len(after), "BatchNorm module count changed unexpectedly")
    for (mean_b, var_b), (mean_a, var_a) in zip(before, after):
        if not torch.equal(mean_b, mean_a) or not torch.equal(var_b, var_a):
            return False
    return True


def run_phase1_train_epoch(
    model: MobileNetV3SmallBinaryV1,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one Phase-1 epoch with backbone BatchNorm frozen via features.eval()."""
    model.train()
    model.features.eval()  # keep pretrained BN running stats fixed

    # Confirm no frozen params are in the optimizer.
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        for p in group["params"]:
            stop_if(id(p) not in trainable_ids, "optimizer contains a frozen parameter")

    bn_before = snapshot_bn_running_stats(model.features)
    frozen_params = [p for p in model.features.parameters()]
    frozen_before = [p.detach().cpu().clone() for p in frozen_params]

    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        stop_if(logits.shape != labels.shape, f"logits {logits.shape} != labels {labels.shape}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    stop_if(
        not bn_stats_unchanged(model.features, bn_before),
        "Phase 1: backbone BatchNorm running statistics were updated",
    )
    for p, before in zip(frozen_params, frozen_before):
        stop_if(
            not torch.equal(p.detach().cpu(), before),
            "Phase 1: frozen backbone parameter was updated",
        )

    return total_loss / total_samples


def run_phase2_train_epoch(
    model: MobileNetV3SmallBinaryV1,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train one Phase-2 epoch with full model.train() (BN may adapt)."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        stop_if(logits.shape != labels.shape, f"logits {logits.shape} != labels {labels.shape}")
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
    thr = threshold_metrics(labels_np.astype(int), probs)

    out = {
        "loss": total_loss / total_samples,
        "roc_auc": float(roc_auc_score(labels_np, probs)),
        "ap": float(average_precision_score(labels_np, probs)),
    }
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "fpr"):
        out[f"{key}_050"] = thr[key]
    for key in ("TN", "FP", "FN", "TP"):
        out[key] = thr[key]
    return out


def history_row(phase: int, epoch: int, train_loss: float, val: dict) -> dict:
    return {
        "phase": phase,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val["loss"],
        "val_roc_auc": val["roc_auc"],
        "val_ap": val["ap"],
        "val_accuracy_050": val["accuracy_050"],
        "val_balanced_accuracy_050": val["balanced_accuracy_050"],
        "val_precision_050": val["precision_050"],
        "val_recall_050": val["recall_050"],
        "val_specificity_050": val["specificity_050"],
        "val_f1_050": val["f1_050"],
        "val_fpr_050": val["fpr_050"],
    }


def save_checkpoint(
    path: Path,
    model: MobileNetV3SmallBinaryV1,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: int,
    val: dict,
    config: dict,
    total_params: int,
    trainable_params: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "validation_metrics": val,
        "configuration": config,
        "seed": RANDOM_SEED,
        "pretrained_weights": WEIGHTS_ENUM,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "freezing_configuration": config.get("freezing_configuration"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "representation": REPRESENTATION,
        "split_protocol": SPLIT_PROTOCOL,
    }
    torch.save(payload, path)


def plot_curves(phase1_df: pd.DataFrame, phase2_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def _style(ax, ylabel: str, title: str) -> None:
        ax.set_xlabel("Epoch within phase")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phase1_df["epoch"], phase1_df["train_loss"], "o-", label="Phase 1 train loss (fixed features)")
    ax.plot(phase1_df["epoch"], phase1_df["val_loss"], "s-", label="Phase 1 val loss (fixed features)")
    ax.plot(phase2_df["epoch"], phase2_df["train_loss"], "o--", label="Phase 2 train loss (full fine-tune)")
    ax.plot(phase2_df["epoch"], phase2_df["val_loss"], "s--", label="Phase 2 val loss (full fine-tune)")
    _style(ax, "Loss", "MobileNetV3-Small transfer: train/val loss")
    fig.tight_layout()
    fig.savefig(LOSS_FIG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phase1_df["epoch"], phase1_df["val_roc_auc"], "o-", label="Phase 1 val ROC-AUC (fixed features)")
    ax.plot(phase2_df["epoch"], phase2_df["val_roc_auc"], "s--", label="Phase 2 val ROC-AUC (full fine-tune)")
    _style(ax, "ROC-AUC", "MobileNetV3-Small transfer: validation ROC-AUC")
    fig.tight_layout()
    fig.savefig(AUC_FIG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phase1_df["epoch"], phase1_df["val_ap"], "o-", label="Phase 1 val AP (fixed features)")
    ax.plot(phase2_df["epoch"], phase2_df["val_ap"], "s--", label="Phase 2 val AP (full fine-tune)")
    _style(ax, "Average Precision", "MobileNetV3-Small transfer: validation AP")
    fig.tight_layout()
    fig.savefig(AP_FIG, dpi=150)
    plt.close(fig)


def main() -> None:
    set_seed(RANDOM_SEED)
    device = select_device()

    print("=" * 60)
    print("Stage 18B — MobileNetV3-Small transfer training")
    print("=" * 60)
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"device: {device}")
    print(f"seed: {RANDOM_SEED}")

    # --- Data (train + validation only) ---
    train_meta = load_split_metadata("train", SPLIT_META_PATH)
    val_meta = load_split_metadata("validation", SPLIT_META_PATH)
    stop_if(len(train_meta) != EXPECTED_TRAIN, f"train={len(train_meta)} != {EXPECTED_TRAIN}")
    stop_if(len(val_meta) != EXPECTED_VAL, f"validation={len(val_meta)} != {EXPECTED_VAL}")

    known_test_rows_used = 0
    unseen_test_rows_used = 0
    stop_if(known_test_rows_used != 0, "known_test rows used")
    stop_if(unseen_test_rows_used != 0, "unseen_test rows used")

    transform = build_imagenet_transforms()
    train_ds = ControlledV1Dataset("train", transform=transform)
    val_ds = ControlledV1Dataset("validation", transform=transform)
    stop_if(len(train_ds) != EXPECTED_TRAIN, f"train dataset {len(train_ds)}")
    stop_if(len(val_ds) != EXPECTED_VAL, f"val dataset {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    criterion = nn.BCEWithLogitsLoss()

    # ============================================================
    # PHASE 1 — Fixed feature extractor
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1 — Fixed pretrained feature extraction (10 epochs)")
    print("=" * 60)

    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS)
    model.freeze_features()
    total_params, phase1_trainable = count_parameters(model)
    phase1_frozen = total_params - phase1_trainable
    stop_if(phase1_trainable <= 0, "Phase 1 has no trainable parameters")
    stop_if(
        any(p.requires_grad for p in model.features.parameters()),
        "Phase 1: some features still require_grad",
    )
    stop_if(
        not all(p.requires_grad for p in model.classifier.parameters()),
        "Phase 1: classifier not fully trainable",
    )

    print(f"Total parameters: {total_params}")
    print(f"Phase-1 trainable parameters: {phase1_trainable}")
    print(f"Phase-1 frozen parameters: {phase1_frozen}")

    model = model.to(device)
    optimizer1 = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    phase1_config = {
        "phase": 1,
        "epochs": PHASE1_EPOCHS,
        "lr": PHASE1_LR,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "batch_size": BATCH_SIZE,
        "freezing_configuration": "features frozen; classifier trainable; features.eval() during train",
        "augmentation": "none",
        "scheduler": "none",
        "early_stopping": "none",
        "normalization": "ImageNet",
    }

    phase1_history: list[dict] = []
    best_phase1: dict | None = None

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = run_phase1_train_epoch(model, train_loader, criterion, optimizer1, device)
        val = evaluate(model, val_loader, criterion, device)
        row = history_row(1, epoch, train_loss, val)
        phase1_history.append(row)

        print(
            f"Phase1 Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | "
            f"val_AUC={val['roc_auc']:.4f} | val_AP={val['ap']:.4f}"
        )

        candidate = {
            "phase": 1,
            "epoch": epoch,
            "val_roc_auc": val["roc_auc"],
            "val_ap": val["ap"],
            "val_loss": val["loss"],
            "validation_metrics": copy.deepcopy(val),
            "train_loss": train_loss,
        }
        if is_better_checkpoint(candidate, best_phase1):
            best_phase1 = candidate
            save_checkpoint(
                PHASE1_CKPT,
                model,
                optimizer1,
                epoch,
                phase=1,
                val=val,
                config=phase1_config,
                total_params=total_params,
                trainable_params=phase1_trainable,
            )
            print(f"  → saved best Phase-1 checkpoint (epoch {epoch})")

    phase1_df = pd.DataFrame(phase1_history)
    PHASE1_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    phase1_df.to_csv(PHASE1_HISTORY, index=False)
    stop_if(best_phase1 is None, "Phase 1 produced no best checkpoint")

    print(
        f"\nPhase 1 best: epoch={best_phase1['epoch']} "
        f"AUC={best_phase1['val_roc_auc']:.6f} "
        f"AP={best_phase1['val_ap']:.6f} "
        f"loss={best_phase1['val_loss']:.6f}"
    )

    # ============================================================
    # PHASE 2 — Full fine-tuning from best Phase-1 checkpoint
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2 — Full-network fine-tuning (20 epochs)")
    print("=" * 60)

    # Fresh model shell + load Phase-1 best weights (do not carry Phase-1 optimizer).
    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS)
    ckpt = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.unfreeze_all()
    total_params2, phase2_trainable = count_parameters(model)
    stop_if(phase2_trainable != total_params2, "Phase 2: not all parameters trainable")
    print(f"Total parameters: {total_params2}")
    print(f"Phase-2 trainable parameters: {phase2_trainable}")
    print(f"Starting from Phase-1 best epoch: {ckpt['epoch']}")

    model = model.to(device)
    optimizer2 = torch.optim.AdamW(
        model.parameters(),
        lr=PHASE2_LR,
        weight_decay=WEIGHT_DECAY,
    )

    phase2_config = {
        "phase": 2,
        "epochs": PHASE2_EPOCHS,
        "lr": PHASE2_LR,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "batch_size": BATCH_SIZE,
        "freezing_configuration": "all parameters trainable; model.train() (BN may adapt)",
        "started_from": str(PHASE1_CKPT.relative_to(PROJECT_ROOT)),
        "started_from_phase1_epoch": int(ckpt["epoch"]),
        "phase1_lr": PHASE1_LR,
        "phase2_lr": PHASE2_LR,
        "lr_rationale": (
            "Phase-1 LR=1e-3 trains a new classifier on fixed features; "
            "Phase-2 LR=1e-4 adapts pretrained representations more cautiously. "
            "Neither LR was tuned from validation."
        ),
        "augmentation": "none",
        "scheduler": "none",
        "early_stopping": "none",
        "normalization": "ImageNet",
    }

    phase2_history: list[dict] = []
    best_phase2: dict | None = None

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = run_phase2_train_epoch(model, train_loader, criterion, optimizer2, device)
        val = evaluate(model, val_loader, criterion, device)
        row = history_row(2, epoch, train_loss, val)
        phase2_history.append(row)

        print(
            f"Phase2 Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | "
            f"val_AUC={val['roc_auc']:.4f} | val_AP={val['ap']:.4f}"
        )

        candidate = {
            "phase": 2,
            "epoch": epoch,
            "val_roc_auc": val["roc_auc"],
            "val_ap": val["ap"],
            "val_loss": val["loss"],
            "validation_metrics": copy.deepcopy(val),
            "train_loss": train_loss,
        }
        if is_better_checkpoint(candidate, best_phase2):
            best_phase2 = candidate
            save_checkpoint(
                PHASE2_CKPT,
                model,
                optimizer2,
                epoch,
                phase=2,
                val=val,
                config=phase2_config,
                total_params=total_params2,
                trainable_params=phase2_trainable,
            )
            print(f"  → saved best Phase-2 checkpoint (epoch {epoch})")

    phase2_df = pd.DataFrame(phase2_history)
    phase2_df.to_csv(PHASE2_HISTORY, index=False)
    stop_if(best_phase2 is None, "Phase 2 produced no best checkpoint")

    print(
        f"\nPhase 2 best: epoch={best_phase2['epoch']} "
        f"AUC={best_phase2['val_roc_auc']:.6f} "
        f"AP={best_phase2['val_ap']:.6f} "
        f"loss={best_phase2['val_loss']:.6f}"
    )

    # ============================================================
    # Final validation selection (Phase 1 vs Phase 2)
    # ============================================================
    selected = prefer_phase1_on_tie(best_phase1, best_phase2)
    if selected["phase"] == 1:
        src_ckpt = PHASE1_CKPT
        selected_label = "Phase 1 (fixed feature extraction)"
    else:
        src_ckpt = PHASE2_CKPT
        selected_label = "Phase 2 (full fine-tuning)"

    shutil.copy2(src_ckpt, SELECTED_CKPT)
    # Annotate selected checkpoint with selection metadata.
    selected_payload = torch.load(SELECTED_CKPT, map_location="cpu", weights_only=False)
    selected_payload["final_selection"] = {
        "selected_phase": selected["phase"],
        "selected_epoch": selected["epoch"],
        "selected_label": selected_label,
        "selection_rule": (
            "higher val ROC-AUC; then higher AP; then lower val loss; "
            "then prefer Phase 1 on remaining tie"
        ),
        "phase1_best": {
            "epoch": best_phase1["epoch"],
            "val_roc_auc": best_phase1["val_roc_auc"],
            "val_ap": best_phase1["val_ap"],
            "val_loss": best_phase1["val_loss"],
        },
        "phase2_best": {
            "epoch": best_phase2["epoch"],
            "val_roc_auc": best_phase2["val_roc_auc"],
            "val_ap": best_phase2["val_ap"],
            "val_loss": best_phase2["val_loss"],
        },
        "threshold_selection_performed": False,
        "test_accessed": False,
    }
    torch.save(selected_payload, SELECTED_CKPT)

    plot_curves(phase1_df, phase2_df)

    p1m = best_phase1["validation_metrics"]
    p2m = best_phase2["validation_metrics"]

    report_lines = [
        "MobileNetV3-Small Transfer Training Report V1",
        "=============================================",
        "",
        "Experiment",
        "----------",
        f"experiment ID: {EXPERIMENT_ID}",
        f"seed: {RANDOM_SEED}",
        f"device: {device}",
        f"pretrained weights: {WEIGHTS_ENUM}",
        f"weights source: {DEFAULT_WEIGHTS.url}",
        "",
        "Data",
        "----",
        f"representation: {REPRESENTATION}",
        f"split protocol: {SPLIT_PROTOCOL}",
        f"train: {EXPECTED_TRAIN}",
        f"validation: {EXPECTED_VAL}",
        "known_test access: NO",
        "unseen_test access: NO",
        f"known_test rows used: {known_test_rows_used}",
        f"unseen_test rows used: {unseen_test_rows_used}",
        "",
        "Model",
        "-----",
        "architecture: MobileNetV3-Small (binary logit head)",
        f"total binary-model parameters: {total_params}",
        f"pretrained weights: {WEIGHTS_ENUM}",
        f"ImageNet mean: {IMAGENET_MEAN}",
        f"ImageNet std: {IMAGENET_STD}",
        "augmentation: none",
        "scheduler: none",
        "",
        "Phase 1 — Fixed feature extraction",
        "----------------------------------",
        f"epochs: {PHASE1_EPOCHS} (fixed budget; not extended)",
        "backbone frozen: YES (features.requires_grad=False)",
        "classifier trainable: YES",
        "BatchNorm rule: model.train(); model.features.eval() each train epoch",
        f"trainable parameter count: {phase1_trainable}",
        f"frozen parameter count: {phase1_frozen}",
        "optimizer: AdamW(classifier.parameters())",
        f"LR: {PHASE1_LR}",
        f"weight decay: {WEIGHT_DECAY}",
        f"best epoch: {best_phase1['epoch']}",
        f"best validation ROC-AUC: {best_phase1['val_roc_auc']:.8f}",
        f"best validation AP: {best_phase1['val_ap']:.8f}",
        f"best validation loss: {best_phase1['val_loss']:.8f}",
        f"diagnostic threshold 0.50 accuracy: {p1m['accuracy_050']:.8f}",
        f"diagnostic threshold 0.50 balanced accuracy: {p1m['balanced_accuracy_050']:.8f}",
        f"diagnostic threshold 0.50 precision: {p1m['precision_050']:.8f}",
        f"diagnostic threshold 0.50 recall: {p1m['recall_050']:.8f}",
        f"diagnostic threshold 0.50 specificity: {p1m['specificity_050']:.8f}",
        f"diagnostic threshold 0.50 F1: {p1m['f1_050']:.8f}",
        f"diagnostic threshold 0.50 FPR: {p1m['fpr_050']:.8f}",
        f"checkpoint: {PHASE1_CKPT.relative_to(PROJECT_ROOT)}",
        "",
        "Phase 2 — Full fine-tuning",
        "--------------------------",
        f"epochs: {PHASE2_EPOCHS} (fixed budget; not extended)",
        "all parameters trainable: YES",
        f"trainable parameter count: {phase2_trainable}",
        "optimizer: NEW AdamW(model.parameters()) — Phase-1 optimizer moments NOT carried over",
        f"LR: {PHASE2_LR}",
        f"weight decay: {WEIGHT_DECAY}",
        f"started from Phase-1 best epoch: {ckpt['epoch']}",
        f"best fine-tuning epoch: {best_phase2['epoch']}",
        f"best validation ROC-AUC: {best_phase2['val_roc_auc']:.8f}",
        f"best validation AP: {best_phase2['val_ap']:.8f}",
        f"best validation loss: {best_phase2['val_loss']:.8f}",
        f"diagnostic threshold 0.50 accuracy: {p2m['accuracy_050']:.8f}",
        f"diagnostic threshold 0.50 balanced accuracy: {p2m['balanced_accuracy_050']:.8f}",
        f"diagnostic threshold 0.50 precision: {p2m['precision_050']:.8f}",
        f"diagnostic threshold 0.50 recall: {p2m['recall_050']:.8f}",
        f"diagnostic threshold 0.50 specificity: {p2m['specificity_050']:.8f}",
        f"diagnostic threshold 0.50 F1: {p2m['f1_050']:.8f}",
        f"diagnostic threshold 0.50 FPR: {p2m['fpr_050']:.8f}",
        f"checkpoint: {PHASE2_CKPT.relative_to(PROJECT_ROOT)}",
        "",
        "Final validation selection",
        "--------------------------",
        "Phase-1 best:",
        f"  AUC={best_phase1['val_roc_auc']:.8f}",
        f"  AP={best_phase1['val_ap']:.8f}",
        f"  loss={best_phase1['val_loss']:.8f}",
        "Phase-2 best:",
        f"  AUC={best_phase2['val_roc_auc']:.8f}",
        f"  AP={best_phase2['val_ap']:.8f}",
        f"  loss={best_phase2['val_loss']:.8f}",
        "Selected candidate:",
        f"  phase: {selected['phase']} ({selected_label})",
        f"  epoch: {selected['epoch']}",
        f"  checkpoint: {SELECTED_CKPT.relative_to(PROJECT_ROOT)}",
        f"  AUC: {selected['val_roc_auc']:.8f}",
        f"  AP: {selected['val_ap']:.8f}",
        "",
        "Safety",
        "------",
        "Phase-1 training performed: YES",
        "Phase-2 fine-tuning performed: YES",
        "known_test accessed: NO",
        "unseen_test accessed: NO",
        "Threshold selection performed: NO",
        "Augmentation introduced: NO",
        "Scheduler introduced: NO",
        "SmallCNNV1 modified: NO",
        "Classical Baseline V1 modified: NO",
        "MobileNet training extended beyond predefined budget: NO",
        "",
        "No threshold selection. No test evaluation.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("STAGE 18B — MOBILENETV3-SMALL TRANSFER TRAINING COMPLETE")
    print("=" * 60)
    print("PHASE 1 — FEATURE EXTRACTION")
    print(f"Best epoch: {best_phase1['epoch']}")
    print(f"Validation ROC-AUC: {best_phase1['val_roc_auc']:.6f}")
    print(f"Validation AP: {best_phase1['val_ap']:.6f}")
    print(f"Validation loss: {best_phase1['val_loss']:.6f}")
    print()
    print("PHASE 2 — FULL FINE-TUNING")
    print(f"Best epoch: {best_phase2['epoch']}")
    print(f"Validation ROC-AUC: {best_phase2['val_roc_auc']:.6f}")
    print(f"Validation AP: {best_phase2['val_ap']:.6f}")
    print(f"Validation loss: {best_phase2['val_loss']:.6f}")
    print()
    print("FINAL VALIDATION SELECTION")
    print(f"Selected phase: {selected['phase']}")
    print(f"Selected epoch: {selected['epoch']}")
    print(f"Validation ROC-AUC: {selected['val_roc_auc']:.6f}")
    print(f"Validation AP: {selected['val_ap']:.6f}")
    print("Checkpoint:")
    print("models/mobilenet_v3_small_selected_v1.pt")
    print()
    print(f"Train images: {EXPECTED_TRAIN}")
    print(f"Validation images: {EXPECTED_VAL}")
    print()
    print("known_test accessed: NO")
    print("unseen_test accessed: NO")
    print("Threshold selected: NO")
    print()
    print("STOP BEFORE TEST EVALUATION")
    print(f"\nWrote {PHASE1_HISTORY}")
    print(f"Wrote {PHASE2_HISTORY}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {LOSS_FIG}")
    print(f"Wrote {AUC_FIG}")
    print(f"Wrote {AP_FIG}")


if __name__ == "__main__":
    main()
