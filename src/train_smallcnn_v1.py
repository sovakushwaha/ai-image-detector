"""Train SmallCNNV1 on train/validation only (Stage 13).

Why this file exists
--------------------
First authorised CNN training experiment. Uses frozen train-only RGB
normalisation from Stage 12, trains 30 fixed epochs, and saves the
best validation-ROC-AUC checkpoint.

known_test and unseen_test remain locked. Classical Baseline V1 is not
modified.

How to run
----------
    source .venv/bin/activate
    python src/train_smallcnn_v1.py

What to expect
--------------
    models/smallcnn_v1_best.pt
    results/smallcnn_v1_training_history.csv
    results/smallcnn_v1_training_report.txt
    figures/smallcnn_v1_loss_curve.png
    figures/smallcnn_v1_validation_auc.png
    figures/smallcnn_v1_validation_ap.png
"""

from __future__ import annotations

import json
import platform
import random
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

from cnn_dataset_v1 import (
    ControlledV1Dataset,
    build_transforms,
    load_train_rgb_stats,
    select_device,
)
from small_cnn_v1 import SmallCNNV1, count_parameters

# --- experiment constants ---
EXPERIMENT_ID = "smallcnn_v1_baseline"
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DIAGNOSTIC_THRESHOLD = 0.50
NUM_WORKERS = 0
EXPECTED_TRAIN = 1376
EXPECTED_VAL = 456

REPRESENTATION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORM_PATH = PROJECT_ROOT / "results" / "cnn_train_normalization_v1.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "smallcnn_v1_best.pt"
HISTORY_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_training_history.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_training_report.txt"
LOSS_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_loss_curve.png"
AUC_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_validation_auc.png"
AP_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_validation_ap.png"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DIAGNOSTIC_THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train one epoch. Returns sample-weighted loss, ROC-AUC, AP."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

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
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    train_loss = total_loss / total_samples
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    train_auc = float(roc_auc_score(labels_np, probs))
    train_ap = float(average_precision_score(labels_np, probs))
    return train_loss, train_auc, train_ap


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, dict]:
    """Evaluate one epoch. Returns loss, ROC-AUC, AP, threshold-0.50 metrics."""
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
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    val_loss = total_loss / total_samples
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    val_auc = float(roc_auc_score(labels_np, probs))
    val_ap = float(average_precision_score(labels_np, probs))
    thresh_metrics = threshold_metrics(labels_np, probs)
    return val_loss, val_auc, val_ap, thresh_metrics


def is_better_candidate(candidate: dict, current_best: dict | None) -> bool:
    """Apply predefined checkpoint selection rule."""
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


def save_figures(history: pd.DataFrame) -> None:
    LOSS_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss", marker="o", markersize=3)
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("SmallCNNV1 loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(LOSS_FIG_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["val_roc_auc"], label="Validation ROC-AUC", color="steelblue", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("SmallCNNV1 validation ROC-AUC")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AUC_FIG_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["val_ap"], label="Validation AP", color="darkorange", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Precision")
    ax.set_title("SmallCNNV1 validation Average Precision")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AP_FIG_PATH, dpi=150)
    plt.close(fig)


def write_report(
    history: pd.DataFrame,
    best_row: pd.Series,
    final_row: pd.Series,
    norm_stats: dict,
    total_params: int,
    trainable_params: int,
    device: torch.device,
) -> str:
    min_train = history.loc[history["train_loss"].idxmin()]
    min_val = history.loc[history["val_loss"].idxmin()]
    max_auc = history.loc[history["val_roc_auc"].idxmax()]
    max_ap = history.loc[history["val_ap"].idxmax()]

    def fmt_row(row: pd.Series, prefix: str) -> list[str]:
        return [
            f"- epoch: {int(row['epoch'])}",
            f"- train loss: {row['train_loss']:.6f}",
            f"- validation loss: {row['val_loss']:.6f}",
            f"- validation ROC-AUC: {row['val_roc_auc']:.6f}",
            f"- validation AP: {row['val_ap']:.6f}",
            f"- validation accuracy @ 0.50: {row['val_accuracy_050']:.6f}",
            f"- validation balanced accuracy @ 0.50: {row['val_balanced_accuracy_050']:.6f}",
            f"- validation precision @ 0.50: {row['val_precision_050']:.6f}",
            f"- validation recall @ 0.50: {row['val_recall_050']:.6f}",
            f"- validation specificity @ 0.50: {row['val_specificity_050']:.6f}",
            f"- validation F1 @ 0.50: {row['val_f1_050']:.6f}",
            f"- confusion matrix @ 0.50: TN={int(row['val_TN'])} FP={int(row['val_FP'])} "
            f"FN={int(row['val_FN'])} TP={int(row['val_TP'])}",
        ]

    lines = [
        "SmallCNNV1 Training Report — Stage 13",
        "======================================",
        "",
        "EXPERIMENT",
        f"- experiment ID: {EXPERIMENT_ID}",
        "- model: SmallCNNV1",
        f"- seed: {RANDOM_SEED}",
        f"- device: {device}",
        f"- Python: {sys.version.split()[0]}",
        f"- torch: {torch.__version__}",
        f"- torchvision: {torchvision.__version__}",
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})",
        "",
        "DATA",
        f"- representation: {REPRESENTATION}",
        f"- split protocol: {SPLIT_PROTOCOL}",
        f"- train count: {EXPECTED_TRAIN}",
        f"- validation count: {EXPECTED_VAL}",
        "- confirmation test sets were not accessed: YES",
        f"- train RGB mean: R={norm_stats['mean_R']:.8f}, G={norm_stats['mean_G']:.8f}, B={norm_stats['mean_B']:.8f}",
        f"- train RGB std: R={norm_stats['std_R']:.8f}, G={norm_stats['std_G']:.8f}, B={norm_stats['std_B']:.8f}",
        "",
        "MODEL",
        f"- total parameters: {total_params}",
        f"- trainable parameters: {trainable_params}",
        "",
        "TRAINING CONFIGURATION",
        "- optimizer: AdamW",
        f"- learning rate: {LEARNING_RATE}",
        f"- weight decay: {WEIGHT_DECAY}",
        f"- batch size: {BATCH_SIZE}",
        f"- epochs: {EPOCHS}",
        "- loss: BCEWithLogitsLoss",
        "- augmentation: none",
        "- scheduler: none",
        "- early stopping: none",
        "- class weighting: none",
        "",
        "BEST VALIDATION EPOCH (checkpoint selection rule)",
        "1. highest validation ROC-AUC",
        "2. if tied, higher validation AP",
        "3. if tied, lower validation loss",
        "4. if tied, earlier epoch",
        "",
    ]
    lines.extend(fmt_row(best_row, "best"))
    lines.extend(
        [
            "",
            "FINAL EPOCH",
        ]
    )
    lines.extend(fmt_row(final_row, "final"))
    lines.extend(
        [
            "",
            "LEARNING-BEHAVIOUR SUMMARY",
            f"- minimum train loss: {min_train['train_loss']:.6f} (epoch {int(min_train['epoch'])})",
            f"- minimum validation loss: {min_val['val_loss']:.6f} (epoch {int(min_val['epoch'])})",
            f"- maximum validation ROC-AUC: {max_auc['val_roc_auc']:.6f} (epoch {int(max_auc['epoch'])})",
            f"- maximum validation AP: {max_ap['val_ap']:.6f} (epoch {int(max_ap['epoch'])})",
            "",
            "SAFETY CHECKS",
            "- CNN training performed: YES",
            "- Backward pass performed: YES",
            "- Optimizer updates performed: YES",
            "- Validation used for development: YES",
            "- known_test accessed: NO",
            "- unseen_test accessed: NO",
            "- Classical Baseline V1 modified: NO",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    set_seed(RANDOM_SEED)
    device = select_device()

    norm_stats = load_train_rgb_stats(NORM_PATH)
    transform = build_transforms(norm_stats)

    train_ds = ControlledV1Dataset("train", transform=transform)
    val_ds = ControlledV1Dataset("validation", transform=transform)
    stop_if(len(train_ds) != EXPECTED_TRAIN, f"train count {len(train_ds)} != {EXPECTED_TRAIN}")
    stop_if(len(val_ds) != EXPECTED_VAL, f"validation count {len(val_ds)} != {EXPECTED_VAL}")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = SmallCNNV1().to(device)
    total_params, trainable_params = count_parameters(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history_rows: list[dict] = []
    best_record: dict | None = None
    best_state: dict | None = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc, train_ap = run_epoch_train(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc, val_ap, val_thresh = run_epoch_eval(
            model, val_loader, criterion, device
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_roc_auc": train_auc,
            "train_ap": train_ap,
            "val_loss": val_loss,
            "val_roc_auc": val_auc,
            "val_ap": val_ap,
            "val_accuracy_050": val_thresh["accuracy"],
            "val_balanced_accuracy_050": val_thresh["balanced_accuracy"],
            "val_precision_050": val_thresh["precision"],
            "val_recall_050": val_thresh["recall"],
            "val_specificity_050": val_thresh["specificity"],
            "val_f1_050": val_thresh["f1"],
            "val_TN": val_thresh["TN"],
            "val_FP": val_thresh["FP"],
            "val_FN": val_thresh["FN"],
            "val_TP": val_thresh["TP"],
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | val_ap={val_ap:.4f}"
        )

        if is_better_candidate(row, best_record):
            best_record = row
            best_state = {
                "experiment_id": EXPERIMENT_ID,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_roc_auc": val_auc,
                "validation_ap": val_ap,
                "validation_loss": val_loss,
                "seed": RANDOM_SEED,
                "normalization_statistics": norm_stats,
                "training_configuration": {
                    "model": "SmallCNNV1",
                    "initialization": "random/from scratch",
                    "representation": REPRESENTATION,
                    "split_protocol": SPLIT_PROTOCOL,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "optimizer": "AdamW",
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "loss": "BCEWithLogitsLoss",
                    "augmentation": "none",
                    "scheduler": "none",
                    "early_stopping": "none",
                    "class_weighting": "none",
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "torchvision_version": torchvision.__version__,
                },
            }

    stop_if(best_state is None or best_record is None, "no best checkpoint recorded")

    history = pd.DataFrame(history_rows)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)
    save_figures(history)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CHECKPOINT_PATH)

    final_row = history.iloc[-1]
    best_row = history[history["epoch"] == best_record["epoch"]].iloc[0]
    report = write_report(
        history,
        best_row,
        final_row,
        norm_stats,
        total_params,
        trainable_params,
        device,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("")
    print("STAGE 13 — SMALLCNNV1 TRAINING COMPLETE")
    print(f"Best epoch: {int(best_record['epoch'])}")
    print(f"Best validation ROC-AUC: {best_record['val_roc_auc']:.6f}")
    print(f"Best validation AP: {best_record['val_ap']:.6f}")
    print(f"Best validation loss: {best_record['val_loss']:.6f}")
    print(f"Final epoch validation ROC-AUC: {final_row['val_roc_auc']:.6f}")
    print(f"Final epoch validation AP: {final_row['val_ap']:.6f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("Test sets accessed: NO")


if __name__ == "__main__":
    main()
