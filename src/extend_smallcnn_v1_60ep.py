"""Extend SmallCNNV1 training to 60 epochs (Stage 14 convergence extension).

Why this file exists
--------------------
Stage 13 ended at epoch 30 while validation metrics were still improving.
This script resumes from the Stage 13 checkpoint (Case A) when
optimizer_state_dict is present, appends epochs 31–60, and selects the
best checkpoint across the full 60-epoch trajectory.

Does NOT overwrite Stage 13 outputs or access test sets.

How to run
----------
    source .venv/bin/activate
    python src/extend_smallcnn_v1_60ep.py

What to expect
--------------
    models/smallcnn_v1_60ep_best.pt
    results/smallcnn_v1_60ep_training_history.csv
    results/smallcnn_v1_60ep_training_report.txt
    figures/smallcnn_v1_60ep_loss_curve.png
    figures/smallcnn_v1_60ep_validation_auc.png
    figures/smallcnn_v1_60ep_validation_ap.png
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from cnn_dataset_v1 import (
    ControlledV1Dataset,
    build_transforms,
    load_train_rgb_stats,
    select_device,
)
from small_cnn_v1 import SmallCNNV1, count_parameters
from train_smallcnn_v1 import (
    BATCH_SIZE,
    EXPECTED_TRAIN,
    EXPECTED_VAL,
    EXPERIMENT_ID,
    LEARNING_RATE,
    NUM_WORKERS,
    RANDOM_SEED,
    REPRESENTATION,
    SPLIT_PROTOCOL,
    WEIGHT_DECAY,
    is_better_candidate,
    run_epoch_eval,
    run_epoch_train,
    set_seed,
    stop_if,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE13_CHECKPOINT = PROJECT_ROOT / "models" / "smallcnn_v1_best.pt"
STAGE13_HISTORY = PROJECT_ROOT / "results" / "smallcnn_v1_training_history.csv"
NORM_PATH = PROJECT_ROOT / "results" / "cnn_train_normalization_v1.json"

CHECKPOINT_60_PATH = PROJECT_ROOT / "models" / "smallcnn_v1_60ep_best.pt"
HISTORY_60_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_60ep_training_history.csv"
REPORT_60_PATH = PROJECT_ROOT / "results" / "smallcnn_v1_60ep_training_report.txt"
LOSS_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_60ep_loss_curve.png"
AUC_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_60ep_validation_auc.png"
AP_FIG_PATH = PROJECT_ROOT / "figures" / "smallcnn_v1_60ep_validation_ap.png"

MAX_EPOCHS = 60
RESUME_FROM_EPOCH = 30
STAGE13_REF = {
    "epoch": 30,
    "val_roc_auc": 0.857148,
    "val_ap": 0.867749,
    "val_loss": 0.475274,
}


def inspect_checkpoint(path: Path) -> dict:
    stop_if(not path.exists(), f"missing checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = [
        "model_state_dict",
        "epoch",
        "validation_roc_auc",
        "validation_ap",
        "validation_loss",
        "seed",
        "normalization_statistics",
    ]
    missing = [key for key in required if key not in ckpt]
    stop_if(missing, f"checkpoint missing keys: {missing}")
    return {
        "checkpoint": ckpt,
        "has_optimizer_state_dict": "optimizer_state_dict" in ckpt,
        "epoch": int(ckpt["epoch"]),
        "validation_roc_auc": float(ckpt["validation_roc_auc"]),
        "validation_ap": float(ckpt["validation_ap"]),
        "validation_loss": float(ckpt["validation_loss"]),
        "seed": int(ckpt["seed"]),
    }


def history_row_from_metrics(
    epoch: int,
    train_loss: float,
    train_auc: float,
    train_ap: float,
    val_loss: float,
    val_auc: float,
    val_ap: float,
    val_thresh: dict,
) -> dict:
    return {
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


def build_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    val_ap: float,
    val_loss: float,
    norm_stats: dict,
    device: torch.device,
) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": 14,
        "max_epochs": MAX_EPOCHS,
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
            "representation": REPRESENTATION,
            "split_protocol": SPLIT_PROTOCOL,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "BCEWithLogitsLoss",
            "augmentation": "none",
            "scheduler": "none",
            "class_weighting": "none",
            "device": str(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        },
    }


def save_figures(history: pd.DataFrame, best_epoch: int) -> None:
    LOSS_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss", marker="o", markersize=2)
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss", marker="s", markersize=2)
    ax.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("SmallCNNV1 60-epoch loss curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(LOSS_FIG_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["val_roc_auc"], color="steelblue", marker="o", markersize=2)
    ax.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("SmallCNNV1 60-epoch validation ROC-AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AUC_FIG_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["val_ap"], color="darkorange", marker="o", markersize=2)
    ax.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Precision")
    ax.set_title("SmallCNNV1 60-epoch validation AP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AP_FIG_PATH, dpi=150)
    plt.close(fig)


def metric_at_epoch(history: pd.DataFrame, epoch: int, column: str) -> float:
    row = history.loc[history["epoch"] == epoch]
    stop_if(row.empty, f"epoch {epoch} missing from history")
    return float(row.iloc[0][column])


def write_report(
    history: pd.DataFrame,
    best_row: pd.Series,
    continuation_method: str,
    device: torch.device,
    checkpoint_info: dict,
) -> str:
    min_train = history.loc[history["train_loss"].idxmin()]
    min_val = history.loc[history["val_loss"].idxmin()]
    max_auc = history.loc[history["val_roc_auc"].idxmax()]
    max_ap = history.loc[history["val_ap"].idxmax()]

    snapshot_epochs = [10, 20, 30, 40, 50, 60]
    snapshot_lines = []
    for ep in snapshot_epochs:
        snapshot_lines.append(
            f"  epoch {ep:2d}: val_auc={metric_at_epoch(history, ep, 'val_roc_auc'):.6f}, "
            f"val_ap={metric_at_epoch(history, ep, 'val_ap'):.6f}, "
            f"val_loss={metric_at_epoch(history, ep, 'val_loss'):.6f}"
        )

    best_epoch = int(best_row["epoch"])
    ep30_auc = metric_at_epoch(history, 30, "val_roc_auc")
    ep50_auc = metric_at_epoch(history, 50, "val_roc_auc")
    ep60_auc = metric_at_epoch(history, 60, "val_roc_auc")
    ep30_loss = metric_at_epoch(history, 30, "val_loss")
    ep60_loss = metric_at_epoch(history, 60, "val_loss")

    lines = [
        "SmallCNNV1 60-Epoch Convergence Extension — Stage 14",
        "=====================================================",
        "",
        "EXPERIMENT",
        "- stage: 14",
        f"- continuation method: {continuation_method}",
        f"- experiment family: {EXPERIMENT_ID}",
        f"- seed: {RANDOM_SEED}",
        f"- device: {device}",
        f"- Python: {sys.version.split()[0]}",
        f"- torch: {torch.__version__}",
        f"- torchvision: {torchvision.__version__}",
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})",
        "",
        "CHECKPOINT INSPECTION (Stage 13)",
        "- path: models/smallcnn_v1_best.pt",
        "- model_state_dict: present",
        f"- optimizer_state_dict: {'present' if checkpoint_info['has_optimizer_state_dict'] else 'absent'}",
        f"- epoch: {checkpoint_info['epoch']}",
        f"- validation_roc_auc: {checkpoint_info['validation_roc_auc']:.6f}",
        f"- validation_ap: {checkpoint_info['validation_ap']:.6f}",
        f"- validation_loss: {checkpoint_info['validation_loss']:.6f}",
        f"- seed: {checkpoint_info['seed']}",
        "- normalization_statistics: present",
        "",
        "PROTOCOL (locked)",
        "- model: SmallCNNV1",
        f"- max epochs: {MAX_EPOCHS}",
        f"- batch size: {BATCH_SIZE}",
        "- optimizer: AdamW",
        f"- learning rate: {LEARNING_RATE}",
        f"- weight decay: {WEIGHT_DECAY}",
        "- loss: BCEWithLogitsLoss",
        "- augmentation: none",
        "- scheduler: none",
        "- class weights: none",
        f"- representation: {REPRESENTATION}",
        f"- split protocol: {SPLIT_PROTOCOL}",
        "- normalization: Stage 12 train-only RGB mean/std (reused)",
        "",
        "DATA",
        f"- train count: {EXPECTED_TRAIN}",
        f"- validation count: {EXPECTED_VAL}",
        "- confirmation test sets were not accessed: YES",
        "",
        "STAGE 13 REFERENCE (epoch 30)",
        f"- validation ROC-AUC: {STAGE13_REF['val_roc_auc']:.6f}",
        f"- validation AP: {STAGE13_REF['val_ap']:.6f}",
        f"- validation loss: {STAGE13_REF['val_loss']:.6f}",
        "",
        "60-EPOCH RESULT (selected best checkpoint)",
        f"- best epoch: {best_epoch}",
        f"- best validation ROC-AUC: {best_row['val_roc_auc']:.6f}",
        f"- validation AP at selected epoch: {best_row['val_ap']:.6f}",
        f"- validation loss at selected epoch: {best_row['val_loss']:.6f}",
        f"- validation accuracy @ 0.50: {best_row['val_accuracy_050']:.6f}",
        f"- validation balanced accuracy @ 0.50: {best_row['val_balanced_accuracy_050']:.6f}",
        f"- validation precision @ 0.50: {best_row['val_precision_050']:.6f}",
        f"- validation recall @ 0.50: {best_row['val_recall_050']:.6f}",
        f"- validation specificity @ 0.50: {best_row['val_specificity_050']:.6f}",
        f"- validation F1 @ 0.50: {best_row['val_f1_050']:.6f}",
        f"- confusion matrix @ 0.50: TN={int(best_row['val_TN'])} FP={int(best_row['val_FP'])} "
        f"FN={int(best_row['val_FN'])} TP={int(best_row['val_TP'])}",
        "",
        "CONVERGENCE SUMMARY",
        f"- best validation ROC-AUC: {max_auc['val_roc_auc']:.6f} (epoch {int(max_auc['epoch'])})",
        f"- best validation AP: {max_ap['val_ap']:.6f} (epoch {int(max_ap['epoch'])})",
        f"- minimum validation loss: {min_val['val_loss']:.6f} (epoch {int(min_val['epoch'])})",
        f"- minimum training loss: {min_train['train_loss']:.6f} (epoch {int(min_train['epoch'])})",
        "",
        "Validation snapshots:",
        *snapshot_lines,
        "",
        "Measured changes:",
        f"- AUC improvement epoch 30 → best epoch ({best_epoch}): {float(best_row['val_roc_auc']) - ep30_auc:+.6f}",
        f"- AUC improvement epoch 50 → epoch 60: {ep60_auc - ep50_auc:+.6f}",
        f"- validation loss change epoch 30 → epoch 60: {ep60_loss - ep30_loss:+.6f}",
        "",
        "SAFETY",
        "- known_test accessed: NO",
        "- unseen_test accessed: NO",
        "- threshold tuning performed: NO",
        "- augmentation introduced: NO",
        "- LR changed: NO",
        "- architecture changed: NO",
        "- Classical Baseline V1 modified: NO",
        "- Stage 13 checkpoint overwritten: NO",
        "- Stage 13 history overwritten: NO",
    ]
    return "\n".join(lines)


def main() -> None:
    set_seed(RANDOM_SEED)
    device = select_device()

    checkpoint_info = inspect_checkpoint(STAGE13_CHECKPOINT)
    ckpt = checkpoint_info["checkpoint"]

    print("Stage 14 — checkpoint inspection")
    print("  model_state_dict: present")
    print(f"  optimizer_state_dict: {'present' if checkpoint_info['has_optimizer_state_dict'] else 'absent'}")
    print(f"  epoch: {checkpoint_info['epoch']}")
    print(f"  validation_roc_auc: {checkpoint_info['validation_roc_auc']:.6f}")
    print(f"  validation_ap: {checkpoint_info['validation_ap']:.6f}")
    print(f"  validation_loss: {checkpoint_info['validation_loss']:.6f}")
    print(f"  seed: {checkpoint_info['seed']}")
    print("  normalization_statistics: present")

    stop_if(not checkpoint_info["has_optimizer_state_dict"], "CASE B required — optimizer_state_dict missing")
    continuation_method = "CASE A — resume from epoch 30 with saved optimizer state"
    stop_if(checkpoint_info["epoch"] != RESUME_FROM_EPOCH, f"expected resume epoch {RESUME_FROM_EPOCH}")

    stop_if(not STAGE13_HISTORY.exists(), f"missing Stage 13 history: {STAGE13_HISTORY}")
    stage13_history = pd.read_csv(STAGE13_HISTORY)
    stop_if(len(stage13_history) != RESUME_FROM_EPOCH, f"Stage 13 history rows {len(stage13_history)}")

    norm_stats = load_train_rgb_stats(NORM_PATH)
    transform = build_transforms(norm_stats)

    train_ds = ControlledV1Dataset("train", transform=transform)
    val_ds = ControlledV1Dataset("validation", transform=transform)
    stop_if(len(train_ds) != EXPECTED_TRAIN, f"train count {len(train_ds)}")
    stop_if(len(val_ds) != EXPECTED_VAL, f"validation count {len(val_ds)}")

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
    model.load_state_dict(ckpt["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Initialise best from epochs 1–30 (Stage 13 history).
    best_record: dict | None = None
    best_state: dict | None = None
    for _, row in stage13_history.iterrows():
        candidate = row.to_dict()
        candidate["epoch"] = int(candidate["epoch"])
        if is_better_candidate(candidate, best_record):
            best_record = candidate

    # Stage 13 checkpoint weights correspond to epoch 30 (final Stage 13 epoch).
    stage13_model = SmallCNNV1().to(device)
    stage13_model.load_state_dict(ckpt["model_state_dict"])
    best_state = build_checkpoint_payload(
        stage13_model,
        optimizer,
        int(best_record["epoch"]),
        float(best_record["val_roc_auc"]),
        float(best_record["val_ap"]),
        float(best_record["val_loss"]),
        norm_stats,
        device,
    )
    best_state["model_state_dict"] = ckpt["model_state_dict"]
    best_state["optimizer_state_dict"] = ckpt["optimizer_state_dict"]

    new_rows: list[dict] = []
    print(f"\nContinuation method: {continuation_method}")
    print(f"Resuming epochs {RESUME_FROM_EPOCH + 1}–{MAX_EPOCHS}\n")

    for epoch in range(RESUME_FROM_EPOCH + 1, MAX_EPOCHS + 1):
        train_loss, train_auc, train_ap = run_epoch_train(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc, val_ap, val_thresh = run_epoch_eval(
            model, val_loader, criterion, device
        )
        row = history_row_from_metrics(
            epoch, train_loss, train_auc, train_ap, val_loss, val_auc, val_ap, val_thresh
        )
        new_rows.append(row)
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | val_ap={val_ap:.4f}"
        )

        if is_better_candidate(row, best_record):
            best_record = row
            best_state = build_checkpoint_payload(
                model, optimizer, epoch, val_auc, val_ap, val_loss, norm_stats, device
            )

    history = pd.concat([stage13_history, pd.DataFrame(new_rows)], ignore_index=True)
    stop_if(len(history) != MAX_EPOCHS, f"expected {MAX_EPOCHS} history rows, got {len(history)}")

    # Re-select best across full history to ensure consistency
    best_record = None
    for _, row in history.iterrows():
        candidate = row.to_dict()
        candidate["epoch"] = int(candidate["epoch"])
        if is_better_candidate(candidate, best_record):
            best_record = candidate

    best_epoch = int(best_record["epoch"])
    best_row = history[history["epoch"] == best_epoch].iloc[0]

    # If best is in 1-30 and not captured during extension loop, use stage13 checkpoint
    if best_epoch <= RESUME_FROM_EPOCH:
        best_state = build_checkpoint_payload(
            stage13_model.to(device),
            optimizer,
            best_epoch,
            float(best_row["val_roc_auc"]),
            float(best_row["val_ap"]),
            float(best_row["val_loss"]),
            norm_stats,
            device,
        )
        best_state["model_state_dict"] = ckpt["model_state_dict"]
        best_state["optimizer_state_dict"] = ckpt["optimizer_state_dict"]

    stop_if(best_state is None, "no best checkpoint recorded")

    HISTORY_60_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_60_PATH, index=False)
    save_figures(history, best_epoch)

    CHECKPOINT_60_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CHECKPOINT_60_PATH)

    report = write_report(history, best_row, continuation_method, device, checkpoint_info)
    REPORT_60_PATH.write_text(report, encoding="utf-8")

    print("")
    print("STAGE 14 — SMALLCNNV1 60-EPOCH CONVERGENCE EXTENSION COMPLETE")
    print(f"Continuation method: {continuation_method}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation ROC-AUC: {best_record['val_roc_auc']:.6f}")
    print(f"Best validation AP: {best_record['val_ap']:.6f}")
    print(f"Best validation loss: {best_record['val_loss']:.6f}")
    print(f"AUC epoch 30 → best: {float(best_record['val_roc_auc']) - metric_at_epoch(history, 30, 'val_roc_auc'):+.6f}")
    print(f"AUC epoch 50 → 60: {metric_at_epoch(history, 60, 'val_roc_auc') - metric_at_epoch(history, 50, 'val_roc_auc'):+.6f}")
    print(f"Val loss epoch 30 → 60: {metric_at_epoch(history, 60, 'val_loss') - metric_at_epoch(history, 30, 'val_loss'):+.6f}")
    print(f"Checkpoint: {CHECKPOINT_60_PATH}")
    print("Test sets accessed: NO")


if __name__ == "__main__":
    main()
