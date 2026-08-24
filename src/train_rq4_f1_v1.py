"""Train RQ4 F1 frequency-only detector (Stage 24A.5–24A.7).

Why this file exists
--------------------
Trains FrequencyOnlyCNNV1 on A2 Resize+JPEG-augmented RGB images converted
to FrequencyTransformV1 spectra. RobustValAUC selects the checkpoint.
ScreenshotStrong evaluated once after selection. No test access.

How to run
----------
    source .venv/bin/activate
    PYTHONPATH=src python src/train_rq4_f1_v1.py
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, load_split_metadata, select_device, stop_if
from rq3_augmentations_v1 import REGIME_CONFIGS, RobustnessAwarePILTransform
from rq4_frequency_cnn_v1 import FrequencyOnlyCNNV1, count_parameters
from rq4_frequency_transform_v1 import FrequencyTransformV1, NORM_PATH

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_TRAIN = 1376
EXPECTED_VAL = 456
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_ID = "rq4_F1_frequency_only_v1"

SELECTION_CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2"]
SCREENSHOT_CONDITION = "screenshot_strong"

SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "rq3_validation_v1_manifest.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

HISTORY_CSV = RESULTS_DIR / "rq4_F1_training_history_v1.csv"
SELECTED_CKPT = MODELS_DIR / "rq4_F1_frequency_only_selected_v1.pt"
REPORT_PATH = RESULTS_DIR / "rq4_F1_development_report_v1.txt"
TRAINING_FIG = FIGURES_DIR / "rq4_F1_training_v1.png"


class FrequencyTrainDataset(Dataset):
    """Apply A2 Resize+JPEG once, then FrequencyTransformV1."""

    def __init__(self, rows: pd.DataFrame, freq_transform: FrequencyTransformV1, seed: int):
        self.rows = rows.reset_index(drop=True)
        self.freq_transform = freq_transform
        self.pil_aug = RobustnessAwarePILTransform(REGIME_CONFIGS["A2"], rng=random.Random(seed))

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
        spectrum = self.freq_transform(augmented)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return spectrum, label, index


class FrequencyEvalDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, freq_transform: FrequencyTransformV1, path_col: str = "path"):
        self.rows = rows.reset_index(drop=True)
        self.freq_transform = freq_transform
        self.path_col = path_col

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / row[self.path_col]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
        stop_if(rgb.size != EXPECTED_SIZE, f"{path} size {rgb.size}")
        spectrum = self.freq_transform(rgb)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return spectrum, label, index


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_condition_frames(meta: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    original = meta[meta["split"] == "validation"][["image_id", "processed_path", "label", "generator"]].copy()
    original = original.rename(columns={"image_id": "source_image_id", "processed_path": "path"})
    original = original.sort_values("source_image_id").reset_index(drop=True)
    stop_if(len(original) != EXPECTED_VAL, f"validation original count {len(original)}")
    frames["original"] = original

    for condition in ["jpeg_q50", "resize_112", "blur_sigma2", SCREENSHOT_CONDITION]:
        sub = manifest[manifest["condition"] == condition].copy()
        sub = sub.rename(columns={"output_path": "path"})
        sub = sub[["source_image_id", "path", "label", "generator"]].sort_values("source_image_id").reset_index(drop=True)
        stop_if(len(sub) != EXPECTED_VAL, f"{condition} count {len(sub)}")
        frames[condition] = sub
    return frames


def is_better_robust(candidate: dict, current: dict | None) -> bool:
    if current is None:
        return True
    if candidate["robust_val_auc"] > current["robust_val_auc"]:
        return True
    if candidate["robust_val_auc"] < current["robust_val_auc"]:
        return False
    if candidate["original_auc"] > current["original_auc"]:
        return True
    if candidate["original_auc"] < current["original_auc"]:
        return False
    if candidate["robust_val_ap"] > current["robust_val_ap"]:
        return True
    if candidate["robust_val_ap"] < current["robust_val_ap"]:
        return False
    return candidate["epoch"] < current["epoch"]


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()
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
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    return labels_np, probs, total_loss / total_samples


@torch.no_grad()
def evaluate_robust_validation(
    model: nn.Module,
    condition_frames: dict[str, pd.DataFrame],
    freq_transform: FrequencyTransformV1,
    device: torch.device,
) -> dict:
    metrics: dict[str, dict[str, float]] = {}
    for condition in SELECTION_CONDITIONS:
        loader = DataLoader(
            FrequencyEvalDataset(condition_frames[condition], freq_transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        labels, probs, loss = predict_probs(model, loader, device)
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def save_checkpoint(
    path: Path,
    model: FrequencyOnlyCNNV1,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val: dict,
    total_params: int,
    trainable_params: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": MODEL_ID,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": val,
        "robust_val_auc": val["robust_val_auc"],
        "robust_val_ap": val["robust_val_ap"],
        "condition_aucs": {c: val["conditions"][c]["roc_auc"] for c in SELECTION_CONDITIONS},
        "condition_aps": {c: val["conditions"][c]["ap"] for c in SELECTION_CONDITIONS},
        "seed": RANDOM_SEED,
        "epochs_budget": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "augmentation": "resize+JPEG-aware (RQ3 A2 recipe) before FrequencyTransformV1",
        "frequency_normalization": str(NORM_PATH.relative_to(PROJECT_ROOT)),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "representation": "controlled_v1",
        "split_protocol": "generator_protocol_v1",
    }
    torch.save(payload, path)


def plot_training(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], "o-")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("F1 training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["robust_val_auc"], "o-", label="RobustValAUC")
    axes[1].plot(history["epoch"], history["original_auc"], "s--", label="Original AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("F1 validation scores")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    TRAINING_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAINING_FIG, dpi=150)
    plt.close(fig)


def write_report(
    selected: dict,
    screenshot: dict,
    total_params: int,
    history: pd.DataFrame,
) -> None:
    lines = [
        "RQ4 Stage 24A — F1 Frequency-Only Development Report",
        "=" * 60,
        "",
        f"Model ID: {MODEL_ID}",
        f"Architecture: FrequencyOnlyCNNV1 (1→16→32→64→128 → AdaptiveAvgPool → Dropout0.2 → Linear128→1)",
        f"Parameters: {total_params}",
        f"Seed: {RANDOM_SEED}",
        f"Epochs: {EPOCHS} FIXED",
        f"Optimizer: AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})",
        f"Augmentation: RQ3 A2 Resize+JPEG on RGB, then FrequencyTransformV1",
        f"Frequency norm: {NORM_PATH}",
        f"Device: {select_device()}",
        f"Platform: {platform.platform()}",
        "",
        "Checkpoint selection rule: RobustValAUC → original AUC → RobustValAP → earlier epoch",
        f"Selected epoch: {selected['epoch']}",
        f"Selected RobustValAUC: {selected['robust_val_auc']:.6f}",
        f"Selected RobustValAP: {selected['robust_val_ap']:.6f}",
        "",
        "Validation (selected checkpoint):",
        f"  Original AUC/AP: {selected['conditions']['original']['roc_auc']:.6f} / {selected['conditions']['original']['ap']:.6f}",
        f"  JPEG50 AUC/AP:   {selected['conditions']['jpeg_q50']['roc_auc']:.6f} / {selected['conditions']['jpeg_q50']['ap']:.6f}",
        f"  Resize112 AUC/AP:{selected['conditions']['resize_112']['roc_auc']:.6f} / {selected['conditions']['resize_112']['ap']:.6f}",
        f"  Blur2 AUC/AP:    {selected['conditions']['blur_sigma2']['roc_auc']:.6f} / {selected['conditions']['blur_sigma2']['ap']:.6f}",
        "",
        "ScreenshotStrong (post-selection once only):",
        f"  AUC: {screenshot['roc_auc']:.6f}",
        f"  AP:  {screenshot['ap']:.6f}",
        "",
        "Integrity:",
        "  Test accessed: NO",
        "  Representation search: NO",
        "  Architecture changed: NO",
        "  Training budget extended: NO",
        "  Screenshot used for selection: NO",
        "",
        f"History rows: {len(history)}",
        f"Checkpoint: {SELECTED_CKPT}",
        f"History CSV: {HISTORY_CSV}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("=== Stage 24A.5–24A.7 Train F1 frequency-only ===")
    stop_if(not NORM_PATH.exists(), f"missing frequency norm: {NORM_PATH}")
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

    train_ds = FrequencyTrainDataset(train_meta, freq_transform, seed=RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = FrequencyOnlyCNNV1().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"F1 parameters: {total_params} (trainable {trainable_params})")
    stop_if(total_params != trainable_params, "unexpected frozen params at start")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history_rows: list[dict] = []
    best: dict | None = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate_robust_validation(model, condition_frames, freq_transform, device)
        row = {
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
        history_rows.append(row)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} loss={train_loss:.4f} "
            f"RobustValAUC={val['robust_val_auc']:.4f} orig={val['original_auc']:.4f}"
        )

        candidate = {
            "epoch": epoch,
            "robust_val_auc": val["robust_val_auc"],
            "robust_val_ap": val["robust_val_ap"],
            "original_auc": val["original_auc"],
            "conditions": val["conditions"],
            "full_val": val,
        }
        if is_better_robust(candidate, best):
            best = candidate
            save_checkpoint(
                SELECTED_CKPT,
                model,
                optimizer,
                epoch,
                val,
                total_params,
                trainable_params,
            )
            print(f"  -> new best checkpoint @ epoch {epoch}")

    stop_if(best is None, "no checkpoint selected")
    history = pd.DataFrame(history_rows)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_CSV, index=False)
    plot_training(history)

    # Reload selected and evaluate screenshot once
    ckpt = torch.load(SELECTED_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    screen_loader = DataLoader(
        FrequencyEvalDataset(condition_frames[SCREENSHOT_CONDITION], freq_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    labels, probs, _ = predict_probs(model, screen_loader, device)
    screenshot = {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
    }

    # Persist screenshot into checkpoint metadata (non-destructive add)
    ckpt["screenshot_strong_auc"] = screenshot["roc_auc"]
    ckpt["screenshot_strong_ap"] = screenshot["ap"]
    ckpt["selected_epoch"] = best["epoch"]
    torch.save(ckpt, SELECTED_CKPT)

    write_report(best, screenshot, total_params, history)

    gate = {
        "frequency_transform_fixed": True,
        "normalization_saved": NORM_PATH.exists(),
        "architecture_fixed": True,
        "training_complete": True,
        "checkpoint_selected": SELECTED_CKPT.exists(),
        "test_accessed": False,
        "representation_search": False,
        "training_budget_extended": False,
        "selected_epoch": best["epoch"],
        "robust_val_auc": best["robust_val_auc"],
        "screenshot_auc": screenshot["roc_auc"],
        "parameters": total_params,
    }
    gate_path = RESULTS_DIR / "rq4_24a_gate_v1.json"
    with open(gate_path, "w") as f:
        json.dump(gate, f, indent=2)
        f.write("\n")

    print("\n=== Stage 24A GATE ===")
    for k, v in gate.items():
        print(f"  {k}: {v}")
    print("Stage 24A COMPLETE. Proceed to 24B.")


if __name__ == "__main__":
    main()
