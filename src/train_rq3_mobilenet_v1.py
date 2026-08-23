"""Train RQ3 transformation-aware MobileNet regimes A1–A3 (Stage 23B).

Why this file exists
--------------------
Trains blur-aware, resize+JPEG-aware, and combined augmentation regimes from
the same ImageNet-pretrained MobileNetV3-Small initialization. Uses train +
validation only; RobustValAUC drives checkpoint selection.

How to run
----------
    source .venv/bin/activate
    python src/train_rq3_mobilenet_v1.py
"""

from __future__ import annotations

import copy
import json
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cnn_dataset_v1 import (
    ControlledV1Dataset,
    EXPECTED_SIZE,
    PROJECT_ROOT,
    load_split_metadata,
    select_device,
    stop_if,
)
from mobilenet_v3_small_binary_v1 import (
    DEFAULT_WEIGHTS,
    MobileNetV3SmallBinaryV1,
    count_parameters,
)
from rq3_augmentations_v1 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    REGIME_CONFIGS,
    RobustnessAwarePILTransform,
    SEED,
    build_eval_transform,
)

# --- locked protocol ---
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
WEIGHTS_ENUM = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
REPRESENTATION = "controlled_v1"
SPLIT_PROTOCOL = "generator_protocol_v1"

SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "rq3_validation_v1_manifest.csv"
A0_METRICS_PATH = PROJECT_ROOT / "results" / "rq3_baseline_validation_metrics_v1.csv"

SELECTION_CONDITIONS = ["original", "jpeg_q50", "resize_112", "blur_sigma2"]
SCREENSHOT_CONDITION = "screenshot_strong"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

SUMMARY_CSV = RESULTS_DIR / "rq3_development_summary_v1.csv"
REPORT_PATH = RESULTS_DIR / "rq3_training_report_v1.txt"
ROBUST_AUC_FIG = FIGURES_DIR / "rq3_robust_val_auc_by_regime_v1.png"
CONDITION_AUC_FIG = FIGURES_DIR / "rq3_validation_auc_by_condition_v1.png"


@dataclass
class RegimeSpec:
    key: str
    label: str
    regime_id: str
    slug: str
    regime_config_key: str
    phase1_ckpt: Path
    phase2_ckpt: Path
    selected_ckpt: Path
    phase1_history: Path
    phase2_history: Path
    training_fig: Path
    simplicity_rank: int


REGIMES: list[RegimeSpec] = [
    RegimeSpec(
        key="A1",
        label="Blur",
        regime_id="mobilenet_blur_aug_v1",
        slug="blur_aug",
        regime_config_key="A1",
        phase1_ckpt=MODELS_DIR / "mobilenet_blur_aug_phase1_best_v1.pt",
        phase2_ckpt=MODELS_DIR / "mobilenet_blur_aug_phase2_best_v1.pt",
        selected_ckpt=MODELS_DIR / "mobilenet_blur_aug_selected_v1.pt",
        phase1_history=RESULTS_DIR / "rq3_A1_phase1_history_v1.csv",
        phase2_history=RESULTS_DIR / "rq3_A1_phase2_history_v1.csv",
        training_fig=FIGURES_DIR / "rq3_A1_training_v1.png",
        simplicity_rank=1,
    ),
    RegimeSpec(
        key="A2",
        label="Resize+JPEG",
        regime_id="mobilenet_resize_jpeg_aug_v1",
        slug="resize_jpeg_aug",
        regime_config_key="A2",
        phase1_ckpt=MODELS_DIR / "mobilenet_resize_jpeg_aug_phase1_best_v1.pt",
        phase2_ckpt=MODELS_DIR / "mobilenet_resize_jpeg_aug_phase2_best_v1.pt",
        selected_ckpt=MODELS_DIR / "mobilenet_resize_jpeg_aug_selected_v1.pt",
        phase1_history=RESULTS_DIR / "rq3_A2_phase1_history_v1.csv",
        phase2_history=RESULTS_DIR / "rq3_A2_phase2_history_v1.csv",
        training_fig=FIGURES_DIR / "rq3_A2_training_v1.png",
        simplicity_rank=2,
    ),
    RegimeSpec(
        key="A3",
        label="Combined",
        regime_id="mobilenet_combined_aug_v1",
        slug="combined_aug",
        regime_config_key="A3",
        phase1_ckpt=MODELS_DIR / "mobilenet_combined_aug_phase1_best_v1.pt",
        phase2_ckpt=MODELS_DIR / "mobilenet_combined_aug_phase2_best_v1.pt",
        selected_ckpt=MODELS_DIR / "mobilenet_combined_aug_selected_v1.pt",
        phase1_history=RESULTS_DIR / "rq3_A3_phase1_history_v1.csv",
        phase2_history=RESULTS_DIR / "rq3_A3_phase2_history_v1.csv",
        training_fig=FIGURES_DIR / "rq3_A3_training_v1.png",
        simplicity_rank=3,
    ),
]


@dataclass
class AugmentationStats:
    total: int = 0
    blur: int = 0
    resize: int = 0
    jpeg: int = 0
    resize_only: int = 0
    jpeg_only: int = 0
    both_resize_jpeg: int = 0
    neither_a2: int = 0
    combo_counts: dict[str, int] = field(default_factory=dict)

    def record(self, applied: dict[str, bool]) -> None:
        self.total += 1
        blur = applied.get("blur", False)
        resize = applied.get("resize", False)
        jpeg = applied.get("jpeg", False)
        if blur:
            self.blur += 1
        if resize:
            self.resize += 1
        if jpeg:
            self.jpeg += 1
        if resize and not jpeg:
            self.resize_only += 1
        if jpeg and not resize:
            self.jpeg_only += 1
        if resize and jpeg:
            self.both_resize_jpeg += 1
        if not resize and not jpeg:
            self.neither_a2 += 1
        key = "+".join(name for name, flag in [("blur", blur), ("resize", resize), ("jpeg", jpeg)] if flag) or "none"
        self.combo_counts[key] = self.combo_counts.get(key, 0) + 1

    def rates(self) -> dict[str, float]:
        if self.total == 0:
            return {}
        out = {
            "total_samples": self.total,
            "blur_rate": self.blur / self.total,
            "resize_rate": self.resize / self.total,
            "jpeg_rate": self.jpeg / self.total,
            "resize_only_rate": self.resize_only / self.total,
            "jpeg_only_rate": self.jpeg_only / self.total,
            "both_resize_jpeg_rate": self.both_resize_jpeg / self.total,
            "neither_a2_rate": self.neither_a2 / self.total,
            "any_augmentation_rate": 1.0 - self.combo_counts.get("none", 0) / self.total,
        }
        for key, count in sorted(self.combo_counts.items()):
            out[f"combo_{key}_rate"] = count / self.total
        return out


class InstrumentedPILTransform(RobustnessAwarePILTransform):
    """Online augmentation with per-sample application logging."""

    def __init__(self, config, stats: AugmentationStats, rng: random.Random | None = None):
        super().__init__(config, rng=rng)
        self.stats = stats

    def __call__(self, image: Image.Image) -> Image.Image:
        out = image.convert("RGB")
        applied = {"blur": False, "resize": False, "jpeg": False}
        for step in self.config.order:
            if step == "blur" and self.config.blur_prob > 0 and self.rng.random() < self.config.blur_prob:
                sigma = self.rng.uniform(self.config.blur_sigma_min, self.config.blur_sigma_max)
                out = self._apply_blur(out, sigma)
                applied["blur"] = True
            elif step == "resize" and self.config.resize_prob > 0 and self.rng.random() < self.config.resize_prob:
                side = self.rng.randint(self.config.resize_side_min, self.config.resize_side_max)
                out = self._apply_resize(out, side)
                applied["resize"] = True
            elif step == "jpeg" and self.config.jpeg_prob > 0 and self.rng.random() < self.config.jpeg_prob:
                quality = self.rng.randint(self.config.jpeg_quality_min, self.config.jpeg_quality_max)
                out = self._apply_jpeg(out, quality)
                applied["jpeg"] = True
        self.stats.record(applied)
        return out

    @staticmethod
    def _apply_blur(image, sigma):
        from rq3_augmentations_v1 import apply_gaussian_blur

        return apply_gaussian_blur(image, sigma)

    @staticmethod
    def _apply_resize(image, side):
        from rq3_augmentations_v1 import apply_resize_degradation

        return apply_resize_degradation(image, side)

    @staticmethod
    def _apply_jpeg(image, quality):
        from rq3_augmentations_v1 import apply_jpeg

        return apply_jpeg(image, quality)


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


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(regime_key: str, stats: AugmentationStats) -> transforms.Compose:
    config = REGIME_CONFIGS[regime_key]
    pil_aug = InstrumentedPILTransform(config, stats=stats, rng=random.Random())
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: pil_aug(img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)))),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


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


def prefer_phase1_on_tie(a: dict, b: dict) -> dict:
    if is_better_robust(a, b) and not is_better_robust(b, a):
        return a
    if is_better_robust(b, a) and not is_better_robust(a, b):
        return b
    return a if a["phase"] == 1 else b


def prefer_primary_candidate(candidates: list[dict]) -> dict:
    """Select one primary RQ3 candidate among A1/A2/A3."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            -c["robust_val_auc"],
            -c["original_auc"],
            -c["robust_val_ap"],
            c["simplicity_rank"],
        ),
    )
    return ordered[0]


def snapshot_bn_running_stats(features: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snaps: list[tuple[torch.Tensor, torch.Tensor]] = []
    for module in features.modules():
        if isinstance(module, nn.BatchNorm2d):
            snaps.append((module.running_mean.detach().cpu().clone(), module.running_var.detach().cpu().clone()))
    return snaps


def bn_stats_unchanged(features: nn.Module, before: list[tuple[torch.Tensor, torch.Tensor]]) -> bool:
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
    model.train()
    model.features.eval()
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
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    stop_if(not bn_stats_unchanged(model.features, bn_before), "Phase 1: BN stats updated")
    for p, before in zip(frozen_params, frozen_before):
        stop_if(not torch.equal(p.detach().cpu(), before), "Phase 1: frozen backbone updated")
    return total_loss / total_samples


def run_phase2_train_epoch(
    model: MobileNetV3SmallBinaryV1,
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
    device: torch.device,
) -> dict:
    eval_transform = build_eval_transform()
    metrics: dict[str, dict[str, float]] = {}
    for condition in SELECTION_CONDITIONS:
        loader = DataLoader(
            PathDataset(condition_frames[condition], eval_transform),
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


@torch.no_grad()
def evaluate_single_condition(
    model: nn.Module,
    rows: pd.DataFrame,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        PathDataset(rows, build_eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    labels, probs, _ = predict_probs(model, loader, device)
    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
    }


def history_row(regime: RegimeSpec, phase: int, epoch: int, train_loss: float, val: dict) -> dict:
    row = {
        "regime": regime.key,
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
    return row


def save_checkpoint(
    path: Path,
    model: MobileNetV3SmallBinaryV1,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: int,
    regime: RegimeSpec,
    val: dict,
    config: dict,
    total_params: int,
    trainable_params: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": regime.regime_id,
        "regime": regime.key,
        "phase": phase,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "validation_metrics": val,
        "robust_val_auc": val["robust_val_auc"],
        "robust_val_ap": val["robust_val_ap"],
        "condition_aucs": {c: val["conditions"][c]["roc_auc"] for c in SELECTION_CONDITIONS},
        "condition_aps": {c: val["conditions"][c]["ap"] for c in SELECTION_CONDITIONS},
        "configuration": config,
        "augmentation_config": REGIME_CONFIGS[regime.regime_config_key].__dict__,
        "seed": RANDOM_SEED,
        "pretrained_weights": WEIGHTS_ENUM,
        "initialization_policy": "ImageNet-pretrained MobileNetV3-Small; NOT from mobilenet_v3_small_selected_v1.pt",
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "representation": REPRESENTATION,
        "split_protocol": SPLIT_PROTOCOL,
    }
    torch.save(payload, path)


def plot_regime_training(regime: RegimeSpec, phase1_df: pd.DataFrame, phase2_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(phase1_df["epoch"], phase1_df["train_loss"], "o-", label="Phase 1 train")
    ax.plot(phase2_df["epoch"], phase2_df["train_loss"], "s--", label="Phase 2 train")
    ax.set_xlabel("Epoch within phase")
    ax.set_ylabel("Train loss")
    ax.set_title(f"{regime.key} training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(phase1_df["epoch"], phase1_df["robust_val_auc"], "o-", label="Phase 1 RobustValAUC")
    ax.plot(phase2_df["epoch"], phase2_df["robust_val_auc"], "s--", label="Phase 2 RobustValAUC")
    ax.set_xlabel("Epoch within phase")
    ax.set_ylabel("RobustValAUC")
    ax.set_title(f"{regime.key} robust validation score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    regime.training_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(regime.training_fig, dpi=150)
    plt.close(fig)


def load_a0_summary() -> dict:
    stop_if(not A0_METRICS_PATH.exists(), f"missing A0 metrics: {A0_METRICS_PATH}")
    df = pd.read_csv(A0_METRICS_PATH)
    stop_if(len(df[df["regime"] == "A0"]) == 0, "A0 metrics missing")
    a0 = df[df["regime"] == "A0"]
    out = {"regime": "A0", "label": "Clean", "selected_phase": "pretrained+clean", "selected_epoch": "frozen"}
    mapping = {
        "original": "original_auc",
        "jpeg_q50": "jpeg50_auc",
        "resize_112": "resize112_auc",
        "blur_sigma2": "blur2_auc",
        SCREENSHOT_CONDITION: "screenshot_strong_auc",
    }
    ap_mapping = {
        "original": "original_ap",
        "jpeg_q50": "jpeg50_ap",
        "resize_112": "resize112_ap",
        "blur_sigma2": "blur2_ap",
        SCREENSHOT_CONDITION: "screenshot_strong_ap",
    }
    for condition, col in mapping.items():
        row = a0[a0["condition"] == condition].iloc[0]
        out[col] = float(row["roc_auc"])
        out[ap_mapping[condition]] = float(row["average_precision"])
    sel = [out["original_auc"], out["jpeg50_auc"], out["resize112_auc"], out["blur2_auc"]]
    out["robust_val_auc"] = float(np.mean(sel))
    out["robust_val_ap"] = float(np.mean([out["original_ap"], out["jpeg50_ap"], out["resize112_ap"], out["blur2_ap"]]))
    out["parameters"] = 1518881
    return out


def train_regime(
    regime: RegimeSpec,
    train_loader: DataLoader,
    condition_frames: dict[str, pd.DataFrame],
    device: torch.device,
) -> tuple[dict, AugmentationStats]:
    print("\n" + "=" * 70)
    print(f"TRAINING {regime.key} — {regime.regime_id}")
    print("=" * 70)
    set_seed(RANDOM_SEED)
    aug_stats = AugmentationStats()

    train_ds = ControlledV1Dataset("train", transform=build_train_transform(regime.regime_config_key, aug_stats))
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=g,
    )

    criterion = nn.BCEWithLogitsLoss()
    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS)
    model.freeze_features()
    total_params, phase1_trainable = count_parameters(model)
    model = model.to(device)

    phase1_config = {
        "phase": 1,
        "epochs": PHASE1_EPOCHS,
        "lr": PHASE1_LR,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW(classifier)",
        "freezing_configuration": "features frozen; classifier trainable; features.eval() during train",
        "augmentation_regime": regime.regime_config_key,
        "selection_metric": "RobustValAUC",
    }
    optimizer1 = torch.optim.AdamW(model.classifier.parameters(), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

    phase1_history: list[dict] = []
    best_phase1: dict | None = None
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = run_phase1_train_epoch(model, train_loader, criterion, optimizer1, device)
        val = evaluate_robust_validation(model, condition_frames, device)
        row = history_row(regime, 1, epoch, train_loss, val)
        phase1_history.append(row)
        print(
            f"{regime.key} P1 E{epoch:02d} | loss={train_loss:.4f} | "
            f"RobustValAUC={val['robust_val_auc']:.4f} | orig={val['original_auc']:.4f}"
        )
        candidate = {"phase": 1, "epoch": epoch, **val, "train_loss": train_loss}
        if is_better_robust(candidate, best_phase1):
            best_phase1 = candidate
            save_checkpoint(
                regime.phase1_ckpt,
                model,
                optimizer1,
                epoch,
                1,
                regime,
                val,
                phase1_config,
                total_params,
                phase1_trainable,
            )
            print(f"  → saved Phase-1 best (epoch {epoch})")

    phase1_df = pd.DataFrame(phase1_history)
    phase1_df.to_csv(regime.phase1_history, index=False)
    stop_if(best_phase1 is None, f"{regime.key} Phase 1 produced no checkpoint")

    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS)
    ckpt = torch.load(regime.phase1_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.unfreeze_all()
    _, phase2_trainable = count_parameters(model)
    model = model.to(device)
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)

    phase2_config = {
        **phase1_config,
        "phase": 2,
        "epochs": PHASE2_EPOCHS,
        "lr": PHASE2_LR,
        "optimizer": "AdamW(all parameters)",
        "started_from": str(regime.phase1_ckpt.relative_to(PROJECT_ROOT)),
        "started_from_phase1_epoch": int(ckpt["epoch"]),
        "freezing_configuration": "all parameters trainable; model.train()",
    }

    phase2_history: list[dict] = []
    best_phase2: dict | None = None
    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = run_phase2_train_epoch(model, train_loader, criterion, optimizer2, device)
        val = evaluate_robust_validation(model, condition_frames, device)
        row = history_row(regime, 2, epoch, train_loss, val)
        phase2_history.append(row)
        print(
            f"{regime.key} P2 E{epoch:02d} | loss={train_loss:.4f} | "
            f"RobustValAUC={val['robust_val_auc']:.4f} | orig={val['original_auc']:.4f}"
        )
        candidate = {"phase": 2, "epoch": epoch, **val, "train_loss": train_loss}
        if is_better_robust(candidate, best_phase2):
            best_phase2 = candidate
            save_checkpoint(
                regime.phase2_ckpt,
                model,
                optimizer2,
                epoch,
                2,
                regime,
                val,
                phase2_config,
                total_params,
                phase2_trainable,
            )
            print(f"  → saved Phase-2 best (epoch {epoch})")

    phase2_df = pd.DataFrame(phase2_history)
    phase2_df.to_csv(regime.phase2_history, index=False)
    stop_if(best_phase2 is None, f"{regime.key} Phase 2 produced no checkpoint")

    selected_meta = prefer_phase1_on_tie(best_phase1, best_phase2)
    src_ckpt = regime.phase1_ckpt if selected_meta["phase"] == 1 else regime.phase2_ckpt
    selected_payload = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    selected_payload["final_selection"] = {
        "selected_phase": selected_meta["phase"],
        "selected_epoch": selected_meta["epoch"],
        "selection_rule": "RobustValAUC → original AUC → RobustValAP → earlier epoch; Phase 1 on tie",
        "phase1_best_robust_val_auc": best_phase1["robust_val_auc"],
        "phase2_best_robust_val_auc": best_phase2["robust_val_auc"],
        "screenshot_used_for_selection": False,
        "test_accessed": False,
    }
    torch.save(selected_payload, regime.selected_ckpt)

    plot_regime_training(regime, phase1_df, phase2_df)

    model = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS).to(device)
    model.load_state_dict(selected_payload["model_state_dict"])
    screenshot = evaluate_single_condition(model, condition_frames[SCREENSHOT_CONDITION], device)

    result = {
        "regime": regime.key,
        "label": regime.label,
        "regime_id": regime.regime_id,
        "selected_phase": int(selected_meta["phase"]),
        "selected_epoch": int(selected_meta["epoch"]),
        "original_auc": float(selected_meta["original_auc"]),
        "jpeg50_auc": float(selected_meta["conditions"]["jpeg_q50"]["roc_auc"]),
        "resize112_auc": float(selected_meta["conditions"]["resize_112"]["roc_auc"]),
        "blur2_auc": float(selected_meta["conditions"]["blur_sigma2"]["roc_auc"]),
        "robust_val_auc": float(selected_meta["robust_val_auc"]),
        "robust_val_ap": float(selected_meta["robust_val_ap"]),
        "screenshot_strong_auc": float(screenshot["roc_auc"]),
        "screenshot_strong_ap": float(screenshot["ap"]),
        "parameters": total_params,
        "simplicity_rank": regime.simplicity_rank,
        "phase1_best": best_phase1,
        "phase2_best": best_phase2,
        "augmentation_rates": aug_stats.rates(),
    }
    return result, aug_stats


def plot_summary_figures(summary_rows: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["regime"] for r in summary_rows]
    robust = [r["robust_val_auc"] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, robust, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(labels)])
    ax.set_ylabel("RobustValAUC")
    ax.set_title("RQ3 validation RobustValAUC by regime")
    ax.set_ylim(0.0, 1.0)
    for i, v in enumerate(robust):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROBUST_AUC_FIG, dpi=150)
    plt.close(fig)

    conditions = ["Original", "JPEG50", "Resize112", "Blur2", "ScreenshotStrong"]
    keys = ["original_auc", "jpeg50_auc", "resize112_auc", "blur2_auc", "screenshot_strong_auc"]
    x = np.arange(len(conditions))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(summary_rows):
        vals = [row[k] for k in keys]
        ax.bar(x + (i - (len(summary_rows) - 1) / 2) * width, vals, width, label=row["regime"])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("RQ3 selected-checkpoint validation AUC by condition")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONDITION_AUC_FIG, dpi=150)
    plt.close(fig)


def write_report(
    a0: dict,
    results: list[dict],
    primary: dict,
    integrity: dict,
) -> None:
    lines = [
        "RQ3 Transformation-Aware Training Report — Stage 23B",
        "====================================================",
        "",
        "1. PROTOCOL",
        f"Base model: MobileNetV3-Small ({WEIGHTS_ENUM})",
        f"Train={EXPECTED_TRAIN}, validation={EXPECTED_VAL}, test access=NO",
        "Phase 1: 10 epochs, frozen features, AdamW lr=1e-3",
        "Phase 2: 20 epochs, full fine-tune, AdamW lr=1e-4",
        "Selection metric: RobustValAUC over original/jpeg50/resize112/blur2",
        "",
        "2. FAIR INITIALIZATION",
        "A1/A2/A3 each started from fresh ImageNet-pretrained MobileNetV3-Small.",
        "NOT initialized from mobilenet_v3_small_selected_v1.pt or from each other.",
        f"Seed reset to {RANDOM_SEED} before each regime.",
        "",
    ]
    for res in results:
        lines.extend(
            [
                f"{res['regime']} RESULTS — {res['label']}",
                f"Selected phase: {res['selected_phase']}, epoch: {res['selected_epoch']}",
                f"Original AUC={res['original_auc']:.8f}",
                f"JPEG50 AUC={res['jpeg50_auc']:.8f}",
                f"Resize112 AUC={res['resize112_auc']:.8f}",
                f"Blur2 AUC={res['blur2_auc']:.8f}",
                f"RobustValAUC={res['robust_val_auc']:.8f}",
                f"ScreenshotStrong AUC={res['screenshot_strong_auc']:.8f} (held-out; not used for selection)",
                "",
            ]
        )
    lines.extend(
        [
            "6. PHASE-1 VS PHASE-2",
        ]
    )
    for res in results:
        lines.append(
            f"{res['regime']}: P1 RobustValAUC={res['phase1_best']['robust_val_auc']:.6f}, "
            f"P2 RobustValAUC={res['phase2_best']['robust_val_auc']:.6f}, selected phase={res['selected_phase']}"
        )
    lines.extend(["", "7. ROBUST VALIDATION COMPARISON", f"A0 RobustValAUC={a0['robust_val_auc']:.8f}"])
    for res in results:
        delta = res["robust_val_auc"] - a0["robust_val_auc"]
        lines.append(f"{res['regime']} ΔRobustValAUC vs A0={delta:+.8f}")
    lines.extend(["", "8. SCREENSHOT HELD-OUT VALIDATION OBSERVATION"])
    for res in results:
        lines.append(f"{res['regime']} screenshot_strong AUC={res['screenshot_strong_auc']:.8f}")
    lines.extend(
        [
            "",
            "9. PRIMARY VALIDATION-SELECTED RQ3 CANDIDATE",
            f"{primary['regime']} ({primary.get('regime_id', primary.get('label'))})",
            f"RobustValAUC={primary['robust_val_auc']:.8f}",
            "",
            "10. AUGMENTATION APPLICATION SANITY",
        ]
    )
    for res in results:
        rates = res["augmentation_rates"]
        lines.append(f"{res['regime']}: {json.dumps(rates, indent=2)}")
    lines.extend(
        [
            "",
            "11. LIMITATIONS",
            "Validation-only development; no test evaluation.",
            "RQ3 is sequential to RQ2; not independent confirmatory evidence.",
            "",
            "12. SCIENTIFIC INTEGRITY",
        ]
    )
    for key, val in integrity.items():
        lines.append(f"{key}: {val}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_seed(RANDOM_SEED)
    device = select_device()
    print("=" * 70)
    print("STAGE 23B — RQ3 TRANSFORMATION-AWARE MOBILENET TRAINING")
    print("=" * 70)
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | device {device}")

    meta = pd.read_csv(SPLIT_META_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    train_meta = load_split_metadata("train", SPLIT_META_PATH)
    val_meta = load_split_metadata("validation", SPLIT_META_PATH)
    stop_if(len(train_meta) != EXPECTED_TRAIN, f"train={len(train_meta)}")
    stop_if(len(val_meta) != EXPECTED_VAL, f"validation={len(val_meta)}")
    stop_if(len(manifest) != 1824, f"manifest rows {len(manifest)}")
    stop_if((manifest["split"] != "validation").any(), "non-validation manifest rows")

    condition_frames = build_condition_frames(meta, manifest)
    a0 = load_a0_summary()

    # Placeholder loader replaced inside each regime with augmented dataset.
    dummy_loader = DataLoader(
        ControlledV1Dataset("train", transform=build_eval_transform()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    results: list[dict] = []
    all_aug_stats: dict[str, AugmentationStats] = {}
    for regime in REGIMES:
        result, stats = train_regime(regime, dummy_loader, condition_frames, device)
        results.append(result)
        all_aug_stats[regime.key] = stats

    summary_rows = [a0] + [
        {
            "regime": r["regime"],
            "label": r["label"],
            "selected_phase": r["selected_phase"],
            "selected_epoch": r["selected_epoch"],
            "original_auc": r["original_auc"],
            "jpeg50_auc": r["jpeg50_auc"],
            "resize112_auc": r["resize112_auc"],
            "blur2_auc": r["blur2_auc"],
            "robust_val_auc": r["robust_val_auc"],
            "robust_val_ap": r["robust_val_ap"],
            "screenshot_strong_auc": r["screenshot_strong_auc"],
            "screenshot_strong_ap": r["screenshot_strong_ap"],
            "parameters": r["parameters"],
        }
        for r in results
    ]
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    primary = prefer_primary_candidate(results)
    plot_summary_figures(summary_rows)

    integrity = {
        "A0 retrained": "NO",
        "A1/A2/A3 trained from same ImageNet-pretrained starting protocol": "YES",
        "Test images accessed": "NO",
        "Test transformed predictions accessed for selection": "NO",
        "Test thresholds changed": "NO",
        "Screenshot validation used for checkpoint selection": "NO",
        "Training budgets extended": "NO",
        "Additional augmentation regimes added": "NO",
        "RQ1 model development reopened": "NO",
    }
    write_report(a0, results, primary, integrity)

    print("\n" + "=" * 70)
    print("STAGE 23B — RQ3 TRANSFORMATION-AWARE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nA0 CLEAN\nRobustValAUC: {a0['robust_val_auc']:.6f}")
    for res in results:
        print(f"\n{res['regime']} {res['label'].upper()}")
        print(f"Selected phase: {res['selected_phase']}")
        print(f"Selected epoch: {res['selected_epoch']}")
        print(f"Original AUC: {res['original_auc']:.6f}")
        print(f"JPEG50 AUC: {res['jpeg50_auc']:.6f}")
        print(f"Resize112 AUC: {res['resize112_auc']:.6f}")
        print(f"Blur2 AUC: {res['blur2_auc']:.6f}")
        print(f"RobustValAUC: {res['robust_val_auc']:.6f}")
        print(f"ScreenshotStrong AUC: {res['screenshot_strong_auc']:.6f}")
    delta = primary["robust_val_auc"] - a0["robust_val_auc"]
    print(f"\nPRIMARY VALIDATION-SELECTED CANDIDATE:\n{primary['regime']} ({primary['regime_id']})")
    print(f"Delta RobustValAUC vs A0: {delta:+.6f}")
    print("\nA0 retrained: NO")
    print("Test access: NO")
    print("Threshold selection: NO")
    print("Training budget extension: NO")
    print("\nRQ3 TRAINING COMPLETE")
    print("STOP BEFORE THRESHOLD SELECTION OR TEST EVALUATION.")


if __name__ == "__main__":
    main()
