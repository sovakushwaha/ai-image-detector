"""Frozen H4 ensemble inference for the investor demo.

Matches V2-11 preprocessing / R1 / equal-weight fold mean / threshold 0.5.
Does not modify scientific scripts or frozen checkpoints.
Loads one fold at a time to limit Mac memory pressure.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import FOLDS, MODELS_DIR, PROJECT_ROOT, SRC_DIR, THRESHOLD
from .integrity import IntegrityResult, verify_freeze_manifest

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_v2_10a_representation_geometry_v1 import ClipLoRAModel  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _agreement_label(std: float) -> str:
    # Descriptive only — not calibrated confidence.
    if std < 0.05:
        return "High agreement across folds"
    if std < 0.12:
        return "Moderate agreement across folds"
    if std < 0.20:
        return "Mixed fold signals"
    return "High fold dispersion"


@dataclass
class EnsembleService:
    device: torch.device
    integrity: IntegrityResult
    _lock: threading.Lock
    ready: bool = False
    load_error: str | None = None

    @classmethod
    def create(cls) -> "EnsembleService":
        integrity = verify_freeze_manifest()
        device = pick_device()
        svc = cls(device=device, integrity=integrity, _lock=threading.Lock())
        if not integrity.verified:
            svc.load_error = (
                "Frozen model integrity check failed. Inference is refused. "
                f"Status={integrity.status}"
            )
            svc.ready = False
        else:
            svc.ready = True
        return svc

    def model_info(self) -> dict[str, Any]:
        return {
            "brand": "SOVA VERIFY",
            "model_name": "FINAL_RESEARCH_MODEL_V2",
            "version": "H4",
            "label": "H4 frozen research ensemble",
            "freeze_status": "FROZEN_WITH_DOCUMENTED_LIMITATIONS",
            "integrity_status": self.integrity.status,
            "integrity_verified": self.integrity.verified,
            "threshold": THRESHOLD,
            "n_folds": len(FOLDS),
            "aggregation": "equal_weight_arithmetic_mean",
            "fold_weights": [0.25, 0.25, 0.25, 0.25],
            "calibration": "none",
            "representation": "LoRA R1 512-d L2-normalized",
            "backbone": "open_clip ViT-B-16-quickgelu / openai",
            "device": str(self.device),
            "ready": self.ready and self.integrity.verified,
            "research_prototype": True,
            "score_semantics": (
                "Uncalibrated AI Detection Score in [0,1]. "
                "Not a verified authenticity probability."
            ),
            "manifest": self.integrity.manifest,
            "project_root": str(PROJECT_ROOT),
        }

    def _load_fold(self, fold: int) -> tuple[Any, Any]:
        ckpt_path = MODELS_DIR / f"clip_lora_fold{fold}_best_v1.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        cfg.setdefault("clip", {})
        cfg["clip"].setdefault("l2_normalize", True)
        if isinstance(cfg.get("head"), str):
            cfg["head"] = {"dropout": 0.2}
        cfg.setdefault("head", {"dropout": 0.2})
        model = ClipLoRAModel(cfg, self.device)
        incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
        crit = [k for k in incompat.missing_keys if "lora_" in k or k.startswith("head.")]
        if crit:
            raise RuntimeError(f"Fold {fold} critical missing keys: {crit[:5]}")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        h4 = joblib.load(MODELS_DIR / f"v2_10h_h4_clean_anchored_robust_logreg_fold{fold}_v1.joblib")
        return model, h4

    def _prepare_tensor(self, image: Image.Image, preprocess) -> torch.Tensor:
        try:
            t = ImageOps.exif_transpose(image)
            if t is not None:
                image = t
        except Exception:
            pass
        rgb = image.convert("RGB")
        return preprocess(rgb).unsqueeze(0)

    @torch.no_grad()
    def _score_fold(self, fold: int, image: Image.Image) -> float:
        model, h4 = self._load_fold(fold)
        try:
            xb = self._prepare_tensor(image, model.preprocess).to(self.device)
            r1 = model.encode(xb).detach().cpu().numpy().astype(np.float32)
            cls1 = list(h4.classes_).index(1)
            p = float(h4.predict_proba(r1)[0, cls1])
            if not np.isfinite(p):
                raise RuntimeError(f"Non-finite score from fold {fold}")
            return float(np.clip(p, 0.0, 1.0))
        finally:
            del model
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

    def analyze_pil(self, image: Image.Image, meta: dict[str, Any]) -> dict[str, Any]:
        if not self.integrity.verified:
            raise PermissionError(self.load_error or "Model integrity failed")
        if not self.ready:
            raise RuntimeError(self.load_error or "Model service not ready")

        with self._lock:
            t0 = time.perf_counter()
            fold_scores: dict[str, float] = {}
            for fold in FOLDS:
                fold_scores[f"fold_{fold}"] = self._score_fold(fold, image)
            scores = np.array([fold_scores[f"fold_{f}"] for f in FOLDS], dtype=np.float64)
            ai_score = float(scores.mean())
            fold_std = float(scores.std(ddof=0))
            classification = "AI-like" if ai_score >= THRESHOLD else "Real-like"
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "ok": True,
            "image": meta,
            "ai_detection_score": ai_score,
            "classification": classification,
            "threshold": THRESHOLD,
            "fold_scores": {
                "fold_1": fold_scores["fold_1"],
                "fold_2": fold_scores["fold_2"],
                "fold_3": fold_scores["fold_3"],
                "fold_4": fold_scores["fold_4"],
            },
            "fold_mean": ai_score,
            "fold_dispersion": fold_std,
            "ensemble_agreement": _agreement_label(fold_std),
            "processing_ms": elapsed_ms,
            "model_version": "H4",
            "model_name": "FINAL_RESEARCH_MODEL_V2",
            "integrity_status": self.integrity.status,
            "integrity_verified": True,
            "device": str(self.device),
            "calibration": "none",
            "aggregation": "equal_weight_arithmetic_mean",
            "research_prototype": True,
            "warning": "A low AI score does not prove that an image is authentic.",
            "score_note": (
                "AI Detection Score is an uncalibrated ensemble output, "
                "not a verified authenticity probability."
            ),
        }


def open_image_bytes(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.copy()
    except UnidentifiedImageError as e:
        raise ValueError("Corrupt or unreadable image") from e
    except Exception as e:
        raise ValueError("Corrupt or unreadable image") from e
