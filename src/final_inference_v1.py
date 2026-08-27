"""FINAL_RESEARCH_MODEL_V1 local inference engine (Stage 28A).

Frozen research detector for single-image prediction. No training, no
recalibration, no API calls, no automatic prediction history.

Preprocessing note
------------------
Research dataset preparation wrote controlled_v1 JPEG q96 files after
shortest-side resize 256 + centre crop 224. That JPEG step was
dataset-standardisation only.

Practical local inference operates from decoded image pixels: EXIF
orientation correction → RGB → shortest-side resize 256 (LANCZOS) →
centre crop 224×224 → ToTensor → ImageNet normalisation. It does NOT
re-encode to JPEG q96 before the model.

Reproduction against saved RQ3 A2 predictions uses the already-prepared
controlled_v1 224×224 files with tensor normalisation only, matching the
frozen research evaluation path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from mobilenet_v3_small_binary_v1 import MobileNetV3SmallBinaryV1, count_parameters

WARNING = (
    "Experimental research detector. Independent external validation showed "
    "substantial degradation on modern generators; this prediction must not be "
    "treated as definitive evidence of image authenticity."
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_SHORT_SIDE = 256
FINAL_SIZE = 224
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class InferenceResultV1:
    model_id: str
    image_path: str
    device: str
    raw_logit: float
    raw_probability: float
    calibrated_probability: float
    selective_decision: str
    historical_binary_diagnostic: str
    historical_binary_threshold: float
    temperature: float
    real_boundary: float
    ai_boundary: float
    warning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(preference: str = "auto") -> torch.device:
    pref = (preference or "auto").lower().strip()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {preference!r} (use auto|cpu|mps|cuda)")


def _resize_shortest_side(image: Image.Image, target: int) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {image.size}")
    if w <= h:
        new_w = target
        new_h = int(round(h * (target / w)))
    else:
        new_h = target
        new_w = int(round(w * (target / h)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _centre_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    if w < size or h < size:
        raise ValueError(f"Cannot crop {image.size} to {size}x{size}")
    left = (w - size) // 2
    top = (h - size) // 2
    return image.crop((left, top, left + size, top + size))


class FinalImageDetectorV1:
    """Frozen FINAL_RESEARCH_MODEL_V1 single-image detector."""

    def __init__(
        self,
        project_root: Path | str | None = None,
        device: str = "auto",
    ) -> None:
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.device = resolve_device(device)
        self.warning = WARNING

        pointer_path = self.project_root / "models" / "FINAL_MODEL_V1.json"
        if not pointer_path.exists():
            raise FileNotFoundError(f"Missing final model pointer: {pointer_path}")
        self.pointer = json.loads(pointer_path.read_text())

        selection_path = self.project_root / self.pointer.get(
            "selection_config", "results/final_model_selection_v1.json"
        )
        if not selection_path.exists():
            raise FileNotFoundError(f"Missing selection config: {selection_path}")
        self.selection = json.loads(selection_path.read_text())

        ckpt_rel = self.pointer["checkpoint"]
        self.checkpoint_path = self.project_root / ckpt_rel
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")

        temp_path = self.project_root / self.pointer["temperature_config"]
        policy_path = self.project_root / self.pointer["selective_policy_config"]
        frozen_path = self.project_root / self.pointer["frozen_config"]
        for p in (temp_path, policy_path, frozen_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing frozen config: {p}")

        self.temperature_config = json.loads(temp_path.read_text())
        self.selective_policy = json.loads(policy_path.read_text())
        self.frozen_config = json.loads(frozen_path.read_text())

        self.temperature = float(self.temperature_config["temperature"])
        self.lower80 = float(self.selective_policy["lower80"])
        self.upper80 = float(self.selective_policy["upper80"])
        self.historical_threshold = float(self.frozen_config["threshold"])
        self.model_id = str(self.pointer.get("final_model_id", "FINAL_RESEARCH_MODEL_V1"))
        self.selected_candidate = str(self.pointer.get("selected_candidate", "C0"))

        expected_ckpt = "models/mobilenet_resize_jpeg_aug_selected_v1.pt"
        if ckpt_rel != expected_ckpt:
            raise RuntimeError(
                f"Checkpoint path mismatch: pointer has {ckpt_rel!r}, expected {expected_ckpt!r}"
            )
        if self.frozen_config.get("checkpoint") not in (None, expected_ckpt, ckpt_rel):
            # tolerate absolute/relative consistency only when present
            pass
        if self.temperature_config.get("checkpoint") not in (None, expected_ckpt, ckpt_rel):
            pass

        self.model = MobileNetV3SmallBinaryV1()
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            self.model.load_state_dict(state["state_dict"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(self.device)
        self.n_parameters = int(count_parameters(self.model)[0])

        expected_params = int(self.frozen_config.get("total_parameters", 1518881))
        if self.n_parameters != expected_params:
            raise RuntimeError(
                f"Parameter count mismatch: model has {self.n_parameters}, "
                f"frozen config expects {expected_params}"
            )

        self._tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def decode_image(self, path: Path | str) -> Image.Image:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported image format {suffix!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )
        try:
            with Image.open(path) as im:
                im.load()
                oriented = ImageOps.exif_transpose(im)
                return oriented.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(f"Unreadable image file: {path}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to decode image: {path} ({exc})") from exc

    def preprocess_practical(self, image: Image.Image) -> Image.Image:
        """Practical path: decoded pixels → resize 256 → centre crop 224 (no JPEG)."""
        rgb = image.convert("RGB")
        resized = _resize_shortest_side(rgb, RESIZE_SHORT_SIDE)
        return _centre_crop(resized, FINAL_SIZE)

    def preprocess_research_controlled_v1(self, image: Image.Image) -> Image.Image:
        """Research reproduction path for already-controlled 224×224 RGB images."""
        rgb = image.convert("RGB")
        if rgb.size != (FINAL_SIZE, FINAL_SIZE):
            raise ValueError(
                f"Research controlled_v1 mode requires {FINAL_SIZE}x{FINAL_SIZE}, got {rgb.size}"
            )
        return rgb

    def _forward(self, rgb_224: Image.Image) -> tuple[float, float, float]:
        tensor = self._tensor_transform(rgb_224).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logit = self.model(tensor).squeeze().float().cpu().item()
        raw_p = float(1.0 / (1.0 + torch.exp(torch.tensor(-logit)).item()))
        cal_p = float(1.0 / (1.0 + torch.exp(torch.tensor(-logit / self.temperature)).item()))
        return float(logit), raw_p, cal_p

    def selective_decision(self, calibrated_p: float) -> str:
        if calibrated_p <= self.lower80:
            return "REAL"
        if calibrated_p >= self.upper80:
            return "AI-GENERATED"
        return "UNCERTAIN"

    def historical_binary_diagnostic(self, raw_p: float) -> str:
        return "AI" if raw_p >= self.historical_threshold else "REAL"

    def predict(
        self,
        path: Path | str,
        *,
        research_controlled_v1: bool = False,
    ) -> InferenceResultV1:
        path = Path(path)
        rgb = self.decode_image(path)
        if research_controlled_v1:
            crop = self.preprocess_research_controlled_v1(rgb)
        else:
            crop = self.preprocess_practical(rgb)
        logit, raw_p, cal_p = self._forward(crop)
        return InferenceResultV1(
            model_id=self.model_id,
            image_path=str(path),
            device=str(self.device),
            raw_logit=logit,
            raw_probability=raw_p,
            calibrated_probability=cal_p,
            selective_decision=self.selective_decision(cal_p),
            historical_binary_diagnostic=self.historical_binary_diagnostic(raw_p),
            historical_binary_threshold=self.historical_threshold,
            temperature=self.temperature,
            real_boundary=self.lower80,
            ai_boundary=self.upper80,
            warning=self.warning,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "selected_candidate": self.selected_candidate,
            "checkpoint": str(self.checkpoint_path.relative_to(self.project_root)),
            "n_parameters": self.n_parameters,
            "temperature": self.temperature,
            "real_boundary": self.lower80,
            "ai_boundary": self.upper80,
            "uncertain_interval": [self.lower80, self.upper80],
            "historical_binary_threshold": self.historical_threshold,
            "device": str(self.device),
            "warning": self.warning,
            "preprocessing_note": (
                "Practical inference uses decoded pixels with resize-256 + centre-crop-224; "
                "does not re-encode JPEG q96. controlled_v1 JPEG q96 was research dataset "
                "standardisation only."
            ),
        }
