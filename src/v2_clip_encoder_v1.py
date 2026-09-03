"""V2 frozen CLIP image encoder (open_clip ViT-B scale).

Infrastructure only — no classifier training, no fine-tuning, no NTIRE access.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps

from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

DEFAULT_MODEL_NAME = "ViT-B-16-quickgelu"  # OpenAI ViT-B/16 weights require QuickGELU
DEFAULT_PRETRAINED = "openai"
FALLBACK_MODEL_NAME = "ViT-B-32-quickgelu"


@dataclass(frozen=True)
class V2ClipModelMeta:
    library: str
    library_version: str
    model_name: str
    pretrained_tag: str
    weight_source: str
    embedding_dim: int
    input_resolution: int
    parameter_count: int
    device: str
    l2_normalize: bool


def resolve_device(device: str = "auto") -> torch.device:
    d = device.lower().strip()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    if d == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if d == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device}")


def load_rgb_image(path: str | Path) -> Image.Image:
    """Native decode → EXIF orientation → RGB. No V1 controlled_v1 JPEG path.

    Corrupt EXIF metadata falls back to untransposed RGB (image pixels kept).
    """
    assert_path_not_final_external_test(str(path))
    with Image.open(path) as im:
        try:
            transposed = ImageOps.exif_transpose(im)
            if transposed is not None:
                im = transposed
        except Exception:
            # Keep pixels; skip orientation if EXIF IFD is malformed.
            pass
        return im.convert("RGB")


class V2ClipEncoderV1:
    """Frozen pretrained CLIP image encoder for V2 development representations."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str = "auto",
        l2_normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.l2_normalize = l2_normalize
        self.device = resolve_device(device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)

        # Resolve embedding dim from a dummy forward is deferred; use visual output_dim
        emb_dim = int(getattr(self.model, "visual").output_dim)
        # open_clip stores image size on visual.image_size (int or tuple)
        img_size = getattr(self.model.visual, "image_size", 224)
        if isinstance(img_size, (tuple, list)):
            input_resolution = int(img_size[0])
        else:
            input_resolution = int(img_size)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.meta = V2ClipModelMeta(
            library="open_clip_torch",
            library_version=open_clip.__version__,
            model_name=model_name,
            pretrained_tag=pretrained,
            weight_source=f"open_clip.create_model_and_transforms({model_name!r}, pretrained={pretrained!r})",
            embedding_dim=emb_dim,
            input_resolution=input_resolution,
            parameter_count=n_params,
            device=str(self.device),
            l2_normalize=l2_normalize,
        )

    def metadata_dict(self) -> dict:
        return asdict(self.meta)

    def preprocess_paths(self, paths: Sequence[str | Path]) -> torch.Tensor:
        tensors = []
        for p in paths:
            img = load_rgb_image(p)
            tensors.append(self.preprocess(img))
        return torch.stack(tensors, dim=0)

    @torch.inference_mode()
    def encode_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Encode preprocessed batch [B,3,H,W] → [B,D] (optionally L2-normalized)."""
        images = images.to(self.device, non_blocking=False)
        feats = self.model.encode_image(images)
        feats = feats.float()
        if self.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def encode_paths(
        self,
        paths: Sequence[str | Path],
        batch_size: int = 16,
        return_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        outs = []
        paths = list(paths)
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            x = self.preprocess_paths(batch_paths)
            feats = self.encode_tensor(x)
            outs.append(feats.detach().cpu())
        emb = torch.cat(outs, dim=0) if outs else torch.empty((0, self.meta.embedding_dim))
        if return_numpy:
            return emb.numpy()
        return emb
