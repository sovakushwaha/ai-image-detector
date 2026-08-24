"""RQ4 F2 RGB + frequency fusion model (Stage 24B.1).

Why this file exists
--------------------
Fuses 1024-d MobileNet RGB embedding (from RQ3 A2) with 128-d F1 frequency
embedding. Shared-augmentation pairing is enforced by the training dataset,
not this module.

F2 is the PRIMARY RQ4 intervention.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mobilenet_v3_small_binary_v1 import DEFAULT_WEIGHTS, MobileNetV3SmallBinaryV1
from rq4_frequency_cnn_v1 import FrequencyOnlyCNNV1, count_parameters


class RGBFrequencyFusionV1(nn.Module):
    """F2: RGB MobileNet branch + F1 frequency branch + fusion head → raw logit."""

    def __init__(self) -> None:
        super().__init__()
        self.rgb_branch = MobileNetV3SmallBinaryV1(weights=DEFAULT_WEIGHTS)
        self.freq_branch = FrequencyOnlyCNNV1()
        self.fusion_head = nn.Sequential(
            nn.Linear(1024 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, 1),
        )
        self.rgb_embedding_dim = 1024
        self.freq_embedding_dim = 128

    def load_rgb_from_a2(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.rgb_branch.load_state_dict(ckpt["model_state_dict"])

    def load_freq_from_f1(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.freq_branch.load_state_dict(ckpt["model_state_dict"])

    def forward_rgb_features(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """1024-d penultimate RGB embedding (Linear 576→1024 + Hardswish)."""
        x = self.rgb_branch.backbone.features(x_rgb)
        x = self.rgb_branch.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.rgb_branch.backbone.classifier[0](x)
        x = self.rgb_branch.backbone.classifier[1](x)
        return x

    def forward_freq_features(self, x_freq: torch.Tensor) -> torch.Tensor:
        """128-d frequency embedding (bypass F1 final classifier)."""
        return self.freq_branch.forward_features(x_freq)

    def forward(self, x_rgb: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        """Return raw logits [B]. No sigmoid."""
        rgb_emb = self.forward_rgb_features(x_rgb)
        freq_emb = self.forward_freq_features(x_freq)
        fused = torch.cat([rgb_emb, freq_emb], dim=1)
        logits = self.fusion_head(fused)
        return logits.squeeze(1)

    def freeze_branches(self) -> None:
        for p in self.rgb_branch.parameters():
            p.requires_grad = False
        for p in self.freq_branch.parameters():
            p.requires_grad = False
        for p in self.fusion_head.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def set_branches_eval(self) -> None:
        """Keep BN running stats fixed while training fusion head."""
        self.rgb_branch.eval()
        self.freq_branch.eval()
        self.fusion_head.train()

    def parameter_groups(self, rgb_lr: float, freq_lr: float, head_lr: float, weight_decay: float):
        return [
            {"params": self.rgb_branch.parameters(), "lr": rgb_lr, "weight_decay": weight_decay},
            {"params": self.freq_branch.parameters(), "lr": freq_lr, "weight_decay": weight_decay},
            {"params": self.fusion_head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]

    def count_component_params(self) -> dict[str, int]:
        rgb_total, _ = count_parameters(self.rgb_branch)
        freq_total, _ = count_parameters(self.freq_branch)
        head_total, _ = count_parameters(self.fusion_head)
        total, trainable = count_parameters(self)
        return {
            "rgb_branch": rgb_total,
            "freq_branch": freq_total,
            "fusion_head": head_total,
            "total": total,
            "trainable": trainable,
        }
