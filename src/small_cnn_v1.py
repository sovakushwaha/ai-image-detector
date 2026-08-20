"""SmallCNNV1 — compact CNN for binary AI-image detection (Stage 12).

Why this file exists
--------------------
This is the first neural architecture for the dissertation pipeline.
It is intentionally small: three convolution blocks, global average
pooling, and a single logit output.

Training will use BCEWithLogitsLoss, which expects RAW LOGITS.
Therefore model.forward() must NOT apply sigmoid.

During later evaluation:
    probability = torch.sigmoid(logit)

How to use
----------
    from small_cnn_v1 import SmallCNNV1, count_parameters
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNNV1(nn.Module):
    """Three-block CNN → AdaptiveAvgPool → Linear(64 → 1 logit).

    Spatial path for a 224×224 RGB input:
        Block1: 3×224×224 → 16×112×112
        Block2: 16×112×112 → 32×56×56
        Block3: 32×56×56 → 64×28×28
        Pool:   64×28×28 → 64×1×1
        Linear: 64 → 1 raw logit
    """

    def __init__(self) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=64, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape [B, 1]. No sigmoid here."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
