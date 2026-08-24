"""RQ4 F1 frequency-only CNN (Stage 24A.4).

Why this file exists
--------------------
Locked lightweight CNN operating on the FrequencyTransformV1 1×224×224
log-magnitude spectrum. Exposes forward_features() for F2 fusion reuse.

Architecture is fixed. Do not modify after training begins.
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyOnlyCNNV1(nn.Module):
    """F1: Conv CNN on normalised FFT log-magnitude spectrum → raw logit."""

    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, 1)
        self.embedding_dim = 128

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-dimensional frequency embedding [B, 128]."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape [B]. No sigmoid."""
        features = self.forward_features(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits.squeeze(1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
