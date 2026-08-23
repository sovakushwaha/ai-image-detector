"""EfficientNet-B0 binary classifier for AI-image detection (Stage 19).

Why this file exists
--------------------
Stage 19 introduces a second transfer-learning baseline. EfficientNet-B0 is
loaded with official ImageNet pretrained weights. Only the final Linear
classification layer is replaced so the model outputs one raw binary logit.

Training will use BCEWithLogitsLoss, which expects RAW LOGITS.
Therefore forward() must NOT apply sigmoid.

How to use
----------
    from efficientnet_b0_binary_v1 import (
        EfficientNetB0BinaryV1,
        count_parameters,
        count_binary_head_parameters,
        DEFAULT_WEIGHTS,
    )
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

DEFAULT_WEIGHTS = EfficientNet_B0_Weights.DEFAULT


class EfficientNetB0BinaryV1(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 with a single binary logit head.

    The pretrained feature backbone is unchanged. Only the final Linear layer
    (originally 1000-class ImageNet output) is replaced by Linear(..., 1).
    """

    def __init__(
        self,
        weights: EfficientNet_B0_Weights = DEFAULT_WEIGHTS,
    ) -> None:
        super().__init__()
        self.weights_enum = weights
        self.backbone = efficientnet_b0(weights=weights)

        final_layer = self.backbone.classifier[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("Expected final classifier layer to be nn.Linear")

        self.original_classifier_output_size = int(final_layer.out_features)
        in_features = int(final_layer.in_features)
        self.backbone.classifier[-1] = nn.Linear(in_features, 1)
        self.binary_classifier_output_size = 1

    @property
    def weights_name(self) -> str:
        return str(self.weights_enum)

    @property
    def weights_url(self) -> str:
        return str(self.weights_enum.url)

    @property
    def features(self) -> nn.Module:
        """Pretrained convolutional feature backbone."""
        return self.backbone.features

    @property
    def classifier(self) -> nn.Module:
        """Classifier head including Dropout and binary Linear(…→1)."""
        return self.backbone.classifier

    def freeze_features(self) -> None:
        """Phase 1: freeze pretrained feature backbone; keep classifier trainable."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Phase 2: enable gradients for all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape [B]. No sigmoid here."""
        logits = self.backbone(x)
        return logits.squeeze(1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_binary_head_parameters(model: EfficientNetB0BinaryV1) -> int:
    """Parameter count of the replacement final Linear layer only."""
    final_layer = model.backbone.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("Expected final classifier layer to be nn.Linear")
    return sum(p.numel() for p in final_layer.parameters())
