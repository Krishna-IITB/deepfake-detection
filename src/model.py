"""Deepfake classifier built on a pretrained EfficientNet backbone.

The classifier is binary: real (0) vs fake (1). The backbone's ImageNet
classifier head is replaced with a small MLP producing a single logit;
use ``BCEWithLogitsLoss`` for training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


_BACKBONES = {
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        1280,
    ),
    "efficientnet_b3": (
        models.efficientnet_b3,
        models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        1536,
    ),
    "efficientnet_b4": (
        models.efficientnet_b4,
        models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        1792,
    ),
}


class DeepfakeDetector(nn.Module):
    """EfficientNet-based binary classifier for face-level deepfake detection.

    Args:
        backbone:    one of ``efficientnet_b0`` / ``efficientnet_b3`` / ``efficientnet_b4``.
        pretrained:  load ImageNet-pretrained weights.
        dropout:     dropout rate inside the new classification head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Available: {list(_BACKBONES)}"
            )
        ctor, weights, feat_dim = _BACKBONES[backbone]
        self.backbone = ctor(weights=weights if pretrained else None)
        # Replace the final classifier with a binary head.
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.backbone(x)


if __name__ == "__main__":
    # Smoke test: build the model and run a dummy forward pass.
    m = DeepfakeDetector(backbone="efficientnet_b0", pretrained=False)
    x = torch.randn(2, 3, 380, 380)
    y = m(x)
    print(f"output shape: {tuple(y.shape)}")
