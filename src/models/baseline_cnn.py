from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ResNetCalorieRegressor(nn.Module):
    """
    Baseline calorie regressor using a pretrained ResNet backbone.

    By default:
    - Uses ResNet-50 pretrained on ImageNet.
    - Replaces the final classification layer with a regression head (1 output).
    - Can optionally freeze the backbone for faster training / small datasets.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout_p: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load backbone
        if backbone_name == "resnet50":
            # Newer torchvision uses weights enums; fall back if needed.
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                backbone = models.resnet50(weights=weights)
            except AttributeError:
                backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        # Optionally freeze all backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Replace final FC with regression head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 1),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, 3, H, W)

        Returns:
            predictions: tensor of shape (B,) with predicted calories.
        """
        out = self.backbone(x)          # (B, 1)
        out = out.squeeze(-1)           # -> (B,)
        return out