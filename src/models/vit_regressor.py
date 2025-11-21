from typing import Optional

import torch
import torch.nn as nn
import timm


class ViTCalorieRegressor(nn.Module):
    """
    Vision Transformer-based calorie regressor using timm.

    By default:
    - Uses a ViT backbone pretrained on ImageNet.
    - Sets num_classes=1 for scalar regression.
    - Can optionally freeze the backbone for small datasets.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Create timm model with a single output unit (regression)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,  # scalar output
        )

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

            # However, we still want to train the head (last layer).
            # For timm ViT, the head is usually in `head`.
            if hasattr(self.backbone, "head"):
                for param in self.backbone.head.parameters():
                    param.requires_grad = True

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