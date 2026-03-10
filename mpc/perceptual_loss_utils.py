"""
VGG Perceptual Loss utilities for MPC scoring.
Uses deep features (ReLU3_3 by default) to focus on structure/semantics.
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision import models
except ImportError as exc:
    raise ImportError("torchvision is required for VGG perceptual loss") from exc


class VGGPerceptualLoss(nn.Module):
    """
    Compute perceptual loss using VGG features.
    Inputs must be in range [0, 1] with shape (B, C, H, W).
    """

    _LAYER_TO_IDX = {
        "relu1_1": 1,
        "relu1_2": 3,
        "relu2_1": 6,
        "relu2_2": 8,
        "relu3_1": 11,
        "relu3_2": 13,
        "relu3_3": 15,
        "relu4_1": 18,
        "relu4_2": 20,
        "relu4_3": 22,
    }

    def __init__(self, layer: str = "relu3_3", device: Optional[torch.device] = None):
        super().__init__()
        if layer not in self._LAYER_TO_IDX:
            raise ValueError(f"Unsupported VGG layer: {layer}")

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[: self._LAYER_TO_IDX[layer] + 1])
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if device is not None:
            self.to(device)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 3, H, W) in [0, 1]
            target: (B, 3, H, W) in [0, 1]
        Returns:
            loss: (B,) perceptual L1 loss per batch
        """
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        feat_pred = self.feature_extractor(pred_norm)
        feat_target = self.feature_extractor(target_norm)
        loss = torch.mean(torch.abs(feat_pred - feat_target), dim=(1, 2, 3))
        return loss