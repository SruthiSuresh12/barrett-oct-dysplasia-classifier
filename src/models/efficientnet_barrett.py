#!/usr/bin/env python3
"""
EfficientNet-B0 fine-tuned for Barrett's esophagus classification.

Architecture:
  - Backbone: EfficientNet-B0 pretrained on ImageNet
  - Classification head: 3-class (normal / barrett / esophagitis)
  - DRS head: auxiliary scalar output for Dysplasia Risk Score regression

Rationale for EfficientNet-B0:
  - Lightweight (5.3M params) → deployable on edge device in tethered capsule reader
  - Strong transfer learning from ImageNet mucosal texture features
  - Compound scaling balances depth/width/resolution efficiently
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class BarrettClassifier(nn.Module):
    """
    EfficientNet-B0 with dual-head output:
      1. Class logits (normal / barrett / esophagitis)
      2. DRS auxiliary scalar (0-1, dysplasia risk proxy)
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Feature extractor (all layers except classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feature_dim = backbone.classifier[1].in_features  # 1280

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Auxiliary DRS regression head (trained jointly with classification)
        self.drs_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

        # Freeze early layers (1-4), fine-tune later layers + head
        self._freeze_early_layers(freeze_up_to=4)

    def _freeze_early_layers(self, freeze_up_to: int = 4):
        """Freeze first N feature blocks for efficient fine-tuning."""
        for i, block in enumerate(self.features):
            if i < freeze_up_to:
                for param in block.parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers (call after initial head training)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, num_classes) classification logits
            drs: (B, 1) dysplasia risk score in [0, 1]
        """
        features = self.features(x)
        pooled = self.avgpool(features)
        flat = torch.flatten(pooled, 1)

        logits = self.classifier(flat)
        drs = self.drs_head(flat)

        return logits, drs

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature vector (for Grad-CAM hook)."""
        features = self.features(x)
        pooled = self.avgpool(features)
        return torch.flatten(pooled, 1)


class BarrettLoss(nn.Module):
    """
    Combined loss: CrossEntropy + auxiliary DRS MSE.
    
    DRS supervision: barrett class = high risk (0.8),
                     esophagitis = moderate (0.4),
                     normal = low (0.05)
    """
    DRS_TARGETS = {0: 0.05, 1: 0.80, 2: 0.40}  # normal=0, barrett=1, esophagitis=2

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        drs_weight: float = 0.2,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()
        self.drs_weight = drs_weight

    def forward(
        self,
        logits: torch.Tensor,
        drs_pred: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, ce_loss, drs_loss
        """
        ce = self.ce_loss(logits, targets)

        # Build DRS supervision targets from class labels
        drs_targets = torch.tensor(
            [self.DRS_TARGETS[t.item()] for t in targets],
            dtype=torch.float32,
            device=targets.device,
        ).unsqueeze(1)
        drs_l = self.mse_loss(drs_pred, drs_targets)

        total = ce + self.drs_weight * drs_l
        return total, ce, drs_l


def build_model(
    num_classes: int = 3,
    dropout: float = 0.3,
    pretrained: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    drs_weight: float = 0.2,
    device: str = "cpu",
) -> Tuple[BarrettClassifier, BarrettLoss]:
    """Convenience builder returning model + loss on specified device."""
    model = BarrettClassifier(num_classes=num_classes, dropout=dropout, pretrained=pretrained)
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = BarrettLoss(class_weights=class_weights, drs_weight=drs_weight)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EfficientNet-B0 | Trainable params: {n_params:,} | Device: {device}")

    return model, loss_fn
