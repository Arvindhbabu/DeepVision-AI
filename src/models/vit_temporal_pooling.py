"""
DeepVision AI

Vision Transformer with Temporal Mean Pooling
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from src.models.base_model import BaseModel


class ViTTemporalPooling(BaseModel):
    """
    Vision Transformer + Temporal Pooling
    """

    def __init__(
        self,
        image_size=224,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
        pooling="mean",
        hidden_dim=256,
        dropout=0.5,
    ):
        super().__init__()

        self.image_size = image_size
        self.pooling = pooling

        # Load ViT backbone
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = vit_b_16(weights=weights)

        # Remove classification head
        self.backbone.heads = nn.Identity()

        # Feature dimension of ViT-B/16
        self.feature_dim = 768

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def temporal_pool(self, features):
        """
        features:
            (B,T,768)
        """

        if self.pooling == "mean":
            return features.mean(dim=1)

        elif self.pooling == "max":
            return features.max(dim=1)[0]

        else:
            raise ValueError(
                f"Unsupported pooling: {self.pooling}"
            )

    def forward(self, x):
        """
        x:
            (B,T,C,H,W)
        """

        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        features = self.backbone(x)

        features = features.reshape(B, T, self.feature_dim)

        video_features = self.temporal_pool(features)

        logits = self.classifier(video_features)

        return logits