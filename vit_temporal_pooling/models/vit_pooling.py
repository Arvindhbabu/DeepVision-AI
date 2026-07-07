import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTTemporalPooling(nn.Module):
    def __init__(self, num_classes=2, freeze_vit=True):
        super().__init__()

        # Load pretrained ViT
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Remove ViT classification head
        self.vit.heads = nn.Identity()

        # Freeze ViT backbone
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Merge batch and time
        x = x.view(B * T, C, H, W)

        # Frame-level embeddings
        features = self.vit(x)  # (B*T, 768)

        # Restore temporal dimension
        features = features.view(B, T, -1)

        # Temporal mean pooling
        video_features = features.mean(dim=1)

        return self.classifier(video_features)