import torch
import torch.nn as nn

from vit_temporal_pooling.models.vit_pooling import ViTTemporalPooling

# REUSE EXISTING DATA PIPELINE
from src.datasets.balanced_sampled_dataset import build_dataloaders
from src.utils.config import load_config


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load same config used by baseline models
    config = load_config("vit_temporal_pooling/config.yaml")

    # Reuse dataloaders (IMPORTANT)
    train_loader, val_loader = build_dataloaders(config)

    # Model
    model = ViTTemporalPooling(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for videos, labels in train_loader:
            videos = videos.to(device)   # (B, T, C, H, W)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), "results/models/vit_temporal_pooling.pth")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    train()