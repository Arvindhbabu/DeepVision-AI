import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from src.config import SEQS_DIR

# =========================
# CONFIG
# =========================
BATCH_SIZE = 4            # CPU-safe
EPOCHS = 5                # baseline training
LR = 1e-4
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "results/models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "efficientnetb2_frame_ckpt.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnetb2_frame.pth")

print(f"[INFO] Training on device: {DEVICE}")


# =========================
# DATASET
# =========================
class FrameDataset(Dataset):
    """
    Frame-level dataset built from video sequences (.npy).
    Each video contributes 30 frames.
    """

    def __init__(self, seq_dir, label):
        self.files = [os.path.join(seq_dir, f) for f in os.listdir(seq_dir)]
        self.label = label

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((260, 260)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.files) * 30

    def __getitem__(self, idx):
        video_idx = idx // 30
        frame_idx = idx % 30

        seq = np.load(self.files[video_idx])
        frame = seq[frame_idx]

        frame = self.transform(frame)
        label = torch.tensor(self.label, dtype=torch.long)

        return frame, label


def build_dataloader():
    ffpp_real = FrameDataset(os.path.join(SEQS_DIR, "ffpp"), label=0)
    celebdf_fake = FrameDataset(os.path.join(SEQS_DIR, "celebdf"), label=1)

    dataset = torch.utils.data.ConcatDataset([ffpp_real, celebdf_fake])

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )


# =========================
# MODEL
# =========================
def build_model():
    model = models.efficientnet_b2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, NUM_CLASSES
    )
    return model


# =========================
# TRAINING (WITH RESUME)
# =========================
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataloader = build_dataloader()
    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0

    # 🔁 RESUME LOGIC
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[INFO] Resuming training from Epoch {start_epoch + 1}")

    model.train()

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        print(f"\n[INFO] Epoch {epoch + 1}/{EPOCHS}")

        for frames, labels in tqdm(dataloader, desc="Training"):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        print(f"[INFO] Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

        # 💾 SAVE CHECKPOINT
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print("[INFO] Checkpoint saved")

    # =========================
    # SAVE FINAL MODEL
    # =========================
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"[INFO] Final model saved: {FINAL_MODEL_PATH}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()
