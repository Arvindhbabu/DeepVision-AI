import os
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import models

from src.config import SEQS_DIR

# =========================
# CONFIG
# =========================
BATCH_SIZE = 1
ACCUM_STEPS = 4
EPOCHS = 6
UNFREEZE_EPOCH = 3
LR = 1e-4
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "results/models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "efficientnetb2_bilstm_ckpt.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnetb2_bilstm.pth")

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"[INFO] Training on device: {DEVICE}")

# =========================
# DATASET
# =========================
class VideoDataset(Dataset):
    def __init__(self, seq_dir, label):
        self.files = [
            os.path.join(seq_dir, f)
            for f in os.listdir(seq_dir)
            if f.endswith(".npy")
        ]
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        seq = np.load(self.files[idx])            # (T, H, W, C)
        seq = torch.tensor(seq, dtype=torch.float32) / 255.0
        seq = seq.permute(0, 3, 1, 2)              # (T, C, H, W)
        label = torch.tensor(self.label, dtype=torch.long)
        return seq, label


# =========================
# STRATIFIED DATALOADERS
# =========================
def build_dataloaders():
    real_ds = VideoDataset(os.path.join(SEQS_DIR, "ffpp"), 0)
    fake_ds = VideoDataset(os.path.join(SEQS_DIR, "celebdf"), 1)

    print(f"[INFO] Real samples : {len(real_ds)}")
    print(f"[INFO] Fake samples : {len(fake_ds)}")

    assert len(real_ds) > 0 and len(fake_ds) > 0, \
        "One of the classes has ZERO samples. Check dataset paths."

    full_ds = ConcatDataset([real_ds, fake_ds])

    real_indices = list(range(len(real_ds)))
    fake_indices = list(range(len(fake_ds)))

    np.random.shuffle(real_indices)
    np.random.shuffle(fake_indices)

    split_real = int(0.8 * len(real_indices))
    split_fake = int(0.8 * len(fake_indices))

    train_indices = (
        real_indices[:split_real] +
        [i + len(real_ds) for i in fake_indices[:split_fake]]
    )

    val_indices = (
        real_indices[split_real:] +
        [i + len(real_ds) for i in fake_indices[split_fake:]]
    )

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, train_ds


# =========================
# MODEL
# =========================
class EfficientNetBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b2(weights="DEFAULT")
        self.cnn = backbone.features

        # Freeze CNN initially
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1408

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.cnn(x)
        feats = self.pool(feats).view(B, T, -1)

        lstm_out, _ = self.lstm(feats)
        out = lstm_out[:, -1, :]
        return self.classifier(out)


def unfreeze_last_blocks(model, n_blocks=2):
    blocks = list(model.cnn.children())
    for block in blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


# =========================
# VALIDATION
# =========================
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            outputs = model(seqs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


# =========================
# TRAINING
# =========================
def train():
    train_loader, val_loader, train_ds = build_dataloaders()
    model = EfficientNetBiLSTM().to(DEVICE)

    # ---------- SAFE CLASS WEIGHTS ----------
    labels = [lbl for _, lbl in train_ds]
    counter = Counter(labels)

    print("[INFO] Train class distribution:", counter)

    total = sum(counter.values())
    weights = []

    for i in range(NUM_CLASSES):
        if counter.get(i, 0) == 0:
            weights.append(1.0)   # SAFE fallback
        else:
            weights.append(total / counter[i])

    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    cnn_optimizer = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        optimizer.zero_grad()

        print(f"\n[INFO] Epoch {epoch + 1}/{EPOCHS}")

        if epoch == UNFREEZE_EPOCH:
            print("[INFO] Unfreezing CNN blocks")
            unfreeze_last_blocks(model)
            cnn_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.cnn.parameters()),
                lr=LR * 0.1
            )

        for i, (seqs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)

            outputs = model(seqs)
            loss = criterion(outputs, labels) / ACCUM_STEPS
            loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                if cnn_optimizer:
                    cnn_optimizer.step()
                    cnn_optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total_samples
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"[INFO] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"[INFO] Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"[INFO] Final model saved: {FINAL_MODEL_PATH}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train()