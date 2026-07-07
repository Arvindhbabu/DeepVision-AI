import cv2
import torch
import numpy as np
import random
import torch.nn as nn
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/models/efficientnetb2_bilstm.pth"

FRAME_SIZE = 224
SEQ_LEN = 30
NUM_CLIPS = 5   # increase to 10 for better accuracy

# =========================
# MODEL (same as training)
# =========================
class EfficientNetBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b2(weights="DEFAULT")
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=1408,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.cnn(x)
        feats = self.pool(feats).view(B, T, -1)

        out, _ = self.lstm(feats)
        return self.classifier(out[:, -1])


# =========================
# VIDEO → MULTI CLIPS
# =========================
def extract_random_clips(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frame = frame[:, :, ::-1]  # BGR → RGB
        frames.append(frame)

    cap.release()
    frames = np.array(frames)

    if len(frames) < SEQ_LEN:
        raise ValueError("Video too short")

    clips = []
    for _ in range(NUM_CLIPS):
        start = random.randint(0, len(frames) - SEQ_LEN)
        clip = frames[start:start + SEQ_LEN]
        clip = torch.tensor(clip, dtype=torch.float32) / 255.0
        clip = clip.permute(0, 3, 1, 2)
        clips.append(clip)

    clips = torch.stack(clips)   # (N, T, C, H, W)
    return clips


# =========================
# PREDICT
# =========================
def predict_video(video_path):
    model = EfficientNetBiLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    clips = extract_random_clips(video_path).to(DEVICE)

    probs = []
    with torch.no_grad():
        for clip in clips:
            clip = clip.unsqueeze(0)
            logits = model(clip)
            prob = torch.softmax(logits, dim=1)[0, 1]  # FAKE prob
            probs.append(prob.item())

    avg_prob = np.mean(probs)
    label = "DEEPFAKE" if avg_prob > 0.5 else "REAL"

    return label, avg_prob * 100, probs


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    video_path = "sample.mp4"  # change this

    label, confidence, clip_probs = predict_video(video_path)

    print("Clip-wise FAKE probabilities:")
    for i, p in enumerate(clip_probs):
        print(f"  Clip {i+1}: {p:.3f}")

    print("\nFinal Result")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")