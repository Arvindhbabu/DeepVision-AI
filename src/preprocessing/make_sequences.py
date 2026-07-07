import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

from src.config import ALIGNED_DIR, SEQS_DIR, SEQ_LEN


def load_image_safe(img_path):
    """Safely load an image, return None if corrupted."""
    try:
        with Image.open(img_path) as img:
            return np.array(img.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        return None


def process_video(video_dir: Path, out_file: Path):
    """Create one sequence from a video folder."""
    frame_files = sorted(video_dir.glob("*.jpg"))

    valid_frames = []
    for f in frame_files:
        img = load_image_safe(f)
        if img is not None:
            valid_frames.append(img)

    # Skip if insufficient frames
    if len(valid_frames) < SEQ_LEN:
        return False

    # Uniform sampling
    indices = np.linspace(0, len(valid_frames) - 1, SEQ_LEN).astype(int)
    sequence = np.stack([valid_frames[i] for i in indices])

    np.save(out_file, sequence.astype(np.uint8))
    return True


def process_dataset(dataset_name):
    aligned_root = Path(ALIGNED_DIR) / dataset_name
    out_root = Path(SEQS_DIR) / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([d for d in aligned_root.iterdir() if d.is_dir()])

    print(f"[INFO] Creating sequences for {dataset_name} ({len(video_dirs)} videos)")

    for video_dir in tqdm(video_dirs, desc=f"Sequences ({dataset_name})"):
        out_file = out_root / f"{video_dir.name}.npy"

        # ✅ RESUME LOGIC
        if out_file.exists():
            continue

        try:
            process_video(video_dir, out_file)
        except Exception:
            # Never crash on a single bad video
            continue


if __name__ == "__main__":
    process_dataset("ffpp")
    process_dataset("celebdf")
    print("[INFO] Sequence creation completed successfully")
