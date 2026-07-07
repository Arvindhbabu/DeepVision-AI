#!/usr/bin/env python3
import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from src.config import RAW_FFPP, RAW_CELEBDF, FRAMES_DIR, FPS

def list_videos(root_folder):
    vids = []
    for root, _, files in os.walk(root_folder):
        for f in files:
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                vids.append(os.path.join(root, f))
    return vids

def extract(video_path, out_dir, target_fps=5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    v_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(v_fps / target_fps)))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fn = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(fn, frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

def process_dataset(root_videos, out_base):
    videos = list_videos(root_videos)
    print(f"Found {len(videos)} videos in {root_videos}")
    for v in tqdm(videos):
        # create output folder by video basename (no extension)
        vidname = Path(v).stem
        out_dir = os.path.join(out_base, vidname)
        if len(os.listdir(out_dir)) if os.path.exists(out_dir) else 0:
            # skip if frames already exist
            continue
        try:
            n = extract(v, out_dir, target_fps=FPS)
        except Exception as e:
            print("Error", v, e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ffpp", "celebdf"],
                        help="which datasets to process: ffpp, celebdf")
    args = parser.parse_args()
    mapping = {
        "ffpp": RAW_FFPP,
        "celebdf": RAW_CELEBDF
    }
    os.makedirs(FRAMES_DIR, exist_ok=True)
    for ds in args.datasets:
        root = mapping.get(ds)
        if root and os.path.exists(root):
            out = os.path.join(FRAMES_DIR, ds)
            process_dataset(root, out)
        else:
            print("Dataset path missing:", root)
