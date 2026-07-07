import os
import random
import shutil

BASE = "data/raw/celebdf_v2"
OUT = "data/raw/celebdf_v2_sampled"

REAL_SRC = os.path.join(BASE, "Celeb-real")
FAKE_SRC = os.path.join(BASE, "Celeb-synthesis")

REAL_DST = os.path.join(OUT, "Celeb-real")
FAKE_DST = os.path.join(OUT, "Celeb-synthesis")

os.makedirs(REAL_DST, exist_ok=True)
os.makedirs(FAKE_DST, exist_ok=True)

random.seed(42)

# Use ALL real videos (Celeb-DF has only ~590)
real_files = os.listdir(REAL_SRC)

# Subsample fake videos
TARGET_FAKE = 1500  # you can change to 2000 if you want
fake_files = random.sample(os.listdir(FAKE_SRC), TARGET_FAKE)

print(f"Using {len(real_files)} real videos")
print(f"Using {len(fake_files)} fake videos")

for f in real_files:
    shutil.copy(os.path.join(REAL_SRC, f), REAL_DST)

for f in fake_files:
    shutil.copy(os.path.join(FAKE_SRC, f), FAKE_DST)

print("Celeb-DF sampling completed successfully.")
