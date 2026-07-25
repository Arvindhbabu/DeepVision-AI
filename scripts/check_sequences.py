from pathlib import Path
import pandas as pd

csv_files = [
    "outputs/splits/train.csv",
    "outputs/splits/val.csv",
    "outputs/splits/test.csv",
]

root = Path("data/intermediate/sequences")

missing = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        folder = "ffpp" if row["dataset"].lower() == "ffpp" else "celebdf"
        npy = root / folder / f"{row['video_name']}.npy"

        if not npy.exists():
            missing.append(str(npy))

print(f"Missing files: {len(missing)}")

for path in missing[:20]:
    print(path)