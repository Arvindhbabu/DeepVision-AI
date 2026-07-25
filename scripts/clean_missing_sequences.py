from pathlib import Path
import pandas as pd

ROOT = Path("data/intermediate/sequences")

csv_files = [
    "outputs/splits/train.csv",
    "outputs/splits/val.csv",
    "outputs/splits/test.csv",
]

total_removed = 0

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    keep_rows = []

    for _, row in df.iterrows():
        folder = "ffpp" if row["dataset"].lower() == "ffpp" else "celebdf"
        npy = ROOT / folder / f"{row['video_name']}.npy"

        if npy.exists():
            keep_rows.append(row)

    new_df = pd.DataFrame(keep_rows)

    removed = len(df) - len(new_df)
    total_removed += removed

    new_df.to_csv(csv_file, index=False)

    print(f"{csv_file}: removed {removed} rows")

print(f"\nTotal removed: {total_removed}")