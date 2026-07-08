from src.datasets.sequence_dataset import SequenceDataset

dataset = SequenceDataset(
    csv_file="outputs/dataset_index.csv",
    sequence_root="data/intermediate/sequences"
)

print("Dataset Size:", len(dataset))

sample = dataset[0]

print()

print("Sequence Shape :", sample["sequence"].shape)
print("Sequence Type  :", sample["sequence"].dtype)
print("Label          :", sample["label"])
print("Dataset        :", sample["dataset"])
print("Video          :", sample["video_name"])

print()

print("Min:", sample["sequence"].min())

print("Max:", sample["sequence"].max())