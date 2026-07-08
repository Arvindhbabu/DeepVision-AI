from src.datasets.sequence_dataset import SequenceDataset
from src.datasets.balanced_sampler import create_balanced_sampler

dataset = SequenceDataset(
    csv_file="outputs/dataset_index.csv",
    sequence_root="data/intermediate/sequences"
)

sampler = create_balanced_sampler(dataset)

print("Dataset Size :", len(dataset))

print("Sampler Size :", len(sampler))

print(type(sampler))