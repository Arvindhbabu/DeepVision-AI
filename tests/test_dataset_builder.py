from src.datasets.tf_dataset_builder import DatasetIndexer

builder = DatasetIndexer("data/raw")

samples = builder.scan()

builder.summary()

# Save dataset index
builder.save_index("outputs/dataset_index.csv")

print("\nFirst 10 Samples\n")

for sample in samples[:10]:
    print(sample)