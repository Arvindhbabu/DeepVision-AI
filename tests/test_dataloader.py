from src.utils.config import load_config
from src.datasets.dataloader import create_dataloaders


def main():

    config = load_config()

    train_loader, val_loader, test_loader = create_dataloaders(config)

    print(f"Train batches : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")

    batch = next(iter(train_loader))

    print("\nBatch Information")

    print("Sequence:", batch["sequence"].shape)
    print("Labels:", batch["label"].shape)
    print("Datasets:", batch["dataset"][:5])
    print("Videos:", batch["video_name"][:5])


if __name__ == "__main__":
    main()