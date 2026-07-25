import torch

from src.utils.config import load_config
from src.datasets.dataloader import create_dataloaders
from src.models.model_factory import ModelFactory


def main():

    config = load_config("configs/default.yaml")

    train_loader, val_loader, test_loader = create_dataloaders(config)

    print(f"Train batches : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")

    batch = next(iter(train_loader))

    print("\nBatch Information")
    print("------------------")
    print("Sequence :", batch["sequence"].shape)
    print("Label    :", batch["label"].shape)
    print("Dataset  :", batch["dataset"][0])
    print("Video    :", batch["video_name"][0])

    model = ModelFactory.create(config)

    model.eval()

    with torch.no_grad():
        outputs = model(batch["sequence"])

    print("\nModel Output")
    print("------------")
    print(outputs.shape)

    print("\nPipeline Test Passed")


if __name__ == "__main__":
    main()