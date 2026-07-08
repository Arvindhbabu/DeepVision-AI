import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import CheckpointManager
from src.utils.logger import create_logger


class DummyDataset(torch.utils.data.Dataset):

    def __len__(self):
        return 64

    def __getitem__(self, idx):

        return {
            "sequence": torch.randn(3, 32),
            "label": torch.randint(0, 2, (1,)).squeeze()
        }


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


def main():

    train_loader = DataLoader(
        DummyDataset(),
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        DummyDataset(),
        batch_size=8
    )

    model = DummyModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        logger=create_logger(),
        checkpoint_manager=CheckpointManager(),
        early_stopping=EarlyStopping(patience=2)
    )

    trainer.train(epochs=3)

    print("\nTrainer Test Passed")


if __name__ == "__main__":
    main()