import torch.nn as nn

from src.training.checkpoint import CheckpointManager
from src.utils.config import load_config


def main():

    manager = CheckpointManager()

    print("Run Directory")

    print(manager.path)

    model = nn.Linear(10, 2)

    manager.save_best_model(model)

    manager.save_last_model(model)

    manager.save_metrics(
        {
            "accuracy": 0.95,
            "loss": 0.12,
        }
    )

    config = load_config()

    manager.save_config(config)

    print("\nCheckpoint Test Passed")


if __name__ == "__main__":
    main()