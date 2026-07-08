"""
DeepVision AI

End-to-End Training Smoke Test
"""

import torch
import torch.nn as nn

from src.utils.config import load_config
from src.datasets.dataloader import create_dataloaders
from src.models.model_factory import ModelFactory


def main():

    print("=" * 60)
    print(" DeepVision AI - Integration Test ")
    print("=" * 60)

    config = load_config()

    print("\nLoading dataloaders...")

    train_loader, _, _ = create_dataloaders(config)

    batch = next(iter(train_loader))

    sequences = batch["sequence"]

    labels = batch["label"]

    print(f"Input Shape : {sequences.shape}")

    print("\nLoading model...")

    model = ModelFactory.create(config)

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    print("\nForward Pass...")

    outputs = model(sequences)

    print(f"Output Shape : {outputs.shape}")

    print("\nComputing Loss...")

    loss = criterion(outputs, labels)

    print(f"Loss : {loss.item():.4f}")

    print("\nBackward Pass...")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("\nBackward Pass Successful")

    print("\nChecking Gradients...")

    gradients = 0

    for name, param in model.named_parameters():

        if param.grad is not None:

            gradients += 1

    print(f"Layers with gradients : {gradients}")

    print("\nIntegration Test PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()