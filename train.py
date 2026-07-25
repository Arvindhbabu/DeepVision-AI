"""
=============================================================
DeepVision AI

Main Training Script

Author : Arvindh Babu
=============================================================
"""

import argparse
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from src.utils.config import load_config
from src.utils.logger import create_logger

from src.datasets.dataloader import create_dataloaders

from src.models.model_factory import ModelFactory

from src.training.optimizer_factory import OptimizerFactory
from src.training.scheduler_factory import SchedulerFactory

from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer


# ============================================================
# Argument Parser
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="DeepVision AI Training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume training",
    )

    return parser.parse_args()


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print("=" * 60)
        print("CUDA AVAILABLE")
        print("=" * 60)

        print(
            "GPU :",
            torch.cuda.get_device_name(0),
        )

        print(
            "CUDA Version :",
            torch.version.cuda,
        )

        print("=" * 60)

    else:

        device = torch.device("cpu")

        print("=" * 60)
        print("Running on CPU")
        print("=" * 60)

    return device


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    config = load_config(args.config)

    logger = create_logger()

    logger.info("=" * 60)
    logger.info("DeepVision AI")
    logger.info("=" * 60)

    device = get_device()

    logger.info(f"Device : {device}")

    # --------------------------------------------------------
    # Dataloaders
    # --------------------------------------------------------

    logger.info("Creating dataloaders...")

    train_loader, val_loader, test_loader = create_dataloaders(
        config
    )

    logger.info("DataLoaders Ready")

    logger.info(
        f"Training batches : {len(train_loader)}"
    )

    logger.info(
        f"Validation batches : {len(val_loader)}"
    )

    logger.info(
        f"Test batches : {len(test_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    logger.info("Creating model...")

    model = ModelFactory.create(config)

    model = model.to(device)

    logger.info(f"Model : {model.__class__.__name__}")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    logger.info(
        f"Total Parameters      : {total_params:,}"
    )

    logger.info(
        f"Trainable Parameters  : {trainable_params:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    logger.info("Loss : CrossEntropyLoss")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = OptimizerFactory.create(
        model,
        config,
    )

    logger.info(
        f"Optimizer : {optimizer.__class__.__name__}"
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = SchedulerFactory.create(
        optimizer,
        config,
    )

    if scheduler is not None:

        logger.info(
            f"Scheduler : {scheduler.__class__.__name__}"
        )

    else:

        logger.info("Scheduler : None")

    # --------------------------------------------------------
    # Checkpoint Manager
    # --------------------------------------------------------

    config["runtime"] = {
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    checkpoint_manager = CheckpointManager()

    checkpoint_manager.save_config(config)

    logger.info(
        f"Experiment Folder : {checkpoint_manager.path}"
    )

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        patience=5
    )

    # --------------------------------------------------------
    # Resume Training
    # --------------------------------------------------------

    if args.resume is not None:

        logger.info(
            f"Loading checkpoint : {args.resume}"
        )

        epoch, best_loss = checkpoint_manager.load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )

        trainer_state = None  # placeholder until trainer is created

        

        logger.info(
            f"Checkpoint Loaded (Epoch {epoch})"
        )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        optimizer=optimizer,

        criterion=criterion,

        train_loader=train_loader,

        val_loader=val_loader,

        device=device,

        logger=logger,

        checkpoint_manager=checkpoint_manager,

        early_stopping=early_stopping,

        scheduler=scheduler,
    )

    if args.resume is not None:
        trainer.state.epoch = epoch
        trainer.state.best_val_loss = best_loss

    # --------------------------------------------------------
    # Start Training
    # --------------------------------------------------------

    logger.info("=" * 60)
    logger.info("DeepVision AI")
    logger.info("=" * 60)
    logger.info(f"Model      : {model.__class__.__name__}")
    logger.info(f"Device     : {device}")
    logger.info(f"Epochs     : {config['training']['epochs']}")
    logger.info(f"Batch Size : {config['dataloader']['batch_size']}")
    logger.info(
        f"Learning Rate : {config['training']['learning_rate']}"
    )
    logger.info(
        f"Dataset : {config['dataset']['name']}"
    )
    logger.info("=" * 60)

    history = []

    try:

        history = trainer.train(
            epochs=config["training"]["epochs"]
        )

    except KeyboardInterrupt:

        logger.warning(
            "Training interrupted by user."
        )

    except Exception:

        logger.exception(
            "Training failed."
        )

        raise

    finally:
        try:
            checkpoint_manager.copy_log(
                "outputs/logs/train.log"
            )
        except Exception as e:
            logger.warning(
                f"Unable to copy log: {e}"
            )

    # --------------------------------------------------------
    # Save Training Log
    # --------------------------------------------------------
    
    try:

        checkpoint_manager.copy_log(
            "outputs/logs/train.log"
        )

    except Exception as e:

        logger.warning(
            f"Unable to copy log file: {e}"
        )

    # --------------------------------------------------------
    # Training Summary
    # --------------------------------------------------------

    logger.info("=" * 60)
    logger.info("Training Completed Successfully")
    logger.info("=" * 60)

    logger.info(
        f"Experiment Directory : {checkpoint_manager.path}"
    )

    logger.info(
        f"Epochs Completed : {len(history)}"
    )

    if len(history) > 0:

        best_epoch = min(
            history,
            key=lambda x: x["val_loss"]
        )

        logger.info(
            f"Best Validation Loss : {best_epoch['val_loss']:.4f}"
        )

        logger.info(
            f"Best Validation Accuracy : {best_epoch['val_acc']:.2f}%"
        )

    logger.info(
        "Artifacts Saved:"
    )

    logger.info(
        f"  • Best Model      : {checkpoint_manager.path / 'best_model.pth'}"
    )

    logger.info(
        f"  • Last Model      : {checkpoint_manager.path / 'last_model.pth'}"
    )

    logger.info(
        f"  • Metrics         : {checkpoint_manager.path / 'metrics.json'}"
    )

    logger.info(
        f"  • Config          : {checkpoint_manager.path / 'config.yaml'}"
    )

    logger.info(
        f"  • Training Log    : {checkpoint_manager.path / 'train.log'}"
    )

    logger.info(
        f"Training Samples : {len(train_loader.dataset)}"
    )

    logger.info(
        f"Validation Samples : {len(val_loader.dataset)}"
    )

    logger.info(
        f"Test Samples : {len(test_loader.dataset)}"
    )

    logger.info("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()