"""
DeepVision AI

Checkpoint Manager
"""

from pathlib import Path
from datetime import datetime
import json
import shutil

import torch
import yaml


class CheckpointManager:
    """
    Handles experiment tracking, checkpoint saving,
    experiment reproducibility and resume training.
    """

    def __init__(self, root_dir="outputs/runs"):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = Path(root_dir) / f"run_{timestamp}"

        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self):
        return self.run_dir

    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------

    def save_best_model(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch=None,
        best_loss=None,
    ):

        checkpoint = {
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(
            checkpoint,
            self.run_dir / "best_model.pth",
        )

    # --------------------------------------------------
    # Save Last Model
    # --------------------------------------------------

    def save_last_model(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch=None,
        best_loss=None,
    ):

        checkpoint = {
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(
            checkpoint,
            self.run_dir / "last_model.pth",
        )

    # --------------------------------------------------
    # Resume Training
    # --------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        epoch = checkpoint.get("epoch", 0)

        best_loss = checkpoint.get(
            "best_loss",
            float("inf"),
        )

        return epoch, best_loss

    # --------------------------------------------------
    # Save Metrics
    # --------------------------------------------------

    def save_metrics(self, metrics):

        with open(
            self.run_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
            )

    # --------------------------------------------------
    # Save Config
    # --------------------------------------------------

    def save_config(self, config):

        with open(
            self.run_dir / "config.yaml",
            "w",
            encoding="utf-8",
        ) as f:

            yaml.safe_dump(
                config,
                f,
                sort_keys=False,
            )

    # --------------------------------------------------
    # Copy Log
    # --------------------------------------------------

    def copy_log(self, log_path):

        log_path = Path(log_path)

        if log_path.exists():

            shutil.copy(
                log_path,
                self.run_dir / log_path.name,
            )

    # --------------------------------------------------
    # Save Arbitrary File
    # --------------------------------------------------

    def save_file(self, file_path):

        file_path = Path(file_path)

        if file_path.exists():

            shutil.copy(
                file_path,
                self.run_dir / file_path.name,
            )