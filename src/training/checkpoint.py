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
    Handles experiment tracking and checkpoint saving.
    """

    def __init__(self, root_dir="outputs/runs"):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = Path(root_dir) / f"run_{timestamp}"

        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self):
        return self.run_dir

    def save_best_model(self, model):

        torch.save(
            model.state_dict(),
            self.run_dir / "best_model.pth",
        )

    def save_last_model(self, model):

        torch.save(
            model.state_dict(),
            self.run_dir / "last_model.pth",
        )

    def save_metrics(self, metrics):

        with open(
            self.run_dir / "metrics.json",
            "w",
        ) as f:

            json.dump(metrics, f, indent=4)

    def save_config(self, config):

        with open(
            self.run_dir / "config.yaml",
            "w",
        ) as f:

            yaml.dump(config, f)

    def copy_log(self, log_path):

        log_path = Path(log_path)

        if log_path.exists():

            shutil.copy(
                log_path,
                self.run_dir / log_path.name,
            )