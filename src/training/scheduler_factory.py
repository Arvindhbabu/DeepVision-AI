"""
DeepVision AI

Scheduler Factory
"""

import torch


class SchedulerFactory:
    """
    Creates LR scheduler from configuration.
    """

    @staticmethod
    def create(optimizer, config):

        scheduler_name = config["training"]["scheduler"].lower()

        epochs = config["training"]["epochs"]

        if scheduler_name == "cosineannealinglr":

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
            )

        if scheduler_name == "steplr":

            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1,
            )

        if scheduler_name == "reducelronplateau":

            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=3,
            )

        return None