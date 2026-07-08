"""
DeepVision AI

Optimizer Factory
"""

import torch


class OptimizerFactory:
    """
    Creates optimizer from configuration.
    """

    @staticmethod
    def create(model, config):

        optimizer_name = config["training"]["optimizer"].lower()

        lr = config["training"]["learning_rate"]

        weight_decay = config["training"]["weight_decay"]

        trainable_params = filter(
            lambda p: p.requires_grad,
            model.parameters()
        )

        if optimizer_name == "adam":

            return torch.optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adamw":

            return torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "sgd":

            return torch.optim.SGD(
                trainable_params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )

        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )