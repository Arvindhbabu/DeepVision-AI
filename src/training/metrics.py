"""
DeepVision AI

Metrics Engine

Computes classification metrics for training,
validation and evaluation.
"""

from typing import Dict

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Metrics:
    """
    Metrics accumulator.

    Example
    -------
    metrics = Metrics()

    metrics.update(outputs, labels)

    result = metrics.compute()

    metrics.reset()
    """

    def __init__(self):

        self.reset()

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self):

        self.targets = []

        self.predictions = []

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):

        preds = torch.argmax(
            outputs,
            dim=1,
        )

        self.predictions.extend(
            preds.detach().cpu().numpy().tolist()
        )

        self.targets.extend(
            labels.detach().cpu().numpy().tolist()
        )

    # --------------------------------------------------------
    # Compute
    # --------------------------------------------------------

    def compute(self) -> Dict[str, float]:

        y_true = np.array(self.targets)

        y_pred = np.array(self.predictions)

        results = {

            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),

            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        }

        return results

    # --------------------------------------------------------
    # Pretty Print
    # --------------------------------------------------------

    @staticmethod
    def print(results: Dict[str, float]):

        print("=" * 50)

        print("DeepVision AI Metrics")

        print("=" * 50)

        for k, v in results.items():

            print(f"{k:12}: {v:.4f}")

        print("=" * 50)