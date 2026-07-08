"""
DeepVision AI

Early Stopping Utility
"""

import numpy as np


class EarlyStopping:
    """
    Stops training when validation loss
    stops improving.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = np.inf

        self.counter = 0

        self.should_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop