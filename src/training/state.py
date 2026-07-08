"""
DeepVision AI

Training State Manager
"""

from dataclasses import dataclass, asdict


@dataclass
class TrainingState:
    """
    Stores the current state of training.

    This object is saved and restored when
    resuming training from checkpoints.
    """

    # -----------------------------
    # Progress
    # -----------------------------

    epoch: int = 0

    global_step: int = 0

    # -----------------------------
    # Best Metrics
    # -----------------------------

    best_val_loss: float = float("inf")

    best_val_accuracy: float = 0.0

    # -----------------------------
    # Current Metrics
    # -----------------------------

    train_loss: float = 0.0

    val_loss: float = 0.0

    train_accuracy: float = 0.0

    val_accuracy: float = 0.0

    # -----------------------------
    # Learning Rate
    # -----------------------------

    learning_rate: float = 0.0

    # -----------------------------
    # Runtime
    # -----------------------------

    epoch_time: float = 0.0

    total_training_time: float = 0.0

    # -----------------------------
    # Flags
    # -----------------------------

    resumed: bool = False

    stopped_early: bool = False

    training_completed: bool = False

    # --------------------------------------------------
    # Convert to Dictionary
    # --------------------------------------------------

    def to_dict(self):
        """
        Convert state to dictionary.
        """

        return asdict(self)

    # --------------------------------------------------
    # Update
    # --------------------------------------------------

    def update(self, **kwargs):
        """
        Update multiple fields.
        """

        for key, value in kwargs.items():

            if hasattr(self, key):

                setattr(self, key, value)

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self):
        """
        Reset runtime values while preserving defaults.
        """

        self.__dict__.update(
            TrainingState().__dict__
        )