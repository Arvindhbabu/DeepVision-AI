"""
DeepVision AI

Base Trainer

Defines the common functionality shared by all trainers.
"""
from src.training.state import TrainingState
from abc import ABC, abstractmethod
import time

import torch

from src.training.history import History
from src.training.metrics import Metrics
from src.training.callbacks import CallbackManager


class BaseTrainer(ABC):
    """
    Base class for all DeepVision AI trainers.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        logger=None,
        scheduler=None,
        checkpoint_manager=None,
        early_stopping=None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.logger = logger
        self.scheduler = scheduler

        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping

        # Training state
        self.state = TrainingState()

        # Utilities
        self.history = History()
        self.metrics = Metrics()
        self.callbacks = CallbackManager()

        # AMP (safe on CPU)
        self.use_amp = (
            torch.cuda.is_available()
            and device.type == "cuda"
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp
        )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def log(self, message):

        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    # --------------------------------------------------
    # Current Learning Rate
    # --------------------------------------------------

    def get_learning_rate(self):

        return self.optimizer.param_groups[0]["lr"]

    # --------------------------------------------------
    # Epoch Timer
    # --------------------------------------------------

    def start_timer(self):

        self._start_time = time.time()

    def stop_timer(self):

        return time.time() - self._start_time

    # --------------------------------------------------
    # Abstract Methods
    # --------------------------------------------------

    @abstractmethod
    def train_one_epoch(self):
        """
        Train for one epoch.
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Validate one epoch.
        """
        pass

    @abstractmethod
    def train(self, epochs):
        """
        Complete training loop.
        """
        pass