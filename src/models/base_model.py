"""
DeepVision AI

Base Model Interface
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Base class for all DeepVision AI models.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor)

        Returns:
            torch.Tensor
        """
        pass

    @property
    def name(self):
        return self.__class__.__name__