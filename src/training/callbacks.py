"""
DeepVision AI

Training Callback System
"""


class Callback:
    """
    Base callback class.
    """

    def on_train_begin(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass


class CallbackManager:
    """
    Manages all callbacks.
    """

    def __init__(self):

        self.callbacks = []

    def add(self, callback):

        self.callbacks.append(callback)

    def on_train_begin(self, trainer):

        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_epoch_begin(self, trainer):

        for cb in self.callbacks:
            cb.on_epoch_begin(trainer)

    def on_epoch_end(self, trainer):

        for cb in self.callbacks:
            cb.on_epoch_end(trainer)

    def on_validation_end(self, trainer):

        for cb in self.callbacks:
            cb.on_validation_end(trainer)

    def on_train_end(self, trainer):

        for cb in self.callbacks:
            cb.on_train_end(trainer)