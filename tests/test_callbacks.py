from src.training.callbacks import Callback
from src.training.callbacks import CallbackManager


class PrintCallback(Callback):

    def on_train_begin(self, trainer):
        print("Training Started")

    def on_epoch_begin(self, trainer):
        print("Epoch Started")

    def on_epoch_end(self, trainer):
        print("Epoch Finished")

    def on_validation_end(self, trainer):
        print("Validation Finished")

    def on_train_end(self, trainer):
        print("Training Finished")


def main():

    manager = CallbackManager()

    manager.add(PrintCallback())

    trainer = object()

    manager.on_train_begin(trainer)

    manager.on_epoch_begin(trainer)

    manager.on_epoch_end(trainer)

    manager.on_validation_end(trainer)

    manager.on_train_end(trainer)


if __name__ == "__main__":
    main()