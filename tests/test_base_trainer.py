from src.training.base_trainer import BaseTrainer


class DummyTrainer(BaseTrainer):

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def train(self, epochs):
        pass


def main():

    print("BaseTrainer imported successfully")

    print(BaseTrainer)

    print(DummyTrainer)


if __name__ == "__main__":
    main()