from src.training.state import TrainingState


def main():

    state = TrainingState()

    state.update(

        epoch=5,

        global_step=340,

        best_val_loss=0.248,

        best_val_accuracy=94.75,

        learning_rate=1e-4,

        resumed=True,

    )

    print()

    print("Training State")

    print("=" * 50)

    for key, value in state.to_dict().items():

        print(f"{key:25}: {value}")

    print("=" * 50)


if __name__ == "__main__":

    main()