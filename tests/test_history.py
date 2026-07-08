from src.training.history import History


def main():

    history = History()

    history.add(
        epoch=1,
        train_loss=0.45,
        val_loss=0.41,
        train_accuracy=92.5,
        val_accuracy=91.8,
    )

    history.add(
        epoch=2,
        train_loss=0.32,
        val_loss=0.28,
        train_accuracy=95.2,
        val_accuracy=94.7,
    )

    print("History Length")

    print(len(history))

    print()

    print("Best Epoch")

    print(history.best())

    history.save_json(
        "outputs/history.json"
    )

    history.save_csv(
        "outputs/history.csv"
    )

    print()

    print("History Saved")


if __name__ == "__main__":
    main()