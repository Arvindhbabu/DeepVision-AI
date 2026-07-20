from src.training.history import History


def main():
    history = History()

    history.add(
        epoch=1,
        train_loss=0.40,
        val_loss=0.35,
        train_acc=92.5,
        val_acc=91.8,
    )

    history.add(
        epoch=2,
        train_loss=0.32,
        val_loss=0.28,
        train_acc=95.2,
        val_acc=94.7,
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