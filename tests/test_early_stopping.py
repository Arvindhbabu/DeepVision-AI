from src.training.early_stopping import EarlyStopping


def main():

    early_stop = EarlyStopping(
        patience=3,
        min_delta=0.001,
    )

    losses = [
        0.70,
        0.55,
        0.40,
        0.35,
        0.33,
        0.331,
        0.332,
        0.335,
        0.340,
    ]

    print("Validation Loss Progress\n")

    for epoch, loss in enumerate(losses, start=1):

        stop = early_stop(loss)

        print(
            f"Epoch {epoch:02d} | Loss = {loss:.4f}"
        )

        if stop:

            print("\nEarly stopping triggered.")

            break


if __name__ == "__main__":
    main()