import torch

from src.training.metrics import Metrics


def main():

    metrics = Metrics()

    outputs = torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )

    labels = torch.tensor(
        [
            1,
            0,
            1,
            0,
        ]
    )

    metrics.update(
        outputs,
        labels,
    )

    results = metrics.compute()

    metrics.print(results)


if __name__ == "__main__":

    main()