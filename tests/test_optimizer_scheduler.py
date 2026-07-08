import torch.nn as nn

from src.utils.config import load_config
from src.training.optimizer_factory import OptimizerFactory
from src.training.scheduler_factory import SchedulerFactory


def main():

    config = load_config()

    model = nn.Linear(10, 2)

    optimizer = OptimizerFactory.create(
        model,
        config,
    )

    scheduler = SchedulerFactory.create(
        optimizer,
        config,
    )

    print("Optimizer")
    print(type(optimizer))

    print()

    print("Scheduler")
    print(type(scheduler))


if __name__ == "__main__":
    main()