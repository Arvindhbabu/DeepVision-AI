from src.utils.config import load_config
from src.models.model_factory import ModelFactory


def main():

    config = load_config()

    model = ModelFactory.create(config)

    print("\nModel Loaded Successfully\n")

    print(type(model))

    print(model.name)


if __name__ == "__main__":
    main()