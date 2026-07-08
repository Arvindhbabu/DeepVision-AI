from src.datasets.split_generator import DatasetSplitGenerator


def main():

    generator = DatasetSplitGenerator()

    generator.generate()


if __name__ == "__main__":
    main()