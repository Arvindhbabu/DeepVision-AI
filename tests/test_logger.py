from src.utils.logger import create_logger


def main():

    logger = create_logger()

    logger.info("DeepVision AI Logger Initialized")

    logger.info("Training Started")

    logger.warning("Sample Warning")

    logger.error("Sample Error")

    print("\nLogger Test Complete")


if __name__ == "__main__":
    main()