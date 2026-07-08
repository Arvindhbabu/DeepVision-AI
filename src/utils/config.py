"""
Configuration utilities for DeepVision AI

Loads and validates YAML configuration files.
"""

from pathlib import Path
import yaml


def load_config(config_path: str = "configs/default.yaml"):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML configuration.

    Returns:
        dict: Configuration dictionary.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def get_path(config, key):
    """
    Return project path from config.
    """

    return Path(config["paths"][key])


def print_config(config):
    """
    Nicely print configuration.
    """

    print("=" * 60)
    print(" DeepVision AI Configuration")
    print("=" * 60)

    for section, values in config.items():

        print(f"\n[{section}]")

        if isinstance(values, dict):
            for k, v in values.items():
                print(f"{k:20} : {v}")

        else:
            print(values)

    print("=" * 60)