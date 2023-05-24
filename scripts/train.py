"""Some training script that uses data module."""
import configlib
from configlib import config as C

import data


# Configuration arguments
parser = configlib.add_parser("Train config")
parser.add_argument("--debug", action="store_true", help="Enable debug mode.")


def train():
    """Main training function."""
    if C["debug"]:
        print("Debugging mode enabled.")
    print("Example dataset:")
    print(data.load_data())


if __name__ == "__main__":
    configlib.parse("last_arguments.txt")
    print("Running with configuration:")
    configlib.print_config()
    train()

