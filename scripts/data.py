"""Some data loading module."""
from typing import List
import random

import configlib
from configlib import config as C

# Configuration arguments
parser = configlib.add_parser("Dataset config")
parser.add_argument("--data_length", default=4, type=int, help="Length of random list.")


def load_data() -> List[int]:
    """Load some random data."""
    data = list(range(C["data_length"]))
    random.shuffle(data)
    return data
