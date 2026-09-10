"""
seed.py

Util function for managing reproducibility with random events
"""


import random

import numpy as np
import torch


BASE_SEED = 42


def set_seed(seed: int = BASE_SEED) -> None:
    """
    Set the seed to manage reproducibility.

    Parameter
    ---------
    seed: int, optional
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    