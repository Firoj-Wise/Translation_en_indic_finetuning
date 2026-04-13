"""
Reproducibility — pin every source of randomness.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Pin random seeds for full reproducibility.

    Sets seeds on: Python stdlib, NumPy, PyTorch (CPU + all CUDA devices).
    Also enables deterministic cuDNN for exact reproducibility
    (marginally slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
