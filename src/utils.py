import torch
import numpy as np
import random
import logging
import sys
import platform

def set_seed(seed: int = 42):
    """Pin random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logger(name: str) -> logging.Logger:
    """Setup a standard stdout logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def log_environment_info(logger: logging.Logger):
    """Log environment details for reproducibility via W&B."""
    logger.info("Environment Info:")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available. Training on CPU.")
