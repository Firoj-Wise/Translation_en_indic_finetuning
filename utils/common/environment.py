"""
Capture environment info for reproducibility & W&B logging.
"""

import platform
import logging
from typing import Dict

import torch


def get_environment_info() -> Dict[str, str]:
    """
    Collect environment details as a dictionary.

    Returned dict is suitable for:
      - Logging to console
      - Saving as a W&B config artifact
      - Embedding in run_metadata.json
    """
    info: Dict[str, str] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_model"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = str(torch.cuda.device_count())
        info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}"
    else:
        info["cuda_version"] = "N/A (CPU only)"
        info["gpu_model"] = "N/A"
        info["gpu_count"] = "0"

    return info


def log_environment_info(logger: logging.Logger) -> Dict[str, str]:
    """Log environment details and return the info dict."""
    info = get_environment_info()
    logger.info("--- Environment ---")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    logger.info("-------------------")
    return info
