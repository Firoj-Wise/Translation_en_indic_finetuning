"""
Custom HuggingFace TrainerCallbacks for enhanced logging.

- **VersioningCallback**: saves run metadata at each checkpoint.
- **GPUMonitorCallback**: logs GPU utilisation to W&B every N steps.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger("training.callbacks")


class VersioningCallback(TrainerCallback):
    """
    Save ``run_metadata.json`` alongside each checkpoint.

    The metadata dict is provided at init time and updated
    with the current step/epoch on each save.
    """

    def __init__(self, run_metadata: Dict[str, Any]) -> None:
        self.run_metadata = run_metadata

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Write metadata into the checkpoint directory
        ckpt_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if os.path.isdir(ckpt_dir):
            meta = dict(self.run_metadata)
            meta["global_step"] = state.global_step
            meta["epoch"] = state.epoch

            meta_path = os.path.join(ckpt_dir, "run_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            logger.debug(f"Versioning metadata saved -> {meta_path}")


class GPUMonitorCallback(TrainerCallback):
    """
    Log GPU utilisation and memory to the active experiment tracker
    every ``log_interval`` steps.
    """

    def __init__(self, log_interval: int = 50) -> None:
        self.log_interval = log_interval
        self._psutil_available = False
        self._torch_cuda = False

        try:
            import psutil  # noqa: F401
            self._psutil_available = True
        except ImportError:
            pass

        import torch
        self._torch_cuda = torch.cuda.is_available()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        if not self._torch_cuda:
            return
        if state.global_step % self.log_interval != 0:
            return

        import torch

        # GPU memory
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9

        gpu_metrics = {
            "gpu/memory_allocated_gb": round(allocated, 2),
            "gpu/memory_reserved_gb": round(reserved, 2),
            "gpu/memory_total_gb": round(total, 2),
            "gpu/memory_util_pct": round(allocated / total * 100, 1),
        }

        # CPU memory
        if self._psutil_available:
            import psutil
            vm = psutil.virtual_memory()
            gpu_metrics["system/ram_used_pct"] = vm.percent

        if logs is not None:
            logs.update(gpu_metrics)
