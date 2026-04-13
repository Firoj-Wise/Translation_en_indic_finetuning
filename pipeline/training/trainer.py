"""
Custom Seq2SeqTrainer with warmup-cosine scheduler & AdamW.

When using LoRA, adapter matrices are the only trainable params,
so discriminative LR and gradual unfreezing are intentionally
omitted (they interfere with PEFT).
"""

import math
import logging
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import Seq2SeqTrainer

logger = logging.getLogger("training.trainer")


def _get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """
    Cosine annealing with linear warmup and a minimum learning rate floor.

    The floor prevents the LR from reaching zero in the final steps,
    which helps refine the last few weights.
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


class RobustLoRATrainer(Seq2SeqTrainer):
    """
    Custom trainer wiring in:
      - AdamW with beta2=0.98 (Vaswani et al.)
      - Warmup-cosine scheduler with LR floor

    Config-driven via ``self.training_config`` (set by the caller).
    """

    def __init__(self, training_config: Dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.training_config = training_config or {}

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Build AdamW + warmup-cosine from config values."""
        tcfg = self.training_config

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(
                tcfg.get("adam_beta1", 0.9),
                tcfg.get("adam_beta2", 0.98),
            ),
            eps=tcfg.get("adam_epsilon", 1e-8),
        )

        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        min_lr_ratio = tcfg.get("min_lr_ratio", 0.05)

        self.lr_scheduler = _get_warmup_cosine_scheduler(
            self.optimizer,
            num_warmup_steps,
            num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )

        logger.info(
            f"Optimizer: AdamW  |  "
            f"Warmup steps: {num_warmup_steps}  |  "
            f"Total steps: {num_training_steps}  |  "
            f"Min LR ratio: {min_lr_ratio}"
        )
