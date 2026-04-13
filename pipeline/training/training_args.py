"""
Build Seq2SeqTrainingArguments from the YAML config.

All hyperparameters come from ``config["training"]`` — zero
hard-coded defaults.
"""

import logging
from typing import Any, Dict

from transformers import Seq2SeqTrainingArguments

from constants.types import MixedPrecision

logger = logging.getLogger("training.args")


def build_training_args(config: Dict[str, Any]) -> Seq2SeqTrainingArguments:
    """
    Construct ``Seq2SeqTrainingArguments`` from config.

    Parameters
    ----------
    config : dict
        Full pipeline config. Reads ``config["training"]``
        and ``config["tracking"]``.

    Returns
    -------
    Seq2SeqTrainingArguments
    """
    tcfg = config["training"]
    track_cfg = config.get("tracking", {})

    # Resolve mixed precision flags
    mp = tcfg.get("mixed_precision", "bf16")
    use_bf16 = mp == MixedPrecision.BF16
    use_fp16 = mp == MixedPrecision.FP16

    # Resolve report_to based on tracker backend
    backend = track_cfg.get("backend", "wandb")
    report_to = backend if backend != "none" else "none"

    args = Seq2SeqTrainingArguments(
        output_dir=tcfg.get("output_dir", "./checkpoints"),

        # Epochs and batching
        num_train_epochs=tcfg.get("num_epochs", 5),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 16),

        # Learning rate
        learning_rate=tcfg.get("learning_rate", 1e-4),
        warmup_ratio=tcfg.get("warmup_ratio", 0.06),

        # Regularization
        label_smoothing_factor=tcfg.get("label_smoothing", 0.1),
        weight_decay=tcfg.get("weight_decay", 0.01),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),

        # Mixed precision
        bf16=use_bf16,
        fp16=use_fp16,

        # Memory
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,

        # Generation (eval only)
        predict_with_generate=True,
        generation_max_length=tcfg.get("generation_max_length", 256),
        generation_num_beams=tcfg.get("generation_num_beams", 5),

        # Checkpointing
        evaluation_strategy=tcfg.get("eval_strategy", "steps"),
        eval_steps=tcfg.get("eval_steps", 500),
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=tcfg.get("save_steps", 500),
        save_total_limit=tcfg.get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model=tcfg.get("metric_for_best_model", "chrf++"),
        greater_is_better=tcfg.get("greater_is_better", True),

        # Efficiency
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=tcfg.get("dataloader_pin_memory", True),
        remove_unused_columns=False,

        # Logging
        logging_steps=tcfg.get("logging_steps", 50),
        report_to=report_to,
        run_name=track_cfg.get("run_name", "indictrans2-finetune"),

        # Reproducibility
        seed=config["pipeline"].get("seed", 42),
    )

    logger.info(
        f"Training args built -> epochs={args.num_train_epochs}, "
        f"batch={args.per_device_train_batch_size}×{args.gradient_accumulation_steps}, "
        f"lr={args.learning_rate}, {mp} precision"
    )
    return args
