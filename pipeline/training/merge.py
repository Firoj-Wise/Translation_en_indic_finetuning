"""
LoRA merge, save, and optional push-to-hub.
"""

import logging
from typing import Any, Dict

import torch
from transformers import AutoTokenizer

logger = logging.getLogger("training.merge")


def merge_and_save(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    config: Dict[str, Any],
) -> str:
    """
    Merge LoRA adapters back into the base model, save locally,
    and optionally push to HuggingFace Hub.

    Parameters
    ----------
    model : torch.nn.Module
        The PEFT-wrapped model (post-training).
    tokenizer : AutoTokenizer
        Tokenizer to save alongside the model.
    config : dict
        Full pipeline config. Reads ``config["training"]`` and ``config["hub"]``.

    Returns
    -------
    str
        Path to the saved merged model directory.
    """
    output_dir = config["training"].get(
        "final_model_dir", "./indictrans2-finetuned-final"
    )

    logger.info("Merging LoRA weights into base model ...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model -> {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Push to hub ───────────────────────────────────────────
    hub_cfg = config.get("hub", {})
    if hub_cfg.get("push_to_hub", False):
        hub_id = hub_cfg.get("model_id")
        if hub_id:
            logger.info(f"Pushing to HuggingFace Hub: {hub_id}")
            merged_model.push_to_hub(hub_id)
            tokenizer.push_to_hub(hub_id)
        else:
            logger.warning(
                "push_to_hub=True but no hub.model_id configured — skipping."
            )

    return output_dir
