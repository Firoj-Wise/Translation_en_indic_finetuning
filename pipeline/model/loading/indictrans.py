"""
IndicTrans2 model and tokenizer loading — Standard LoRA & QLoRA.

All configuration is read from the YAML config so this module
has zero hard-coded hyperparameters.
"""

import logging
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger("model.loading")


def load_tokenizer(config: Dict[str, Any]) -> AutoTokenizer:
    """
    Load the IndicTrans2 tokenizer.

    Parameters
    ----------
    config : dict
        Full pipeline config. Reads ``config["model"]``.

    Returns
    -------
    AutoTokenizer
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    trust = model_cfg.get("trust_remote_code", True)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust,
    )
    return tokenizer


def load_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Load the IndicTrans2 model with optional LoRA / QLoRA adapters.

    Parameters
    ----------
    config : dict
        Full pipeline config. Reads ``config["model"]``.

    Returns
    -------
    torch.nn.Module
        PEFT-wrapped model (if LoRA enabled) or base model.
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    trust = model_cfg.get("trust_remote_code", True)

    # ── QLoRA (4-bit quantised loading) ───────────────────────
    qlora_cfg = model_cfg.get("qlora", {})
    if qlora_cfg.get("enabled", False):
        logger.info("Loading model with QLoRA (4-bit quantisation)")
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        compute_dtype = getattr(torch, qlora_cfg.get("compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora_cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=qlora_cfg.get("double_quant", True),
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust,
        )
        # Cast LayerNorm to fp32 — critical step
        model = prepare_model_for_kbit_training(model)

    else:
        # ── Standard loading (full precision) ─────────────────
        logger.info(f"Loading model: {model_name} (full precision)")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=trust,
        )

    # ── Enable input grads for gradient checkpointing ─────────
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(
            _make_inputs_require_grad
        )

    # ── Apply LoRA adapters ───────────────────────────────────
    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        rank = lora_cfg.get("rank", 16)
        alpha = lora_cfg.get("alpha", rank * 2)
        dropout = lora_cfg.get("dropout", 0.05)
        bias = lora_cfg.get("bias", "none")
        targets = lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2",
        ])

        logger.info(
            f"Applying LoRA: r={rank}, α={alpha}, "
            f"dropout={dropout}, targets={targets}"
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=targets,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def load_tokenizer_and_model(
    config: Dict[str, Any],
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """Convenience wrapper to load both tokenizer and model."""
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    return tokenizer, model
