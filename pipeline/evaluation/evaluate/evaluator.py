"""
Full post-training evaluator.

Runs on the locked test set, evaluates ALL directions separately,
and optionally breaks down by domain.

Results are returned as a list of :class:`EvalResult` objects and
optionally logged to the tracker.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

from constants.lang_codes import display_direction
from constants.types import EvalResult
from pipeline.evaluation.metrics.bleu import compute_bleu
from pipeline.evaluation.metrics.chrf import compute_chrf

logger = logging.getLogger("evaluation.evaluator")


def evaluate_all_directions(
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    model: torch.nn.Module = None,
    tokenizer: AutoTokenizer = None,
) -> List[EvalResult]:
    """
    Evaluate the final merged model on every (src_lang, tgt_lang)
    direction present in ``test_df``.

    Parameters
    ----------
    test_df : pd.DataFrame
        Locked test data with columns: src, tgt, src_lang, tgt_lang, domain.
    config : dict
        Full pipeline config.
    model : torch.nn.Module, optional
        Pre-loaded model. If None, loads from ``config["training"]["final_model_dir"]``.
    tokenizer : AutoTokenizer, optional
        Pre-loaded tokenizer. If None, loads alongside model.

    Returns
    -------
    List[EvalResult]
        One result per direction.
    """
    eval_cfg = config.get("evaluation", {})
    batch_size = eval_cfg.get("batch_size", 32)
    num_beams = eval_cfg.get("num_beams", 5)
    max_length = eval_cfg.get("max_length", 256)

    # Load model if not provided
    if model is None or tokenizer is None:
        model_dir = config["training"].get(
            "final_model_dir", "./indictrans2-finetuned-final"
        )
        trust = config["model"].get("trust_remote_code", True)
        logger.info(f"Loading merged model from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=trust
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, trust_remote_code=trust
        )

    device = 0 if torch.cuda.is_available() else -1
    results: List[EvalResult] = []

    for (src_lang, tgt_lang), group_df in test_df.groupby(
        ["src_lang", "tgt_lang"]
    ):
        direction = display_direction(src_lang, tgt_lang)
        logger.info(f"Evaluating {direction} ({len(group_df)} samples) ...")

        translator = hf_pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
        )

        sources = group_df["src"].tolist()
        references = [[t] for t in group_df["tgt"].tolist()]

        predictions = [
            r["translation_text"] for r in translator(sources)
        ]

        bleu = compute_bleu(predictions, references)
        chrf = compute_chrf(predictions, references)

        result = EvalResult(
            direction=direction,
            n_samples=len(sources),
            bleu=bleu,
            chrf_pp=chrf,
        )
        results.append(result)

        logger.info(
            f"  {direction}: BLEU={bleu:.2f}  chrF++={chrf:.2f}"
        )

    # ── Summary table ─────────────────────────────────────────
    logger.info("--- Evaluation Summary ---")
    for r in results:
        logger.info(
            f"  {r.direction:>25s}  |  "
            f"n={r.n_samples:<6}  "
            f"BLEU={r.bleu:<7.2f}  "
            f"chrF++={r.chrf_pp:<7.2f}"
        )
    logger.info("--------------------------")

    return results
