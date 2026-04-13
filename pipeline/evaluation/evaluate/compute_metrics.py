"""
Trainer-callback metric computation — passed to ``compute_metrics``
in the HF Trainer.
"""

import logging

import numpy as np
from transformers import AutoTokenizer

from pipeline.evaluation.metrics.bleu import compute_bleu
from pipeline.evaluation.metrics.chrf import compute_chrf

logger = logging.getLogger("evaluation.compute_metrics")


def make_compute_metrics(tokenizer: AutoTokenizer):
    """
    Return a ``compute_metrics`` callable compatible with
    ``Seq2SeqTrainer(compute_metrics=...)``.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Used to decode prediction and label token IDs.

    Returns
    -------
    Callable that takes ``(eval_preds)`` and returns a metrics dict.
    """

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )

        # Replace -100 (ignored indices) with pad token for decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Clean up
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels_wrapped = [[label.strip()] for label in decoded_labels]

        bleu = compute_bleu(decoded_preds, decoded_labels_wrapped)
        chrf = compute_chrf(decoded_preds, decoded_labels_wrapped)

        return {
            "bleu": bleu,
            "chrf++": chrf,
        }

    return compute_metrics
