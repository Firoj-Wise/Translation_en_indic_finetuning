"""
sacrebleu wrapper — corpus-level BLEU computation.
"""

from typing import List

import evaluate


_metric = None


def _get_metric():
    global _metric
    if _metric is None:
        _metric = evaluate.load("sacrebleu")
    return _metric


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
) -> float:
    """
    Compute corpus-level BLEU via sacrebleu.

    Parameters
    ----------
    predictions : list[str]
        System outputs.
    references : list[list[str]]
        Reference translations (each a list of 1+ references).

    Returns
    -------
    float
        BLEU score rounded to 2 decimal places.
    """
    metric = _get_metric()
    result = metric.compute(predictions=predictions, references=references)
    return round(result["score"], 2)
