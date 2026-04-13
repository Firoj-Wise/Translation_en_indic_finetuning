"""
chrF++ wrapper — better metric for Devanagari languages.

Uses ``word_order=2`` by default (chrF++ rather than plain chrF),
which is the standard in low-resource MT evaluation.
"""

from typing import List

import evaluate


_metric = None


def _get_metric():
    global _metric
    if _metric is None:
        _metric = evaluate.load("chrf")
    return _metric


def compute_chrf(
    predictions: List[str],
    references: List[List[str]],
    word_order: int = 2,
) -> float:
    """
    Compute corpus-level chrF++ score.

    Parameters
    ----------
    predictions : list[str]
        System outputs.
    references : list[list[str]]
        Reference translations.
    word_order : int
        Word n-gram order. 2 = chrF++. Default: 2.

    Returns
    -------
    float
        chrF++ score rounded to 2 decimal places.
    """
    metric = _get_metric()
    result = metric.compute(
        predictions=predictions,
        references=references,
        word_order=word_order,
    )
    return round(result["score"], 2)
