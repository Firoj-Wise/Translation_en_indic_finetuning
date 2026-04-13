"""
Individual validation rules for translation pairs.

Each rule is a callable ``(row: pd.Series) → bool`` returning True if
the row PASSES (is valid). Rules can be composed via :func:`build_rules`.
"""

import re
from typing import Callable

import pandas as pd

from constants.lang_codes import is_devanagari_target

# Type alias for a rule function
Rule = Callable[[pd.Series], bool]


def min_length_rule(min_len: int = 3) -> Rule:
    """Reject pairs where either side has fewer words than ``min_len``."""
    def _check(row: pd.Series) -> bool:
        return (
            len(row["src"].split()) >= min_len
            and len(row["tgt"].split()) >= min_len
        )
    _check.__name__ = "min_length"
    return _check


def max_length_rule(max_len: int = 200) -> Rule:
    """Reject pairs where either side has more words than ``max_len``."""
    def _check(row: pd.Series) -> bool:
        return (
            len(row["src"].split()) <= max_len
            and len(row["tgt"].split()) <= max_len
        )
    _check.__name__ = "max_length"
    return _check


def length_ratio_rule(max_ratio: float = 3.5) -> Rule:
    """Reject pairs where the word-count ratio exceeds ``max_ratio``."""
    def _check(row: pd.Series) -> bool:
        s_len = len(row["src"].split())
        t_len = len(row["tgt"].split())
        ratio = max(s_len, t_len) / max(min(s_len, t_len), 1)
        return ratio <= max_ratio
    _check.__name__ = "length_ratio"
    return _check


def not_identical_rule() -> Rule:
    """Reject pairs where source and target are identical (copy-paste)."""
    def _check(row: pd.Series) -> bool:
        return row["src"].strip() != row["tgt"].strip()
    _check.__name__ = "not_identical"
    return _check


def not_empty_or_numeric_rule() -> Rule:
    """
    Reject pairs where either side is empty or contains only
    numbers/punctuation (no real language content).
    """
    _pattern = re.compile(r"^[\d\s\W]+$")

    def _check(row: pd.Series) -> bool:
        src, tgt = row["src"].strip(), row["tgt"].strip()
        if not src or not tgt:
            return False
        if _pattern.match(src) or _pattern.match(tgt):
            return False
        return True
    _check.__name__ = "not_empty_or_numeric"
    return _check


def devanagari_ascii_ratio_rule(max_ratio: float = 0.20) -> Rule:
    """
    For Devanagari targets, reject if ASCII characters comprise more
    than ``max_ratio`` of the total characters.  This catches mislabelled
    or heavily code-switched pairs.
    """
    def _check(row: pd.Series) -> bool:
        tgt_lang = row.get("tgt_lang", "")
        if not is_devanagari_target(tgt_lang):
            return True  # Rule only applies to Devanagari targets

        tgt = row["tgt"].strip()
        if not tgt:
            return False

        ascii_count = sum(1 for c in tgt if c.isascii() and c.isalpha())
        total = len(tgt.replace(" ", ""))
        if total == 0:
            return False

        return (ascii_count / total) <= max_ratio
    _check.__name__ = "devanagari_ascii_ratio"
    return _check
