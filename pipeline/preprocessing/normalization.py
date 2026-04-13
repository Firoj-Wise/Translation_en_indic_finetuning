"""
Text normalization for Devanagari and Latin scripts.

All normalization is applied BEFORE tokenization to ensure
consistent subword splits.
"""

import re
import unicodedata


# Pre-compiled patterns (zero-width chars + whitespace collapse)
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str, form: str = "NFC") -> str:
    """
    Apply Unicode normalization, strip zero-width characters,
    and collapse whitespace.

    Parameters
    ----------
    text : str
        Raw input text.
    form : str
        Unicode normalization form. Default ``"NFC"`` (canonical
        composition) — the only safe choice for Devanagari.
        **Never** use ``"NFKD"`` — it decomposes Devanagari characters.

    Returns
    -------
    str
        Cleaned, normalized text.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize(form, text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
