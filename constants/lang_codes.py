"""
Central registry of FLORES-200 language codes used by IndicTrans2.

Prevents scattered string literals and provides human-readable
aliases for the codes used throughout the pipeline.
"""

from typing import Dict, FrozenSet

# ── FLORES-200 tags recognised by IndicTrans2 ────────────────
LANG_CODE_ENGLISH = "eng_Latn"
LANG_CODE_NEPALI = "npi_Deva"
LANG_CODE_MAITHILI = "mai_Deva"

# Canonical set of all supported codes
SUPPORTED_LANG_CODES: FrozenSet[str] = frozenset({
    LANG_CODE_ENGLISH,
    LANG_CODE_NEPALI,
    LANG_CODE_MAITHILI,
})

# Human-readable display names
LANG_DISPLAY_NAMES: Dict[str, str] = {
    LANG_CODE_ENGLISH: "English",
    LANG_CODE_NEPALI: "Nepali",
    LANG_CODE_MAITHILI: "Maithili",
}

# Script family — used by validation rules to decide which
# character-set checks to apply on the target side.
DEVANAGARI_CODES: FrozenSet[str] = frozenset({
    LANG_CODE_NEPALI,
    LANG_CODE_MAITHILI,
})

LATIN_CODES: FrozenSet[str] = frozenset({
    LANG_CODE_ENGLISH,
})


def is_devanagari_target(tgt_lang: str) -> bool:
    """Return True if the target language uses Devanagari script."""
    return tgt_lang in DEVANAGARI_CODES


def display_direction(src_lang: str, tgt_lang: str) -> str:
    """Return a human-readable direction label, e.g. 'English -> Nepali'."""
    src = LANG_DISPLAY_NAMES.get(src_lang, src_lang)
    tgt = LANG_DISPLAY_NAMES.get(tgt_lang, tgt_lang)
    return f"{src} -> {tgt}"
