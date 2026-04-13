"""
Composable transform chain builder.

Builds a sequential list of DataFrame → DataFrame transforms
from the config, applied before tokenization.
"""

import logging
from typing import Any, Callable, Dict, List

import pandas as pd

from pipeline.preprocessing.normalization import normalize_text

logger = logging.getLogger("preprocessing.transform")

# Type alias
Transform = Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]


def _normalize_transform(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Apply Unicode normalization to src and tgt columns."""
    form = config["preprocessing"].get("normalization", "NFC")
    logger.info(f"Applying {form} normalization ...")
    df["src"] = df["src"].apply(lambda t: normalize_text(t, form))
    df["tgt"] = df["tgt"].apply(lambda t: normalize_text(t, form))
    return df


def build_transforms(
    config: Dict[str, Any],
) -> List[Transform]:
    """
    Construct the ordered preprocessing transform chain.

    Currently just normalization, but designed to be extended
    (e.g., transliteration, length-bucketing, etc.).

    Parameters
    ----------
    config : dict
        Full pipeline config.

    Returns
    -------
    List of ``(df, config) → df`` callables.
    """
    transforms: List[Transform] = [
        _normalize_transform,
    ]
    return transforms


def apply_transforms(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Run all transforms in sequence."""
    transforms = build_transforms(config)
    for tfm in transforms:
        df = tfm(df, config)
    return df
