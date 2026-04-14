"""
Unified data loader — dispatches to HuggingFace or local based on config.
"""

import logging
from typing import Any, Dict

import pandas as pd

from constants.types import IngestionSource
from pipeline.ingestion.huggingface_loader import load_from_huggingface
from pipeline.ingestion.local_loader import load_from_local

logger = logging.getLogger("ingestion")


def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the raw parallel corpus from the configured source.

    Parameters
    ----------
    config : dict
        Full pipeline config. Reads ``config["ingestion"]["source"]``.

    Returns
    -------
    pd.DataFrame with columns: src, tgt, src_lang, tgt_lang, domain

    Raises
    ------
    ValueError
        If the source is not recognised.
    """
    source = config["ingestion"]["source"]

    if source == IngestionSource.HUGGINGFACE:
        df = load_from_huggingface(config)

    elif source == IngestionSource.LOCAL:
        df = load_from_local(config)

    else:
        raise ValueError(
            f"Unknown ingestion source: '{source}'. "
            f"Expected 'huggingface' or 'local'."
        )

    # Sanity check required columns
    required = {"src", "tgt", "src_lang", "tgt_lang"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Loaded data is missing required columns: {missing}"
        )

    logger.info(
        f"Ingestion complete -> {len(df)} rows, "
        f"{df['src_lang'].nunique()} source langs, "
        f"{df.get('domain', pd.Series()).nunique() or 1} domains"
    )
    return df
