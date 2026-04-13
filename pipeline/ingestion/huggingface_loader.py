"""
Load translation datasets from HuggingFace Hub.

Placeholder implementation — replace ``dataset_name`` in
``configs/default.yaml`` with the real HF dataset identifier
when available.
"""

import logging
from typing import Any, Dict

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger("ingestion.huggingface")


def load_from_huggingface(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Pull a parallel corpus from HuggingFace and normalise columns
    to the standard schema: ``{src, tgt, src_lang, tgt_lang, domain}``.

    Parameters
    ----------
    config : dict
        The full pipeline config. Reads from ``config["ingestion"]["huggingface"]``
        for dataset_name, subset, split, and column mappings.

    Returns
    -------
    pd.DataFrame with columns: src, tgt, src_lang, tgt_lang, domain
    """
    hf_cfg = config["ingestion"]["huggingface"]

    dataset_name = hf_cfg["dataset_name"]
    subset = hf_cfg.get("subset")
    split = hf_cfg.get("split", "train")

    logger.info(f"Loading HuggingFace dataset: {dataset_name} "
                f"(subset={subset}, split={split})")

    ds = load_dataset(dataset_name, subset, split=split)

    # Map HF column names → standard schema
    col_map = {
        hf_cfg.get("src_column", "src"): "src",
        hf_cfg.get("tgt_column", "tgt"): "tgt",
        hf_cfg.get("src_lang_column", "src_lang"): "src_lang",
        hf_cfg.get("tgt_lang_column", "tgt_lang"): "tgt_lang",
        hf_cfg.get("domain_column", "domain"): "domain",
    }

    df = ds.to_pandas()

    # Only rename columns that actually exist in the dataset
    existing_renames = {
        k: v for k, v in col_map.items()
        if k in df.columns
    }
    df = df.rename(columns=existing_renames)

    # Ensure required columns are present
    required = {"src", "tgt", "src_lang", "tgt_lang"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"HuggingFace dataset is missing required columns "
            f"after mapping: {missing}. Available: {list(df.columns)}"
        )

    # Fill optional domain column
    if "domain" not in df.columns:
        df["domain"] = "general"

    # Apply sample size limit if configured (useful for debug runs)
    sample_size = config["ingestion"].get("sample_size")
    if sample_size and len(df) > sample_size:
        seed = config["pipeline"].get("seed", 42)
        df = df.sample(n=sample_size, random_state=seed)
        logger.info(f"Sampled {sample_size} rows for debug run")

    logger.info(f"Loaded {len(df)} rows from HuggingFace")
    return df.reset_index(drop=True)
