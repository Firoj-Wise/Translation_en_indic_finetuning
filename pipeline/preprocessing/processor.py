"""
Data processor — splitting, tokenization, and dataset assembly.

Orchestrates:
  1. Train / Val / Test split (stratified by direction + domain)
  2. Per-direction tokenization with the IndicTrans2 tokenizer
  3. Test-set locking (saved once, never modified)
"""

import logging
import os
from functools import partial
from typing import Any, Dict, Tuple

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from constants.types import SplitStats
from pipeline.preprocessing.build_transform import apply_transforms

logger = logging.getLogger("preprocessing")


# ── Tokenization helpers ──────────────────────────────────────

def _tokenize_batch(
    batch: Dict,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    max_length: int,
) -> Dict:
    """
    Tokenize a batch of examples for one language direction.

    The IndicTrans2 tokenizer expects the format:
        ``"src_lang tgt_lang <text>"``
    passed to ``__call__`` when in source mode.
    """
    formatted_srcs = [
        f"{src_lang} {tgt_lang} {text}" for text in batch["src"]
    ]

    inputs = tokenizer(
        formatted_srcs,
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            batch["tgt"],
            max_length=max_length,
            truncation=True,
            padding=False,
        )

    inputs["labels"] = targets["input_ids"]
    return inputs


def _tokenize_direction_group(
    group_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    num_proc: int,
) -> Dataset:
    """Tokenize a single direction group and return a HF Dataset."""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    ds = Dataset.from_pandas(group_df)

    tokenized = ds.map(
        partial(
            _tokenize_batch,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=max_length,
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
    )
    return tokenized


# ── Main processor ────────────────────────────────────────────

def process(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    config: Dict[str, Any],
) -> Tuple[Dataset, Dataset, pd.DataFrame, SplitStats]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Validated data.
    tokenizer : AutoTokenizer
        IndicTrans2 tokenizer.
    config : dict
        Full pipeline config.

    Returns
    -------
    (train_dataset, val_dataset, test_df, split_stats)
        train/val are tokenized HF Datasets.
        test_df is the locked raw DataFrame (for post-training eval).
    """
    seed = config["pipeline"].get("seed", 42)
    prep_cfg = config["preprocessing"]
    max_length = prep_cfg.get("max_seq_length", 256)
    num_proc = prep_cfg.get("tokenizer_num_proc", 4)
    ratios = prep_cfg.get("split_ratios", {})

    # ── 1. Apply transforms (normalization etc.) ──────────────
    df = apply_transforms(df, config)

    # ── 2. Shuffle then split ─────────────────────────────────
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_ratio = ratios.get("train", 0.90)
    val_ratio = ratios.get("val", 0.05)

    train_end = int(train_ratio * len(df))
    val_end = int((train_ratio + val_ratio) * len(df))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # ── 3. Lock test set ──────────────────────────────────────
    output_dir = config["training"].get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    test_path = os.path.join(output_dir, "test_locked.csv")
    test_df.to_csv(test_path, index=False)
    logger.info(f"Test set locked -> {test_path} ({len(test_df):,} rows)")

    # ── 4. Log split statistics ───────────────────────────────
    directions = sorted(
        df.apply(
            lambda r: f"{r['src_lang']} -> {r['tgt_lang']}", axis=1
        ).unique().tolist()
    )
    domains = sorted(df["domain"].unique().tolist()) if "domain" in df else []

    stats = SplitStats(
        train=len(train_df),
        val=len(val_df),
        test=len(test_df),
        directions=directions,
        domains=domains,
    )
    logger.info(
        f"Splits -> Train: {stats.train:,}  "
        f"Val: {stats.val:,}  Test: {stats.test:,}"
    )

    # ── 5. Tokenize per direction ─────────────────────────────
    def _tokenize_split(split_df: pd.DataFrame) -> Dataset:
        datasets = []
        for (src_lang, tgt_lang), group_df in split_df.groupby(
            ["src_lang", "tgt_lang"]
        ):
            tok_ds = _tokenize_direction_group(
                group_df, tokenizer, src_lang, tgt_lang,
                max_length, num_proc,
            )
            datasets.append(tok_ds)
        return concatenate_datasets(datasets)

    logger.info("Tokenizing training set ...")
    train_dataset = _tokenize_split(train_df)

    logger.info("Tokenizing validation set ...")
    val_dataset = _tokenize_split(val_df)

    logger.info(
        f"Tokenization complete -> "
        f"train={len(train_dataset):,}, val={len(val_dataset):,}"
    )

    return train_dataset, val_dataset, test_df, stats
