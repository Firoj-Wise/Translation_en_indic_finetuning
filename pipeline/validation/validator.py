"""
Validation orchestrator — applies rules, deduplicates, and
produces a :class:`ValidationReport`.
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from pipeline.validation.build_rules import build_rules
from pipeline.validation.rules import Rule
from pipeline.validation.validation_report import ValidationReport

logger = logging.getLogger("validation")


def validate(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Run the full validation pipeline on raw data.

    1. Apply each rule and track per-rule rejections.
    2. Deduplicate on the configured column.
    3. Produce a :class:`ValidationReport`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with columns: src, tgt, src_lang, tgt_lang, domain.
    config : dict
        Full pipeline config.

    Returns
    -------
    (clean_df, report)
    """
    report = ValidationReport(total_input=len(df))
    rules = build_rules(config)

    # ── Apply rules sequentially ──────────────────────────────
    for rule in rules:
        before = len(df)
        mask = df.apply(rule, axis=1)
        df = df[mask].copy()
        rejected = before - len(df)
        report.record_rejection(rule.__name__, rejected)
        if rejected > 0:
            logger.info(
                f"Rule '{rule.__name__}' rejected {rejected:,} rows "
                f"({before:,} -> {len(df):,})"
            )

    # ── Deduplication ─────────────────────────────────────────
    dedup_col = config["validation"].get("deduplicate_on", "src")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[dedup_col])
    dups = before_dedup - len(df)
    report.record_dedup(dups)
    if dups > 0:
        logger.info(
            f"Deduplication on '{dedup_col}' removed {dups:,} rows "
            f"({before_dedup:,} -> {len(df):,})"
        )

    df = df.reset_index(drop=True)
    report.finalise(total_output=len(df))
    report.log()

    return df, report
