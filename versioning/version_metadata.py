"""
Run metadata capture and persistence.

Creates a versioning envelope for each pipeline run, containing:
  - Run ID (UUID)
  - Git commit
  - Config + dataset hashes
  - Timestamp
  - Final metrics
  - Environment info
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from constants.types import EvalResult, RunMetadata
from versioning.content_hashing import (
    get_git_commit,
    hash_config,
    hash_dataset,
)

logger = logging.getLogger("versioning.metadata")


def create_run_metadata(
    config: Dict[str, Any],
    df: pd.DataFrame,
    environment: Dict[str, str],
) -> RunMetadata:
    """
    Build a :class:`RunMetadata` for the current run.

    Parameters
    ----------
    config : dict
        Full pipeline config.
    df : pd.DataFrame
        The full dataset (used for hashing).
    environment : dict
        Environment info from :func:`get_environment_info`.

    Returns
    -------
    RunMetadata
    """
    ver_cfg = config.get("versioning", {})
    seed = config["pipeline"].get("seed", 42)

    meta = RunMetadata(
        run_id=str(uuid.uuid4()),
        git_commit=get_git_commit() or "unknown",
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=environment,
    )

    if ver_cfg.get("hash_config", True):
        meta.config_hash = hash_config(config)

    if ver_cfg.get("hash_dataset", True):
        meta.dataset_hash = hash_dataset(df, seed=seed)

    logger.info(
        f"Run metadata created -> id={meta.run_id[:8]}... "
        f"config={meta.config_hash[:8]}... "
        f"dataset={meta.dataset_hash[:8]}..."
    )
    return meta


def update_with_results(
    meta: RunMetadata,
    results: List[EvalResult],
    training_duration: float = 0.0,
) -> RunMetadata:
    """Update metadata with final metrics and training duration."""
    meta.training_duration_seconds = training_duration

    for r in results:
        safe_dir = r.direction.replace(" ", "_").replace("->", "to")
        meta.final_metrics[f"{safe_dir}/bleu"] = r.bleu
        meta.final_metrics[f"{safe_dir}/chrf++"] = r.chrf_pp

    return meta


def save_run_metadata(
    meta: RunMetadata,
    config: Dict[str, Any],
) -> Path:
    """
    Persist run metadata to the versioning output directory.

    Creates ``<runs_dir>/<run_id>/run_metadata.json``.
    """
    runs_dir = config.get("versioning", {}).get("output_dir", "./runs")
    run_dir = Path(runs_dir) / meta.run_id[:8]
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Run metadata saved -> {meta_path}")
    return meta_path
