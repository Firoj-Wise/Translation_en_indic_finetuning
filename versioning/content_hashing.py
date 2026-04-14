"""
Content hashing for dataset and config fingerprinting.

Creates a deterministic SHA-256 fingerprint from:
  - Config YAML content
  - Dataset content (sampled rows)
  - Git commit hash (if available)

Two runs with identical data + config produce the same fingerprint.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

import pandas as pd
try:
    import git
except ImportError:
    git = None

logger = logging.getLogger("versioning.hashing")


def hash_string(content: str) -> str:
    """SHA-256 hash of a string, returned as hex."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_config(config: Dict[str, Any]) -> str:
    """
    Deterministic hash of the pipeline config.

    Serialises to sorted JSON to ensure key-order independence.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    h = hash_string(canonical)
    logger.debug(f"Config hash: {h[:12]}...")
    return h


def hash_dataset(
    df: pd.DataFrame,
    sample_n: int = 5000,
    seed: int = 42,
) -> str:
    """
    Deterministic hash of dataset content.

    For large datasets, samples ``sample_n`` rows for speed.
    The hash covers all columns including src, tgt, and lang codes.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    sample_n : int
        Max rows to hash.
    seed : int
        Seed for deterministic sampling.

    Returns
    -------
    str
        SHA-256 hex string.
    """
    if len(df) > sample_n:
        sample = df.sample(n=sample_n, random_state=seed)
    else:
        sample = df

    # Sort for determinism, then serialise
    sample = sample.sort_values(by=list(sample.columns)).reset_index(drop=True)
    canonical = sample.to_csv(index=False)
    h = hash_string(canonical)
    logger.debug(f"Dataset hash ({len(sample)} rows sampled): {h[:12]}...")
    return h


def get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or None."""
    try:
        if git is None:
            return None
        repo = git.Repo(search_parent_directories=True)
        commit = repo.head.commit.hexsha
        logger.debug(f"Git commit: {commit[:12]}…")
        return commit
    except Exception:
        logger.debug("Git commit not available")
        return None
