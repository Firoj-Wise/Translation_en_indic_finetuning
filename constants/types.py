"""
Shared type definitions used across the pipeline.

Using dataclasses for lightweight, immutable data carriers
that serialise cleanly to dicts/JSON.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────

class IngestionSource(str, Enum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class MixedPrecision(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    NONE = "no"


class TrackerBackend(str, Enum):
    WANDB = "wandb"
    NONE = "none"


# ── Data Carriers ─────────────────────────────────────────────

@dataclass(frozen=True)
class Direction:
    """A single translation direction."""
    src_lang: str
    tgt_lang: str

    def __str__(self) -> str:
        return f"{self.src_lang} -> {self.tgt_lang}"


@dataclass
class EvalResult:
    """Evaluation metrics for one direction."""
    direction: str
    n_samples: int
    bleu: float
    chrf_pp: float
    domain: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ValidationStats:
    """Per-rule rejection counts from the validation stage."""
    total_input: int = 0
    total_output: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_bad_ratio: int = 0
    rejected_identical: int = 0
    rejected_empty_numeric: int = 0
    rejected_ascii_script: int = 0
    duplicates_removed: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunMetadata:
    """Versioning envelope for a single pipeline run."""
    run_id: str = ""
    git_commit: str = ""
    config_hash: str = ""
    dataset_hash: str = ""
    timestamp: str = ""
    training_duration_seconds: float = 0.0
    final_metrics: Dict[str, float] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SplitStats:
    """Sizes of each data split after preprocessing."""
    train: int = 0
    val: int = 0
    test: int = 0
    directions: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)
