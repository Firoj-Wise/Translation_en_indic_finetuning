"""
Structured report from the validation stage.

Tracks per-rule rejection counts and produces loggable summaries.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from constants.types import ValidationStats

logger = logging.getLogger("validation.report")


class ValidationReport:
    """
    Accumulates validation statistics and serialises them.

    Usage::

        report = ValidationReport(total_input=len(df))
        report.record_rejection("min_length", count=42)
        ...
        report.finalise(total_output=len(clean_df))
        report.log()
        report.save("./runs/validation_report.json")
    """

    def __init__(self, total_input: int) -> None:
        self.stats = ValidationStats(total_input=total_input)
        self._rule_counts: Dict[str, int] = {}

    def record_rejection(self, rule_name: str, count: int) -> None:
        """Record how many rows were rejected by a specific rule."""
        self._rule_counts[rule_name] = count

        # Map rule name -> stats field
        field_map = {
            "min_length": "rejected_too_short",
            "max_length": "rejected_too_long",
            "length_ratio": "rejected_bad_ratio",
            "not_identical": "rejected_identical",
            "not_empty_or_numeric": "rejected_empty_numeric",
            "devanagari_ascii_ratio": "rejected_ascii_script",
        }

        field = field_map.get(rule_name)
        if field and hasattr(self.stats, field):
            setattr(self.stats, field, count)

    def record_dedup(self, count: int) -> None:
        """Record how many duplicates were removed."""
        self.stats.duplicates_removed = count

    def finalise(self, total_output: int) -> None:
        """Set the final output count."""
        self.stats.total_output = total_output

    def log(self) -> None:
        """Write a human-readable summary to the logger."""
        s = self.stats
        logger.info("--- Validation Report ---")
        logger.info(f"  Input rows:        {s.total_input:>8,}")
        for rule_name, count in self._rule_counts.items():
            logger.info(f"  Rejected ({rule_name}): {count:>8,}")
        logger.info(f"  Duplicates removed:{s.duplicates_removed:>8,}")
        logger.info(f"  Output rows:       {s.total_output:>8,}")
        logger.info(f"  Retention rate:    {s.total_output / max(s.total_input, 1):.1%}")
        logger.info("-------------------------")


    def to_dict(self) -> Dict[str, Any]:
        """Return full stats as a dict for W&B / JSON."""
        d = self.stats.to_dict()
        d["per_rule_rejections"] = dict(self._rule_counts)
        return d

    def save(self, path: str) -> Path:
        """Persist the report as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Validation report saved -> {out}")
        return out
