"""
Benchmark runner — compare current run against previous runs.

Loads past results from the versioning output directory,
generates comparison tables, and logs them to the tracker.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from constants.types import EvalResult

logger = logging.getLogger("benchmarking.runner")


def _load_previous_results(runs_dir: str) -> List[Dict[str, Any]]:
    """Load all run_metadata.json files from previous runs."""
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return []

    results = []
    for meta_file in sorted(runs_path.glob("*/run_metadata.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source_file"] = str(meta_file)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {meta_file}: {e}")

    return results


def compare_against_previous(
    current_results: List[EvalResult],
    current_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Compare current evaluation results against previous runs.

    Parameters
    ----------
    current_results : list[EvalResult]
        This run's evaluation results.
    current_metadata : dict
        This run's metadata (run_id, config_hash, etc.).
    config : dict
        Full pipeline config.

    Returns
    -------
    dict or None
        Comparison summary, or None if no previous runs exist.
    """
    runs_dir = config.get("versioning", {}).get("output_dir", "./runs")
    previous = _load_previous_results(runs_dir)

    if not previous:
        logger.info("No previous runs found — skipping comparison.")
        return None

    logger.info(f"Comparing against {len(previous)} previous run(s)")

    # Build comparison table
    comparison = {
        "current_run_id": current_metadata.get("run_id", "current"),
        "current_timestamp": current_metadata.get("timestamp", ""),
        "previous_runs": len(previous),
        "directions": [],
    }

    for result in current_results:
        direction_info = {
            "direction": result.direction,
            "current_bleu": result.bleu,
            "current_chrf": result.chrf_pp,
        }

        # Find best previous score for this direction
        best_prev_bleu = 0.0
        best_prev_chrf = 0.0
        for prev_run in previous:
            prev_metrics = prev_run.get("final_metrics", {})
            for key, val in prev_metrics.items():
                if result.direction in key:
                    if "bleu" in key.lower():
                        best_prev_bleu = max(best_prev_bleu, val)
                    elif "chrf" in key.lower():
                        best_prev_chrf = max(best_prev_chrf, val)

        direction_info["best_prev_bleu"] = best_prev_bleu
        direction_info["best_prev_chrf"] = best_prev_chrf
        direction_info["bleu_delta"] = round(
            result.bleu - best_prev_bleu, 2
        )
        direction_info["chrf_delta"] = round(
            result.chrf_pp - best_prev_chrf, 2
        )

        comparison["directions"].append(direction_info)

    # Log comparison
    logger.info("--- Benchmark Comparison ---")
    for d in comparison["directions"]:
        bleu_sign = "+" if d["bleu_delta"] >= 0 else ""
        chrf_sign = "+" if d["chrf_delta"] >= 0 else ""
        logger.info(
            f"  {d['direction']:>25s}  |  "
            f"BLEU: {d['current_bleu']:.2f} ({bleu_sign}{d['bleu_delta']:.2f})  "
            f"chrF++: {d['current_chrf']:.2f} ({chrf_sign}{d['chrf_delta']:.2f})"
        )
    logger.info("----------------------------")

    return comparison
