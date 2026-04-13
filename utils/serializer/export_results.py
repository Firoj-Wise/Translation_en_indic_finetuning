"""
Export evaluation results to JSON and CSV for offline analysis.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List

from constants.types import EvalResult


def export_to_json(
    results: List[EvalResult],
    output_path: str,
    run_metadata: Dict | None = None,
) -> Path:
    """
    Save evaluation results as a JSON file.

    Parameters
    ----------
    results : list[EvalResult]
        One entry per direction evaluated.
    output_path : str
        Where to write the JSON.
    run_metadata : dict, optional
        Extra metadata to embed (run_id, git commit, etc.).

    Returns
    -------
    Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": run_metadata or {},
        "results": [r.to_dict() for r in results],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def export_to_csv(
    results: List[EvalResult],
    output_path: str,
) -> Path:
    """
    Save evaluation results as a flat CSV table.

    Columns: direction, n_samples, bleu, chrf_pp, domain
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["direction", "n_samples", "bleu", "chrf_pp", "domain"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    return path
