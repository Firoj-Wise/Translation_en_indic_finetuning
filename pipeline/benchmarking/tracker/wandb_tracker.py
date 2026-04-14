"""
Weights & Biases tracker implementation.

Wraps the W&B API behind the :class:`BaseTracker` interface so the
rest of the pipeline never calls ``wandb.*`` directly.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import wandb
from pipeline.benchmarking.tracker.wrapped_tracker import BaseTracker

logger = logging.getLogger("benchmarking.wandb")


class WandBTracker(BaseTracker):
    """Concrete tracker backed by Weights & Biases."""

    def __init__(self) -> None:
        self._run = None

    def init_run(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
    ) -> None:
        import wandb

        self._run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            reinit=True,
        )
        logger.info(
            f"W&B run initialised: {project}/{run_name} "
            f"(url: {self._run.url})"
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        if self._run is None:
            return
        import wandb

        self._run.config.update(config, allow_val_change=True)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        if self._run is None:
            return
        import wandb

        wandb.log(metrics, step=step)

    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
    ) -> None:
        if self._run is None:
            return
        import wandb

        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})

    def log_artifact(
        self,
        name: str,
        data: Dict[str, Any],
        artifact_type: str = "result",
    ) -> None:
        if self._run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)

        # Save dict as JSON file
        tmp = Path(tempfile.mkdtemp()) / f"{name}.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        artifact.add_file(str(tmp))
        self._run.log_artifact(artifact)
        logger.info(f"W&B artifact logged: {name} ({artifact_type})")

    def log_eval_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Log evaluation results as a W&B table with bar charts.

        Parameters
        ----------
        results : list[dict]
            Each dict has: direction, n_samples, bleu, chrf_pp
        """
        if self._run is None:
            return
        import wandb

        # Table
        columns = ["direction", "n_samples", "bleu", "chrf++"]
        data = [
            [r["direction"], r["n_samples"], r["bleu"], r["chrf_pp"]]
            for r in results
        ]
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"evaluation/results_table": table})

        # Bar charts
        bleu_chart = wandb.plot.bar(
            table, "direction", "bleu",
            title="BLEU by Direction",
        )
        chrf_chart = wandb.plot.bar(
            table, "direction", "chrf++",
            title="chrF++ by Direction",
        )
        wandb.log({
            "evaluation/bleu_chart": bleu_chart,
            "evaluation/chrf_chart": chrf_chart,
        })

    def finish(self) -> None:
        if self._run is not None:
            wandb.finish()
            logger.info("W&B run finished.")
