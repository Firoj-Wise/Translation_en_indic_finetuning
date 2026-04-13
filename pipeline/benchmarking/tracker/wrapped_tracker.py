"""
Abstract tracker interface — swap backends via config.

All tracker calls in the pipeline go through this interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from constants.types import EvalResult

logger = logging.getLogger("benchmarking.tracker")


class BaseTracker(ABC):
    """Abstract tracker interface."""

    @abstractmethod
    def init_run(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
    ) -> None:
        """Initialise a new tracking run."""
        ...

    @abstractmethod
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log the full config as an artifact."""
        ...

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log scalar metrics."""
        ...

    @abstractmethod
    def log_table(
        self, name: str, columns: List[str], data: List[List[Any]]
    ) -> None:
        """Log a table (e.g., evaluation results)."""
        ...

    @abstractmethod
    def log_artifact(
        self, name: str, data: Dict[str, Any], artifact_type: str = "result"
    ) -> None:
        """Log a JSON artifact (e.g., validation report)."""
        ...

    @abstractmethod
    def finish(self) -> None:
        """Close the run."""
        ...


class NoOpTracker(BaseTracker):
    """Tracker that does nothing — used when ``backend: none``."""

    def init_run(self, project, run_name, config):
        logger.info("Tracker: NoOp (logging disabled)")

    def log_config(self, config):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_table(self, name, columns, data):
        pass

    def log_artifact(self, name, data, artifact_type="result"):
        pass

    def finish(self):
        pass
