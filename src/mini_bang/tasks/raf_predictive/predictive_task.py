from __future__ import annotations

from mini_bang.tasks.base import RemoteSimulationTask


class RAFLevel6PredictiveTask(RemoteSimulationTask):
    """Thin adapter for the RAF predictive task package."""

    task_id = "raf/predictive"
    config_package = __package__
