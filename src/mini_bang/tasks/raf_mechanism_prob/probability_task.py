from __future__ import annotations

from mini_bang.tasks.base import RemoteSimulationTask


class RAFLevel5MechanismProbabilityTask(RemoteSimulationTask):
    """Thin adapter for the RAF mechanism probability task package."""

    task_id = "raf/mechanism-prob"
    config_package = __package__
