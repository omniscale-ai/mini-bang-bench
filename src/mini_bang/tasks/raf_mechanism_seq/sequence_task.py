from __future__ import annotations

from mini_bang.tasks.base import RemoteSimulationTask


class RAFLevel5MechanismSequenceTask(RemoteSimulationTask):
    """Thin adapter for the RAF mechanism sequence task package."""

    task_id = "raf/mechanism-seq"
    config_package = __package__
