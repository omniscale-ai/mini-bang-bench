from __future__ import annotations

from mini_bang.tasks.base import RemoteSimulationTask


class RAFLevel2TimingTask(RemoteSimulationTask):
    """Thin adapter for the RAF timing task package."""

    task_id = "raf/timing"
    config_package = __package__
