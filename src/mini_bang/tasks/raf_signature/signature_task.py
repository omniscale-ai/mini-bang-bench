from __future__ import annotations

from mini_bang.tasks.base import RemoteSimulationTask


class RAFLevel4SignatureTask(RemoteSimulationTask):
    """Thin adapter for the RAF signature task package."""

    task_id = "raf/signature"
    config_package = __package__
