from __future__ import annotations

import copy
from typing import Any

from mini_bang.framework.task import SimulationTask
from mini_bang.tasks.config_loader import load_task_config


class ConfiguredSimulationTask(SimulationTask):
    """Thin task-package adapter backed by ``config.json``.

    The benchmark runtime no longer calls ``build`` or ``validate`` on task
    objects. This adapter only exposes package metadata for external tools that
    still instantiate the declared task class.
    """

    config_package: str
    config_filename: str = "config.json"

    def __init__(self) -> None:
        if not getattr(self, "config_package", None):
            raise ValueError("ConfiguredSimulationTask subclasses must set config_package")
        self._config: dict[str, Any] = load_task_config(
            self.config_package,
            self.config_filename,
        )

    def config_copy(self) -> dict[str, Any]:
        """Return a deep copy of the task configuration."""
        return copy.deepcopy(self._config)

    def config_value(self, key: str, default: Any = None) -> Any:
        """Return a deep copy of a config value."""
        return copy.deepcopy(self._config.get(key, default))


class RemoteSimulationTask(ConfiguredSimulationTask):
    """Deprecated name retained for task adapter class paths."""
