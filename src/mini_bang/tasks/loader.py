from __future__ import annotations

import json
import os
import pkgutil
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, Iterable

from mini_bang.framework.task import SimulationTask, TaskSpec


class TaskLoader:
    """Manifest-backed registry for benchmark tasks.

    Task packages are discovered under ``mini_bang.tasks`` by looking for a
    ``config.json`` file that declares at least ``task_id`` and ``task_class``.
    The legacy ``register(task)`` method remains available for tests or external
    code that wants to inject task instances directly.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, SimulationTask] = {}
        self._specs: Dict[str, TaskSpec] = {}
        self._auto_register()

    def _auto_register(self) -> None:
        for spec in discover_task_specs():
            self.register_spec(spec)

    def register(self, task: SimulationTask) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"Task '{task.task_id}' already registered")
        if task.task_id in self._specs:
            raise ValueError(f"Task '{task.task_id}' already registered")
        self._tasks[task.task_id] = task
        if task.spec is not None:
            self._specs[task.task_id] = task.spec

    def register_spec(self, spec: TaskSpec) -> None:
        """Register a discovered task manifest without instantiating the task."""
        if spec.task_id in self._specs or spec.task_id in self._tasks:
            raise ValueError(f"Task '{spec.task_id}' already registered")
        self._specs[spec.task_id] = spec

    def get(self, task_id: str) -> SimulationTask:
        if task_id in self._tasks:
            return self._tasks[task_id]
        if task_id in self._specs:
            task = self._specs[task_id].instantiate_task()
            self._tasks[task_id] = task
            return task
        raise KeyError(f"Unknown task '{task_id}'")

    def all(self) -> Iterable[SimulationTask]:
        for task_id in self.task_ids():
            yield self.get(task_id)

    def task_ids(self) -> tuple[str, ...]:
        """Return all known task ids without forcing task instantiation."""
        return tuple(sorted(self._specs.keys() | self._tasks.keys()))

    def get_spec(self, task_id: str) -> TaskSpec:
        """Return the manifest spec for a discovered task."""
        if task_id in self._specs:
            return self._specs[task_id]
        if task_id in self._tasks and self._tasks[task_id].spec is not None:
            return self._tasks[task_id].spec
        raise KeyError(f"Task '{task_id}' has no discovered spec")

    def specs(self) -> Iterable[TaskSpec]:
        """Return all discovered task specs."""
        return tuple(self._specs.values())


def discover_task_packages(root: str = "mini_bang.tasks") -> Iterable[str]:
    """Yield task package names under ``root`` that contain a manifest."""
    root_pkg = import_module(root)
    for module_info in pkgutil.iter_modules(root_pkg.__path__, root_pkg.__name__ + "."):
        if not module_info.ispkg:
            continue
        package = module_info.name
        if _task_config_path(module_info) is not None:
            yield package


def discover_task_specs(root: str = "mini_bang.tasks") -> Iterable[TaskSpec]:
    """Load every task manifest under ``root`` into a ``TaskSpec``."""
    root_pkg = import_module(root)
    for module_info in pkgutil.iter_modules(root_pkg.__path__, root_pkg.__name__ + "."):
        if not module_info.ispkg:
            continue
        config_path = _task_config_path(module_info)
        if config_path is None:
            continue
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        package_dir = config_path.parent
        yield TaskSpec.from_config(module_info.name, config, package_dir=package_dir)
    for package in _env_packages("MINI_BANG_TASK_PACKAGES"):
        yield from _specs_from_package(package)
    try:
        selected = entry_points(group="mini_bang.tasks")
    except TypeError:  # pragma: no cover
        selected = entry_points().get("mini_bang.tasks", [])
    for entry_point in selected:
        yield from _coerce_task_plugin(entry_point.load(), entry_point.name)


def _task_config_path(module_info: pkgutil.ModuleInfo) -> Path | None:
    finder_path = getattr(module_info.module_finder, "path", None)
    if finder_path is None:
        return None
    package_dir = Path(finder_path) / module_info.name.rpartition(".")[2]
    config_path = package_dir / "config.json"
    return config_path if config_path.is_file() else None


def _coerce_task_plugin(value: Any, name: str) -> Iterable[TaskSpec]:
    if isinstance(value, TaskSpec):
        yield value
        return
    if callable(value) and not hasattr(value, "__path__"):
        result = value()
        if isinstance(result, TaskSpec):
            yield result
        else:
            yield from result
        return
    package = getattr(value, "__name__", None)
    if not package:
        raise ValueError(f"task plugin '{name}' did not return a package or TaskSpec")
    yield from _specs_from_package(package)


def _specs_from_package(package: str) -> Iterable[TaskSpec]:
    module = import_module(package)
    package_dir = Path(getattr(module, "__file__", "")).resolve().parent
    config_path = package_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"task package '{package}' has no config.json")
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    yield TaskSpec.from_config(package, config, package_dir=package_dir)


def _env_packages(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]
