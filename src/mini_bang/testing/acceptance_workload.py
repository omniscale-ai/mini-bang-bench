"""Reference workload of the cluster acceptance checks, declared by a task manifest.

A task package opts in with `"acceptance": "package.module"` in its `config.json`.
The module provides:

- `SMOKE_AGENT`: `module:Class` of a deterministic agent that solves the default
  instance without a model;
- `ANSWER`: a constant answer that the default instance accepts and scores;
- `observation_request()`: a small `SimulationRequest` for worker/local parity;
- `normalize(payload)`: the payload made comparable across processes.

The task id and the path of its default gold come from the task manifest, so the
checks name no simulator or task themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class AcceptanceWorkload:
    task_id: str
    gold_path: str  # relative to the installed mini_bang package
    module: ModuleType

    @property
    def smoke_agent(self) -> str:
        return self.module.SMOKE_AGENT

    @property
    def answer(self) -> dict[str, Any]:
        return self.module.ANSWER

    def observation_request(self) -> Any:
        return self.module.observation_request()

    def normalize(self, payload: Any) -> Any:
        return self.module.normalize(payload)


def acceptance_workload() -> AcceptanceWorkload:
    """The workload of the first discovered task manifest that declares one."""
    from mini_bang.tasks.loader import TaskLoader

    for spec in TaskLoader().specs():
        module_name = spec.config.get("acceptance")
        if module_name:
            package = spec.package.split(".", 1)[1].replace(".", "/")
            return AcceptanceWorkload(task_id=spec.task_id, module=import_module(module_name),
                                      gold_path=f"{package}/gold/{spec.default_instance_id}.json")
    raise LookupError("No task manifest declares an acceptance workload")
