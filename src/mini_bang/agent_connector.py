from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mini_bang.backends.base import SimulationBackend
from mini_bang.framework import TaskEnvironment, TaskSubmission, ValidationResult
from mini_bang.framework.records import RunRecord
from mini_bang.framework.runtime import AgentProtocol, RunManager
from mini_bang.tasks.loader import TaskLoader


@dataclass
class AgentRunResult:
    task_id: str
    environment: TaskEnvironment
    submission: TaskSubmission
    validation: ValidationResult
    run_record: RunRecord | None = None


class AgentConnector:
    """
    High-level entry point used by benchmark authors and agent developers.
    """

    def __init__(
        self,
        loader: TaskLoader | None = None,
        *,
        backend: SimulationBackend | None = None,
        run_manager: RunManager | None = None,
    ) -> None:
        self._tasks = loader or TaskLoader()
        self._run_manager = run_manager or RunManager(loader=self._tasks, backend=backend)

    def list_tasks(self) -> list[str]:
        return list(self._tasks.task_ids())

    def prepare(self, task_id: str, *, instance_id: str | None = None) -> TaskEnvironment:
        environment = self._run_manager.prepare(task_id, instance_id=instance_id).environment
        if environment is None:
            raise RuntimeError("RunManager did not create a task environment")
        return environment

    def run(
        self,
        agent: AgentProtocol | Callable[[TaskEnvironment], Any],
        task_id: str,
        *,
        instance_id: str | None = None,
    ) -> AgentRunResult:
        managed = self._run_manager.run(agent, task_id, instance_id=instance_id)
        return AgentRunResult(
            task_id=managed.session.task_id,
            environment=managed.environment,
            submission=managed.submission,
            validation=managed.validation,
            run_record=managed.record,
        )

    async def run_async(
        self,
        agent: AgentProtocol | Callable[[TaskEnvironment], Any],
        task_id: str,
        *,
        instance_id: str | None = None,
    ) -> AgentRunResult:
        managed = await self._run_manager.run_async(agent, task_id, instance_id=instance_id)
        return AgentRunResult(
            task_id=managed.session.task_id,
            environment=managed.environment,
            submission=managed.submission,
            validation=managed.validation,
            run_record=managed.record,
        )
