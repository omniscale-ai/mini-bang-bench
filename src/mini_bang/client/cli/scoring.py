from __future__ import annotations

from typing import Any

from mini_bang.client.cli.errors import CliError, EXIT_INTERNAL
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.services import ExecutionServices
from mini_bang.framework.scoring import (
    ScoringServices,
    scoring_context_from_config,
    scoring_requirements,
)


def build_official_scoring_services(
    *,
    backend: Any,
    session: Any,
    trace: list[dict[str, Any]],
    resource_usage: dict[str, Any],
    context: RuntimeContext | None = None,
) -> ScoringServices | None:
    """Scoring services for runner-local scoring, or None when the task declares none.

    A task whose `scoring` block requires an executable runner cannot be scored
    here: submitted code needs the isolated runner of a scoring job backend.
    """
    requirements = scoring_requirements(session.task_spec)
    if requirements.requires_executable_runner:
        raise CliError(
            f"task {session.task_id} requires an executable scoring runner, which runner-local "
            "scoring does not have; score it through a Kubernetes scoring backend",
            EXIT_INTERNAL,
        )
    if not requirements.requires_services:
        return None
    getter = getattr(backend, "execution_services", None)
    execution_services = getter() if callable(getter) else None
    if execution_services is None:
        execution_services = ExecutionServices()
    if context is None:
        base = scoring_context_from_config(session.task_spec) or RuntimeContext.from_env(
            run_id=session.run_id,
            task_id=session.task_id,
            instance_id=session.instance_id,
        )
        context = RuntimeContext(
            run_id=session.run_id,
            task_id=session.task_id,
            instance_id=session.instance_id,
            artifact=base.artifact,
            payload=base.payload,
            security={**base.security, "authority": "scoring", "official_mode": True},
            env=base.env,
        )
    return ScoringServices(
        execution_services,
        context,
        trace=trace,
        resource_usage=resource_usage,
    )
