from __future__ import annotations

import asyncio
import copy
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from mini_bang.naming import readable_id

from mini_bang.artifacts import ArtifactResolver, create_default_artifact_resolver
from mini_bang.backends import create_backend_from_env
from mini_bang.backends.base import SimulationBackend, check_backend_runtime
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.framework.answer_format import answer_problems
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.framework.distributed_scoring import (
    mark_distributed_score_timing,
    score_with_scoring_job_backend,
)
from mini_bang.framework.executable import ExecutableSubmissionRunner
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring import (
    Scorer,
    ScoringServices,
    score_submission,
    scoring_context_from_config,
    scoring_requirements,
)
from mini_bang.framework.task import (
    RunSession,
    TaskEnvironment,
    TaskSubmission,
    ValidationResult,
    with_task_level,
)
from mini_bang.tasks.loader import TaskLoader


class AgentProtocol(Protocol):
    """Protocol for objects that solve a task environment."""

    def solve(self, task: TaskEnvironment) -> TaskSubmission | Any:
        ...


AgentRunner = AgentProtocol


def unscored_validation(session: RunSession, submission: TaskSubmission,
                        artifact_store: BenchmarkArtifactStore) -> ValidationResult | None:
    """Result of an answer that is not scored here: malformed, or a test instance without installed gold."""
    problems = answer_problems(session.task_spec, session.instance_spec, submission.answer)
    if problems:
        session.trace.append({"event": "submission_malformed", "timestamp": time.time()})
        return ValidationResult(False, "Malformed answer: " + "; ".join(problems), {"valid_submission": False})
    if not artifact_store.has_gold(session.instance_spec):
        session.trace.append({"event": "stored_for_scoring_server", "timestamp": time.time()})
        return ValidationResult(False, "Test instance: the answer is kept for the scoring server "
                                "(`mini-bang submit`)", {"valid_submission": True, "scored_locally": False})
    return None


@dataclass
class ManagedRunResult:
    """Harness result paired with its durable run record."""

    session: RunSession
    environment: TaskEnvironment
    submission: TaskSubmission
    validation: ValidationResult
    record: RunRecord


class RunManager:
    """Artifact-driven lifecycle manager for benchmark runs."""

    def __init__(
        self,
        *,
        loader: TaskLoader | None = None,
        backend: SimulationBackend | None = None,
        artifact_store: BenchmarkArtifactStore | None = None,
        artifact_resolver: ArtifactResolver | None = None,
        runtime_context: RuntimeContext | None = None,
        scoring_runtime_context: RuntimeContext | None = None,
        execution_services: ExecutionServices | None = None,
        executable_runner: ExecutableSubmissionRunner | None = None,
        scoring_job_backend: Any | None = None,
    ) -> None:
        self._loader = loader or TaskLoader()
        self._artifact_resolver = artifact_resolver or create_default_artifact_resolver()
        self._runtime_context = runtime_context
        self._scoring_runtime_context = scoring_runtime_context
        self._execution_services = execution_services
        self._executable_runner = executable_runner
        self._scoring_job_backend = scoring_job_backend
        self._backend = backend or create_backend_from_env()
        self._artifact_store = artifact_store or BenchmarkArtifactStore(
            artifact_resolver=self._artifact_resolver
        )
        if getattr(self._artifact_store, "artifact_resolver", None) is None:
            self._artifact_store.artifact_resolver = self._artifact_resolver

    def prepare(self, task_id: str, *, instance_id: str | None = None) -> RunSession:
        """Create one concrete runtime session and its task-specific API."""
        task_spec = self._loader.get_spec(task_id)
        instance_spec = self._artifact_store.load_instance(task_spec, instance_id)
        policy = task_spec.create_policy()
        run_id = readable_id("run", task_spec.task_id, instance_spec.instance_id)
        runtime_context = self._session_runtime_context(
            run_id=run_id,
            task_id=task_spec.task_id,
            instance_id=instance_spec.instance_id,
        )
        runtime_services = self._session_runtime_services(runtime_context)
        # A backend that cannot honor this runtime fails here, before any budget.
        check_backend_runtime(self._backend, context=runtime_context, services=runtime_services)
        session = RunSession(
            task_spec=task_spec,
            instance_spec=instance_spec,
            backend=self._backend,
            policy=policy,
            run_id=run_id,
            backend_metadata=self._backend.describe(),
            artifact_resolver=self._artifact_resolver,
            runtime_context=runtime_context,
            runtime_services=runtime_services,
        )
        policy.initialize_session(session)
        api = task_spec.create_api(session=session)
        metadata = with_task_level(instance_spec.to_public_metadata(), task_spec.metadata)
        metadata["task_version"] = task_spec.version
        metadata["backend"] = self._backend.describe()
        metadata["capabilities"] = session.capabilities()
        description = instance_spec.description or task_spec.description
        environment = TaskEnvironment(
            description=description,
            api=api,
            metadata=metadata,
        )
        session.environment = environment
        session.trace.append(
            {
                "event": "session_prepared",
                "timestamp": time.time(),
                "task_id": task_spec.task_id,
                "instance_id": instance_spec.instance_id,
                "task_version": task_spec.version,
                "instance_version": instance_spec.version,
                **with_task_level({}, task_spec.metadata),
                "backend_metadata": self._backend.describe(),
            }
        )
        return session

    def run(
        self,
        agent: AgentProtocol | Callable[[TaskEnvironment], Any],
        task_id: str,
        *,
        instance_id: str | None = None,
    ) -> ManagedRunResult:
        """Run an agent and return a scored durable run record."""
        timing: dict[str, Any] = {}
        run_started = time.perf_counter()

        build_started = time.perf_counter()
        session = self.prepare(task_id, instance_id=instance_id)
        if session.environment is None:
            raise RuntimeError("RunSession did not create a task environment")
        environment = session.environment
        timing["prepare_seconds"] = time.perf_counter() - build_started

        agent_started = time.perf_counter()
        submission = self._invoke_agent(agent, environment)
        if not isinstance(submission, TaskSubmission):
            submission = TaskSubmission(answer=submission)
        timing["agent_seconds"] = time.perf_counter() - agent_started
        session.trace.append(
            {
                "event": "submission_received",
                "timestamp": time.time(),
            }
        )

        timing["wall_seconds"] = time.perf_counter() - run_started
        session.timing.update(timing)
        provisional_record = self._build_record(session, submission, validation=None)
        unscored = unscored_validation(session, submission, self._artifact_store)
        score_started = time.perf_counter()
        distributed = None if unscored else score_with_scoring_job_backend(
            scoring_job_backend=self._scoring_job_backend,
            artifact_store=self._artifact_store,
            session=session,
            submission=submission,
            provisional_record=provisional_record,
            scoring_context=self._scoring_context(session),
        )
        if distributed is not None:
            validation, record = distributed
            mark_distributed_score_timing(
                session,
                score_started=score_started,
                run_started=run_started,
            )
            return ManagedRunResult(
                session=session,
                environment=environment,
                submission=submission,
                validation=validation,
                record=record,
            )

        score_started = time.perf_counter()
        validation, scoring_trace, scoring_resource_usage = (
            (unscored, [], {}) if unscored else self._score_locally(session, submission, provisional_record))
        session.timing["score_seconds"] = time.perf_counter() - score_started
        session.timing["wall_seconds"] = time.perf_counter() - run_started
        record = self._build_record(
            session,
            submission,
            validation=validation,
            scoring_trace=scoring_trace,
            scoring_resource_usage=scoring_resource_usage,
        )
        return ManagedRunResult(
            session=session,
            environment=environment,
            submission=submission,
            validation=validation,
            record=record,
        )

    def _score_locally(
        self,
        session: RunSession,
        submission: TaskSubmission,
        provisional_record: RunRecord,
    ) -> tuple[ValidationResult, list[dict[str, Any]], dict[str, Any]]:
        """Score with the installed gold; returns (validation, scoring trace, scoring resource usage)."""
        scorer: Scorer = session.task_spec.create_scorer()
        gold_store = self._artifact_store.load_gold(session.instance_spec)
        scoring_trace: list[dict[str, Any]] = []
        scoring_resource_usage: dict[str, Any] = {}
        scoring_services = self._build_scoring_services(
            session,
            trace=scoring_trace,
            resource_usage=scoring_resource_usage,
        )
        validation = score_submission(
            scorer,
            task_spec=session.task_spec,
            instance_spec=session.instance_spec,
            submission=submission,
            run_record=provisional_record,
            gold_store=gold_store,
            scoring_services=scoring_services,
        )
        session.trace.append({"event": "scored", "timestamp": time.time(), "gold_metadata": gold_store.to_metadata()})
        return validation, scoring_trace, scoring_resource_usage

    async def run_async(
        self,
        agent: AgentProtocol | Callable[[TaskEnvironment], Any],
        task_id: str,
        *,
        instance_id: str | None = None,
    ) -> ManagedRunResult:
        """Async run path for agents that fan out simulation calls."""
        timing: dict[str, Any] = {}
        run_started = time.perf_counter()

        build_started = time.perf_counter()
        session = self.prepare(task_id, instance_id=instance_id)
        if session.environment is None:
            raise RuntimeError("RunSession did not create a task environment")
        environment = session.environment
        timing["prepare_seconds"] = time.perf_counter() - build_started

        agent_started = time.perf_counter()
        submission = await self._invoke_agent_async(agent, environment)
        if not isinstance(submission, TaskSubmission):
            submission = TaskSubmission(answer=submission)
        timing["agent_seconds"] = time.perf_counter() - agent_started
        session.trace.append(
            {
                "event": "submission_received",
                "timestamp": time.time(),
            }
        )

        timing["wall_seconds"] = time.perf_counter() - run_started
        session.timing.update(timing)
        provisional_record = self._build_record(session, submission, validation=None)
        unscored = unscored_validation(session, submission, self._artifact_store)
        score_started = time.perf_counter()
        distributed = None if unscored else await asyncio.to_thread(
            score_with_scoring_job_backend,
            scoring_job_backend=self._scoring_job_backend,
            artifact_store=self._artifact_store,
            session=session,
            submission=submission,
            provisional_record=provisional_record,
            scoring_context=self._scoring_context(session),
        )
        if distributed is not None:
            validation, record = distributed
            mark_distributed_score_timing(
                session,
                score_started=score_started,
                run_started=run_started,
            )
            return ManagedRunResult(
                session=session,
                environment=environment,
                submission=submission,
                validation=validation,
                record=record,
            )

        score_started = time.perf_counter()
        validation, scoring_trace, scoring_resource_usage = (
            (unscored, [], {}) if unscored
            else await asyncio.to_thread(self._score_locally, session, submission, provisional_record))
        session.timing["score_seconds"] = time.perf_counter() - score_started
        session.timing["wall_seconds"] = time.perf_counter() - run_started
        record = self._build_record(
            session,
            submission,
            validation=validation,
            scoring_trace=scoring_trace,
            scoring_resource_usage=scoring_resource_usage,
        )
        return ManagedRunResult(
            session=session,
            environment=environment,
            submission=submission,
            validation=validation,
            record=record,
        )

    def _build_record(
        self,
        session: RunSession,
        submission: TaskSubmission,
        *,
        validation: ValidationResult | None,
        scoring_trace: list[dict[str, Any]] | None = None,
        scoring_resource_usage: dict[str, Any] | None = None,
    ) -> RunRecord:
        versions = {
            "task_version": session.task_spec.version,
            "instance_version": session.instance_spec.version,
        }
        return RunRecord.from_parts(
            run_id=session.run_id,
            task_id=session.task_id,
            instance_id=session.instance_id,
            submission=submission,
            validation=validation,
            scoring_trace=scoring_trace,
            scoring_resource_usage=scoring_resource_usage,
            trace=session.trace,
            timing=session.timing,
            resource_usage=session.resource_usage,
            backend_metadata=session.backend_metadata or self._backend.describe(),
            versions=versions,
        )

    def _invoke_agent(
        self,
        agent: AgentProtocol | Callable[[TaskEnvironment], Any],
        environment: TaskEnvironment,
    ) -> TaskSubmission | Any:
        if hasattr(agent, "solve") and callable(getattr(agent, "solve")):
            return agent.solve(environment)
        if callable(agent):
            return agent(environment)
        raise TypeError("Agent must be callable or provide a 'solve' method")

    def _session_runtime_context(
        self,
        *,
        run_id: str,
        task_id: str,
        instance_id: str,
    ) -> RuntimeContext:
        base_context = self._runtime_context or RuntimeContext.from_env()
        return base_context.with_run(
            run_id=run_id,
            task_id=task_id,
            instance_id=instance_id,
        )

    def _session_runtime_services(
        self,
        context: RuntimeContext,
    ) -> RuntimeServices | None:
        describe = getattr(self._backend, "describe", lambda: {})()
        if describe.get("backend") != "local" or describe.get("engine") == "process_pool":
            return None
        if self._runtime_context is not None:
            return RuntimeServices.from_context(context)
        services = RuntimeServices.from_context(context)
        services.artifact_resolver = self._artifact_resolver
        return services

    def _build_scoring_services(
        self,
        session: RunSession,
        *,
        trace: list[dict[str, Any]],
        resource_usage: dict[str, Any],
    ) -> ScoringServices | None:
        requirements = scoring_requirements(session.task_spec)
        if not requirements.requires_runtime:
            return None
        if requirements.requires_executable_runner and self._executable_runner is None:
            raise RuntimeError(
                f"Task '{session.task_id}' requires an ExecutableSubmissionRunner for scoring"
            )
        context = self._scoring_context(session)
        execution_services = self._execution_services
        if execution_services is None:
            from mini_bang.execution.services import ExecutionServices

            getter = getattr(self._backend, "execution_services", None)
            execution_services = getter() if callable(getter) else None
            if execution_services is None:
                execution_services = ExecutionServices()
        return ScoringServices(
            execution_services,
            context,
            executable_runner=self._executable_runner,
            trace=trace,
            resource_usage=resource_usage,
        )

    def _scoring_context(self, session: RunSession) -> RuntimeContext:
        base = (
            self._scoring_runtime_context
            or scoring_context_from_config(session.task_spec)
            or RuntimeContext.from_env()
        )
        security = {**base.security, "authority": "scoring"}
        return RuntimeContext(
            run_id=session.run_id,
            task_id=session.task_id,
            instance_id=session.instance_id,
            artifact=copy.deepcopy(base.artifact),
            payload=copy.deepcopy(base.payload),
            security=security,
            env=copy.deepcopy(base.env),
        )

    async def _invoke_agent_async(
        self,
        agent: AgentProtocol | Callable[[TaskEnvironment], Any],
        environment: TaskEnvironment,
    ) -> TaskSubmission | Any:
        solve_async = getattr(agent, "solve_async", None)
        if callable(solve_async):
            return await solve_async(environment)
        if hasattr(agent, "solve") and callable(getattr(agent, "solve")):
            solve = getattr(agent, "solve")
            if inspect.iscoroutinefunction(solve):
                return await solve(environment)
            result = await asyncio.to_thread(agent.solve, environment)
            if inspect.isawaitable(result):
                return await result
            return result
        if callable(agent):
            if inspect.iscoroutinefunction(agent):
                return await agent(environment)
            result = await asyncio.to_thread(agent, environment)
            if inspect.isawaitable(result):
                return await result
            return result
        raise TypeError("Agent must be callable or provide a 'solve' method")
