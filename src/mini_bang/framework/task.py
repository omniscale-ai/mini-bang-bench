from __future__ import annotations
import asyncio
import copy
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mini_bang.artifacts import ArtifactResolver, create_default_artifact_resolver
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.framework.backend_call import arun_batch_with_optional_context, run_batch_with_optional_context
from mini_bang.importing import import_from_path
from mini_bang.framework import request_views
from mini_bang.agent_types import AgentAPI, TaskEnvironment, TaskSubmission
if TYPE_CHECKING:  # pragma: no cover
    from mini_bang.framework.policy import TaskPolicy
    from mini_bang.framework.scoring import Scorer

# The task config owns the level; an instance file's own copy is never shown.
TASK_LEVEL_KEYS = ("level", "level_scheme")


def with_task_level(metadata: dict[str, Any], task_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return ``metadata`` with level keys taken only from the task config metadata."""
    result = {key: value for key, value in metadata.items() if key not in TASK_LEVEL_KEYS}
    result.update({key: copy.deepcopy(task_metadata[key]) for key in TASK_LEVEL_KEYS if key in task_metadata})
    return result


@dataclass
class ValidationResult:
    """Scoring result for one submission against one instance."""

    success: bool
    details: str | None = None
    metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class InstanceSpec:
    """Immutable description of one concrete evaluation instance."""

    task_id: str
    instance_id: str
    package: str
    description: str = ""
    instructions: str = ""
    visible_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    gold_path: Path | None = None
    config_path: Path | None = None

    @classmethod
    def from_config(
        cls,
        *,
        task_id: str,
        package: str,
        instance_id: str,
        config: dict[str, Any],
        config_path: Path | None = None,
        gold_path: Path | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> "InstanceSpec":
        """Build an instance spec from a visible instance artifact.

        With ``task_metadata`` the task config's level replaces the instance file's
        own level; the raw file stays unchanged in ``config``.
        """
        metadata = copy.deepcopy(config.get("metadata", {}))
        if task_metadata is not None:
            metadata = with_task_level(metadata, task_metadata)
        return cls(
            task_id=task_id,
            instance_id=instance_id,
            package=package,
            description=str(config.get("description", "")),
            instructions=str(config.get("instructions", "")),
            visible_data=copy.deepcopy(config.get("visible", {})),
            metadata=metadata,
            config=copy.deepcopy(config),
            version=str(config.get("instance_version", config.get("version", "1"))),
            gold_path=gold_path,
            config_path=config_path,
        )

    def to_public_metadata(self) -> dict[str, Any]:
        """Return agent-visible metadata for this instance."""
        metadata = copy.deepcopy(self.metadata)
        metadata.update(copy.deepcopy(self.visible_data))
        metadata["task_id"] = self.task_id
        metadata["instance_id"] = self.instance_id
        metadata["instance_version"] = self.version
        return metadata


@dataclass(frozen=True)
class TaskSpec:
    """Immutable manifest-backed description of a benchmark task family."""

    task_id: str
    package: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    task_class_path: str | None = None
    api_class_path: str | None = None
    scorer_class_path: str | None = None
    policy_class_path: str | None = None
    default_instance_id: str = "default"
    config: dict[str, Any] = field(default_factory=dict)
    package_dir: Path | None = None

    @classmethod
    def from_config(
        cls,
        package: str,
        config: dict[str, Any],
        *,
        package_dir: Path | None = None,
    ) -> "TaskSpec":
        """Build a task spec from a task package ``config.json`` payload."""
        task_id = config.get("task_id") or config.get("id")
        if not task_id:
            raise ValueError(f"Task package '{package}' is missing task_id")
        task_class_path = (
            config.get("task_class")
            or config.get("task", {}).get("class")
            or config.get("task", {}).get("class_path")
        )
        api_class_path = (
            config.get("api_class")
            or config.get("api", {}).get("class")
            or config.get("api", {}).get("class_path")
        )
        scorer_class_path = (
            config.get("scorer_class")
            or config.get("scorer", {}).get("class")
            or config.get("scorer", {}).get("class_path")
        )
        policy_class_path = (
            config.get("policy_class")
            or config.get("policy", {}).get("class")
            or config.get("policy", {}).get("class_path")
        )
        if not api_class_path:
            raise ValueError(f"Task spec '{task_id}' must declare api_class")
        if not scorer_class_path:
            raise ValueError(f"Task spec '{task_id}' must declare scorer_class")
        if not policy_class_path:
            raise ValueError(f"Task spec '{task_id}' must declare policy_class")
        if package_dir is None:
            module = import_module(package)
            module_file = getattr(module, "__file__", None)
            package_dir = Path(module_file).resolve().parent if module_file else None
        return cls(
            task_id=str(task_id),
            package=package,
            description=str(config.get("description", "")),
            metadata=copy.deepcopy(config.get("metadata", {})),
            version=str(config.get("task_version", config.get("version", "1"))),
            task_class_path=task_class_path,
            api_class_path=api_class_path,
            scorer_class_path=scorer_class_path,
            policy_class_path=policy_class_path,
            default_instance_id=str(config.get("default_instance_id", "default")),
            config=copy.deepcopy(config),
            package_dir=package_dir,
        )

    def create_api(self, *, session: "RunSession") -> AgentAPI:
        """Create the task-specific public API for one run session."""
        api_cls = import_from_path(self.api_class_path)
        api = api_cls(session=session)
        if not isinstance(api, AgentAPI):
            raise TypeError(f"API class '{self.api_class_path}' must implement AgentAPI")
        return api

    def create_policy(self) -> "TaskPolicy":
        """Create the task policy declared by the manifest."""
        from mini_bang.framework.policy import TaskPolicy
        policy_cls = import_from_path(self.policy_class_path)
        policy = policy_cls()
        if not isinstance(policy, TaskPolicy):
            raise TypeError(f"Policy class '{self.policy_class_path}' must implement TaskPolicy")
        return policy

    def create_scorer(self) -> "Scorer":
        """Create the task scorer declared by the manifest."""
        from mini_bang.framework.scoring import Scorer
        scorer_cls = import_from_path(self.scorer_class_path)
        scorer = scorer_cls()
        if not isinstance(scorer, Scorer):
            raise TypeError(f"Scorer class '{self.scorer_class_path}' must implement Scorer")
        return scorer

    def instantiate_task(self) -> "SimulationTask":
        """Instantiate an optional package adapter; the core runtime does not call it.
        The adapter supplies an object facade for package extension points."""
        if not self.task_class_path:
            raise ValueError(f"Task spec '{self.task_id}' does not declare task_class")
        task_cls = import_from_path(self.task_class_path)
        task = task_cls()
        if not isinstance(task, SimulationTask):
            raise TypeError(f"Task class '{self.task_class_path}' must be a SimulationTask")
        task._task_spec = self
        return task

    def config_copy(self) -> dict[str, Any]:
        """Return a deep copy of the raw task manifest."""
        return copy.deepcopy(self.config)


@dataclass
class RunSession:
    """Session-owned budgets, trace, timing, resources and backend for one run.
    Hidden gold artifacts and scoring logic are intentionally absent."""

    task_spec: TaskSpec
    instance_spec: InstanceSpec
    backend: Any
    policy: "TaskPolicy"
    run_id: str = field(default_factory=lambda: uuid4().hex)
    budget_counters: dict[str, int] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    resource_usage: dict[str, Any] = field(default_factory=dict)
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    environment: TaskEnvironment | None = None
    artifact_resolver: ArtifactResolver = field(default_factory=create_default_artifact_resolver)
    runtime_context: RuntimeContext = field(default_factory=RuntimeContext.from_env)
    runtime_services: RuntimeServices | None = None
    _lock: Any = field(default_factory=threading.RLock, repr=False)
    _checkpoint: Any = field(default=None, repr=False)

    @property
    def task_id(self) -> str:
        return self.task_spec.task_id

    @property
    def instance_id(self) -> str:
        return self.instance_spec.instance_id

    def budget_snapshot(self) -> dict[str, Any]:
        capabilities = self.policy.capabilities(self)
        return {
            "counters": copy.deepcopy(self.budget_counters),
            "capabilities": copy.deepcopy(capabilities),
        }

    def remaining_calls(self) -> int:
        capabilities = self.policy.capabilities(self)
        return int(capabilities.get("remaining_generate_calls", 0))

    def capabilities(self) -> dict[str, Any]:
        caps = copy.deepcopy(self.policy.capabilities(self))
        caps["task_id"] = self.task_id
        caps["instance_id"] = self.instance_id
        return caps

    def execute_simulation(
        self,
        *,
        method_name: str,
        request: Any,
        materialize: bool = True,
    ) -> Any:
        """Validate, execute, and trace one task API simulation request."""
        results = self.execute_simulation_batch(
            method_name=method_name, requests=[request], materialize=materialize
        )
        return results[0]

    def execute_simulation_batch(
        self,
        *,
        method_name: str,
        requests: list[Any],
        materialize: bool = True,
    ) -> list[Any]:
        """Validate, execute, and trace multiple simulation requests in one backend call."""
        contexts = self._validate_and_consume_requests(
            method_name=method_name,
            requests=requests,
        )
        requests = [context["backend_request"] for context in contexts]
        with request_views.private_backend_errors(contexts):
            results = run_batch_with_optional_context(
                self.backend, requests, context=self.runtime_context, services=self.runtime_services)
        if len(results) != len(contexts):
            raise RuntimeError(
                f"Backend returned {len(results)} results for {len(contexts)} requests"
            )
        return [
            self._record_simulation_result(
                context=context,
                result=result,
                materialize=materialize,
            )
            for context, result in zip(contexts, results)
        ]

    async def execute_simulation_async(
        self,
        *,
        method_name: str,
        request: Any,
        materialize: bool = True,
    ) -> Any:
        """Async variant of ``execute_simulation`` for concurrent agents."""
        results = await self.execute_simulation_batch_async(
            method_name=method_name, requests=[request], materialize=materialize
        )
        return results[0]

    async def execute_simulation_batch_async(
        self,
        *,
        method_name: str,
        requests: list[Any],
        materialize: bool = True,
    ) -> list[Any]:
        """Validate, execute, and trace multiple simulation requests asynchronously."""
        contexts = self._validate_and_consume_requests(
            method_name=method_name,
            requests=requests,
        )
        backend_requests = [context["backend_request"] for context in contexts]
        with request_views.private_backend_errors(contexts):
            results = await arun_batch_with_optional_context(
                self.backend, backend_requests, context=self.runtime_context, services=self.runtime_services)
        if len(results) != len(contexts):
            raise RuntimeError(
                f"Backend returned {len(results)} results for {len(contexts)} requests"
            )
        return [
            self._record_simulation_result(
                context=context,
                result=result,
                materialize=materialize,
            )
            for context, result in zip(contexts, results)
        ]

    def _validate_and_consume_requests(
        self,
        *,
        method_name: str,
        requests: list[Any],
    ) -> list[dict[str, Any]]:
        contexts = [
            self._new_request_context(method_name=method_name, request=request)
            for request in requests
        ]
        with self._lock:
            budget_counters_before = copy.deepcopy(self.budget_counters)
            batch_budget_before = self.budget_snapshot()
            try:
                validate_batch = getattr(self.policy, 'validate_batch', None)
                if validate_batch is not None:
                    validate_batch(session=self, method_name=method_name, requests=requests)
                for batch_index, context in enumerate(contexts):
                    context["batch_index"] = batch_index
                    context["batch_size"] = len(contexts)
                    context["budget_before"] = self.budget_snapshot()
                    self.policy.validate_request(
                        task_spec=self.task_spec,
                        instance_spec=self.instance_spec,
                        session=self,
                        method_name=method_name,
                        request=context["request"],
                    )
                    self.policy.consume_request(
                        session=self,
                        method_name=method_name,
                        request=context["request"],
                    )
                    context["budget_after"] = self.budget_snapshot()
                    context["backend_request"] = request_views.backend_request(self, context["request"])
            except Exception as exc:
                self.budget_counters.clear()
                self.budget_counters.update(budget_counters_before)
                failed_context = context if "context" in locals() else None
                failed_started_at = failed_context["started_at"] if failed_context else time.time()
                failed_started = failed_context["started"] if failed_context else time.perf_counter()
                self.trace.append(
                    {
                        "event": "simulation_request_rejected",
                        "method": method_name,
                        "request_hash": failed_context["request_hash"] if failed_context else None,
                        "request": failed_context["request_payload"] if failed_context else None,
                        "batch_size": len(contexts),
                        "batch_index": failed_context.get("batch_index") if failed_context else None,
                        "timestamp_start": failed_started_at,
                        "timestamp_end": time.time(),
                        "wall_seconds": time.perf_counter() - failed_started,
                        "budget_before": batch_budget_before,
                        "budget_after": self.budget_snapshot(),
                        "error": repr(exc),
                    }
                )
                raise
            from mini_bang.execution.progress import report_admission
            report_admission(contexts)
            if self._checkpoint is not None: self._checkpoint()
        return contexts

    def _new_request_context(self, *, method_name: str, request: Any) -> dict[str, Any]:
        request_hash = request.request_hash
        request_payload = request.to_payload()
        started_at = time.time()
        started = time.perf_counter()
        return {
            "method_name": method_name,
            "request": request,
            "request_hash": request_hash,
            "request_payload": request_payload,
            "started_at": started_at,
            "started": started,
        }

    def _record_simulation_result(
        self,
        *,
        context: dict[str, Any],
        result: Any,
        materialize: bool,
    ) -> Any:
        if request_views.resolved(context):  # public values only (framework/request_views.py)
            return request_views.record_resolved_result(self, context, result, materialize)
        elapsed = time.perf_counter() - context["started"]
        backend_metadata = getattr(result, "backend_metadata", {}) or {}
        resource_delta = getattr(result, "resource_usage", {}) or {}
        with self._lock:
            for key, value in resource_delta.items():
                if isinstance(value, (int, float)):
                    self.resource_usage[key] = self.resource_usage.get(key, 0) + value
            if backend_metadata:
                self.backend_metadata.update(copy.deepcopy(backend_metadata))
            versions = getattr(result, "versions", {}) or {}
            payload_ref = getattr(result, "payload_ref", None)
            response_metadata = {
                "cache_hit": bool(getattr(result, "cache_hit", False)),
                "materialized": materialize,
                "versions": copy.deepcopy(versions),
            }
            if payload_ref is not None:
                response_metadata["payload_ref"] = payload_ref.to_dict()
            self.trace.append(
                {
                    "event": "simulation_request",
                    "method": context["method_name"],
                    "request_hash": context["request_hash"],
                    "request": context["request_payload"],
                    "timestamp_start": context["started_at"],
                    "timestamp_end": time.time(),
                    "wall_seconds": elapsed,
                    "backend_metadata": copy.deepcopy(backend_metadata),
                    "budget_before": context["budget_before"],
                    "budget_after": context["budget_after"],
                    "resource_delta": copy.deepcopy(resource_delta),
                    "response_metadata": response_metadata,
                    "execution_trace": copy.deepcopy(getattr(result, "trace", [])),
                }
            )
        if not materialize:
            handle = getattr(result, "to_payload_handle", None)
            if callable(handle):
                return handle()
        materializer = getattr(result, "materialize_payload", None)
        if callable(materializer):
            return materializer()
        return result.payload

class SimulationTask:
    """Thin optional adapter for task packages.

    The artifact-driven runtime is built from ``TaskSpec`` and
    ``InstanceSpec``; task objects must not own benchmark-critical state.
    """
    task_id: str
    _task_spec: TaskSpec | None = None

    @property
    def spec(self) -> TaskSpec | None:
        """Return the manifest-backed spec when this adapter was discovered."""
        return self._task_spec
