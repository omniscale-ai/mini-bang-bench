from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mini_bang.execution.models import SimulationRequest

if TYPE_CHECKING:  # pragma: no cover
    from mini_bang.framework.task import InstanceSpec, RunSession, TaskSpec


class RequestRejectedError(RuntimeError):
    """A policy rejected a scientific request before admission or budget debit."""


class PublicRequestError(ValueError):
    """A request rejection whose message is safe to show to the agent.

    The official runner returns the message of this error unchanged. Any other
    rejection reaches the agent only as a generic failure, so task code that
    wants the agent to read the reason raises this class or a subclass.
    """


class BudgetExhaustedError(RequestRejectedError, PublicRequestError):
    """The task budget has no room for the request; the agent sees this reason."""


@runtime_checkable
class TaskPolicy(Protocol):
    """Task-specific request policy for capabilities, validation, and budgets."""

    def initialize_session(self, session: "RunSession") -> None:
        """Initialize session-owned counters before an agent receives the API."""
        ...

    def capabilities(self, session: "RunSession") -> dict:
        """Return machine-readable limits and allowed request shapes."""
        ...

    def validate_request(
        self,
        *,
        task_spec: "TaskSpec",
        instance_spec: "InstanceSpec",
        session: "RunSession",
        method_name: str,
        request: SimulationRequest,
    ) -> None:
        """Raise an exception when a simulation request violates task policy."""
        ...

    def consume_request(
        self,
        *,
        session: "RunSession",
        method_name: str,
        request: SimulationRequest,
    ) -> None:
        """Update session-owned counters after validation and before execution."""
        ...


class BaseTaskPolicy(TaskPolicy):
    """Small default policy for tasks that only need a call budget."""

    budget_key = "generate_samples"

    def initialize_session(self, session: "RunSession") -> None:
        session.budget_counters.setdefault(self.budget_key, 0)

    def max_calls(self, session: "RunSession") -> int:
        """Return the configured request budget for this task."""
        raw = session.task_spec.config.get("api", {}).get("max_generate_calls", 0)
        return int(raw) if raw is not None else 0

    def remaining_calls(self, session: "RunSession") -> int:
        """Return remaining high-level API calls."""
        limit = self.max_calls(session)
        used = session.budget_counters.get(self.budget_key, 0)
        return max(0, limit - used)

    def capabilities(self, session: "RunSession") -> dict:
        return {
            "max_generate_calls": self.max_calls(session),
            "used_generate_calls": session.budget_counters.get(self.budget_key, 0),
            "remaining_generate_calls": self.remaining_calls(session),
        }

    def validate_request(
        self,
        *,
        task_spec: "TaskSpec",
        instance_spec: "InstanceSpec",
        session: "RunSession",
        method_name: str,
        request: SimulationRequest,
    ) -> None:
        if method_name != self.budget_key:
            raise ValueError(f"Unsupported task API method '{method_name}'")
        if request.runs < 1:
            raise ValueError("runs must be at least 1")
        _validate_resource_override(task_spec, request)
        if self.remaining_calls(session) <= 0:
            raise BudgetExhaustedError(f"{self.budget_key} call limit exceeded for this task")

    def consume_request(
        self,
        *,
        session: "RunSession",
        method_name: str,
        request: SimulationRequest,
    ) -> None:
        session.budget_counters[self.budget_key] = (
            session.budget_counters.get(self.budget_key, 0) + 1
        )


@dataclass(frozen=True)
class RequestCost:
    request_count: int = 1
    raw_step_count: int = 0
    field_count: int = 0
    estimated_bytes: int = 0
    metric_count: int = 0
    case_count: int = 0
    executable_submission_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "request_count": self.request_count,
            "raw_step_count": self.raw_step_count,
            "field_count": self.field_count,
            "estimated_bytes": self.estimated_bytes,
            "metric_count": self.metric_count,
            "case_count": self.case_count,
            "executable_submission_count": self.executable_submission_count,
        }


@dataclass
class AllowedParamPolicy:
    macro_params: set[str] = field(default_factory=set)
    micro_params: set[str] = field(default_factory=set)
    sample_params: set[str] = field(default_factory=set)

    def validate(self, request: SimulationRequest) -> None:
        _reject_extra_keys(request.macro_params, self.macro_params, "macro_params")
        _reject_extra_keys(request.micro_params, self.micro_params, "micro_params")
        _reject_extra_keys(request.sample_params, self.sample_params, "sample_params")


@dataclass
class WindowPolicy:
    max_raw_steps: int | None = None
    start_key: str = "start_step"
    end_key: str = "end_step"

    def validate(self, request: SimulationRequest) -> None:
        if self.max_raw_steps is None:
            return
        start = int(request.sample_params.get(self.start_key, 0))
        end = int(request.sample_params.get(self.end_key, request.sample_params.get("rollout_steps", 0)))
        if max(0, end - start) > self.max_raw_steps:
            raise ValueError("requested observation window exceeds raw-step limit")


@dataclass
class FieldPolicy:
    allowed_fields: set[str] = field(default_factory=set)
    max_fields: int | None = None

    def validate(self, request: SimulationRequest) -> None:
        fields = list(request.sample_params.get("fields") or [])
        if self.max_fields is not None and len(fields) > self.max_fields:
            raise ValueError("too many fields requested")
        if self.allowed_fields:
            invalid = sorted(set(fields) - self.allowed_fields)
            if invalid:
                raise ValueError(f"fields are not allowed: {invalid}")


@dataclass
class GranularityPolicy:
    step_multiple: int | None = None

    def validate(self, request: SimulationRequest) -> None:
        if not self.step_multiple:
            return
        for key in ("start_step", "end_step", "rollout_steps"):
            value = request.sample_params.get(key)
            if value is not None and int(value) % self.step_multiple != 0:
                raise ValueError(f"{key} must be a multiple of {self.step_multiple}")


@dataclass
class SeedPolicy:
    allowed_modes: set[str] = field(
        default_factory=lambda: {"planner_derived", "agent_supplied", "fixed_by_instance"}
    )
    fixed_seed: int | None = None

    def validate(self, request: SimulationRequest) -> None:
        if request.seed_mode not in self.allowed_modes:
            raise ValueError(f"seed_mode '{request.seed_mode}' is not allowed")
        if self.fixed_seed is None:
            return
        if request.seed_mode != "fixed_by_instance":
            raise ValueError("fixed instance seed requires seed_mode='fixed_by_instance'")
        if "seed" not in request.sample_params or int(request.sample_params["seed"]) != self.fixed_seed:
            raise ValueError("fixed instance seed cannot be overridden")


@dataclass
class CostBudgetPolicy:
    limits: dict[str, int] = field(default_factory=dict)
    counter_prefix: str = "cost"

    def initialize_session(self, session: "RunSession") -> None:
        for key in RequestCost().as_dict():
            session.budget_counters.setdefault(self._counter_key(key), 0)

    def estimate(self, request: SimulationRequest) -> RequestCost:
        sample = request.sample_params
        fields = sample.get("fields") or []
        metrics = sample.get("metrics") or []
        raw_steps = int(sample.get("raw_step_count", sample.get("rollout_steps", 0)) or 0)
        return RequestCost(
            request_count=1,
            raw_step_count=max(0, raw_steps) * max(1, request.runs),
            field_count=len(fields) * max(1, request.runs),
            metric_count=len(metrics) * max(1, request.runs),
            estimated_bytes=int(sample.get("estimated_bytes", 0) or 0),
            case_count=int(sample.get("case_count", 0) or 0),
            executable_submission_count=int(sample.get("executable_submission_count", 0) or 0),
        )

    def validate_request(self, session: "RunSession", request: SimulationRequest) -> None:
        cost = self.estimate(request)
        for key, value in cost.as_dict().items():
            limit = self.limits.get(key)
            if limit is None:
                continue
            used = session.budget_counters.get(self._counter_key(key), 0)
            if used + value > limit:
                raise BudgetExhaustedError(f"{key} budget exceeded")

    def consume_request(self, session: "RunSession", request: SimulationRequest) -> None:
        for key, value in self.estimate(request).as_dict().items():
            counter_key = self._counter_key(key)
            session.budget_counters[counter_key] = session.budget_counters.get(counter_key, 0) + value

    def capabilities(self, session: "RunSession") -> dict[str, Any]:
        remaining = {}
        for key, limit in self.limits.items():
            used = session.budget_counters.get(self._counter_key(key), 0)
            remaining[key] = max(0, limit - used)
        return {"cost_limits": dict(self.limits), "remaining_costs": remaining}

    def _counter_key(self, key: str) -> str:
        return f"{self.counter_prefix}.{key}"


def _reject_extra_keys(values: dict[str, Any], allowed: set[str], label: str) -> None:
    if not allowed:
        return
    extras = sorted(set(values) - allowed)
    if extras:
        raise ValueError(f"{label} keys are not allowed: {extras}")


def _validate_resource_override(task_spec: "TaskSpec", request: SimulationRequest) -> None:
    override = request.sample_params.get("resource_spec") or request.sample_params.get("resources")
    if override is None:
        return
    resources = task_spec.config.get("resources", {})
    api = task_spec.config.get("api", {})
    allowed = bool(
        resources.get("allow_request_overrides")
        or api.get("allow_resource_overrides")
    )
    if not allowed:
        raise ValueError("request-level resource overrides are not allowed by task policy")
