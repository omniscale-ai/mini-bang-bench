from __future__ import annotations

import dataclasses
import inspect
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import urlparse

from mini_bang.artifacts import ArtifactHandle, ArtifactRef, coerce_artifact_ref
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.executable import (
    ExecutableSubmissionRunner,
    ExecutionLimits,
    ExecutionResult,
)
from mini_bang.framework.execution_watchdog import ExecutionAborted
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission, ValidationResult

# `executable-forced-hold`, option (a): an abort whose evidence comes only
# from the participant's own process (signal, exit 70/71/75/76, idling, CPU fuse)
# is retried once; a second such abort scores 0 as a failed submission.
# Any other evidence (`trusted_monitor`, `harness_reported_infrastructure` for the
# harness's own faults, none) stays held (ExecutionAborted).
PARTICIPANT_ABORT_RETRIES = 1
# The watchdog's own exit code for an aborted run; any nonzero code means failure.
PARTICIPANT_ABORT_EXIT_CODE = 124


@dataclass(frozen=True)
class ScoringPlan:
    """Task-owned hidden evaluation plan with framework-opaque case data."""

    run_id: str
    task_id: str
    instance_id: str
    cases: list[dict[str, Any]]
    inputs: dict[str, Any]
    targets: dict[str, Any]
    limits: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringRequirements:
    """Runtime services that a task manifest's ``scoring`` block declares."""

    requires_services: bool = False
    requires_executable_runner: bool = False

    @property
    def requires_runtime(self) -> bool:
        return self.requires_services or self.requires_executable_runner


class ScoringServices:
    """Scorer-facing wrapper over shared execution services and scorer authority."""

    def __init__(
        self,
        execution_services: ExecutionServices,
        scoring_context: RuntimeContext,
        executable_runner: ExecutableSubmissionRunner | None = None,
        *,
        trace: list[dict[str, Any]] | None = None,
        resource_usage: dict[str, Any] | None = None,
    ) -> None:
        self._execution_services = execution_services
        self._scoring_context = scoring_context
        self._executable_runner = executable_runner
        self.trace = trace if trace is not None else []
        self.resource_usage = resource_usage if resource_usage is not None else {}

    @property
    def scoring_context(self) -> RuntimeContext:
        return self._scoring_context

    def run_simulation(self, request: SimulationRequest) -> SimulationResult:
        return self.run_batch([request])[0]

    def run_batch(self, requests: Iterable[SimulationRequest]) -> list[SimulationResult]:
        started = time.perf_counter()
        timestamp = time.time()
        request_list = list(requests)
        for request in request_list:
            self.trace.append(_scoring_started_event(request, timestamp))
        try:
            results = self._execution_services.run_batch(
                request_list,
                context=self._scoring_context,
            )
        except Exception as exc:
            self._trace_simulation_failures(request_list, started, exc)
            raise
        for result in results:
            self._record_simulation_finished(result, started)
        return results

    def materialize_artifact(self, ref: ArtifactRef | dict[str, Any]) -> ArtifactHandle:
        artifact_ref = coerce_artifact_ref(ref)
        started = time.perf_counter()
        self.trace.append(
            {
                "event": "scoring_artifact_materialization_started",
                "timestamp_start": time.time(),
                "hidden": True,
                "ref_redacted": True,
                "scheme": urlparse(artifact_ref.uri).scheme or "file",
            }
        )
        try:
            handle = self._execution_services.materialize_artifact(
                artifact_ref,
                context=self._scoring_context,
            )
        except Exception as exc:
            self.trace.append(
                {
                    "event": "scoring_artifact_materialization_failed",
                    "timestamp_end": time.time(),
                    "wall_seconds": time.perf_counter() - started,
                    "hidden": True,
                    "ref_redacted": True,
                    "error": repr(exc),
                }
            )
            raise
        self._add_resource_usage({"artifact_bytes": handle.size_bytes or 0})
        self.trace.append(
            {
                "event": "scoring_artifact_materialized",
                "timestamp_end": time.time(),
                "wall_seconds": time.perf_counter() - started,
                "hidden": True,
                "ref_redacted": True,
                "size_bytes": handle.size_bytes,
                "sha256": handle.sha256,
                "media_type": handle.media_type,
            }
        )
        return handle

    def run_executable(
        self,
        submission_ref: Any,
        input_manifest: dict[str, Any],
        limits: ExecutionLimits | dict[str, Any] | None = None,
        output_schema: Callable[[dict[str, Any]], None] | None = None,
    ) -> ExecutionResult:
        if self._executable_runner is None:
            raise RuntimeError("ScoringServices requires an ExecutableSubmissionRunner")
        sandbox = None
        if self._scoring_context.security.get("official_mode"):
            if self._executable_runner.__class__.__name__ == "InProcessExecutableRunner":
                raise RuntimeError("official scoring cannot use InProcessExecutableRunner")
            from mini_bang.framework.sandbox_policy import require_hardened_runner

            sandbox = require_hardened_runner(self._executable_runner)
        resolved_limits = _coerce_limits(limits)
        aborts: list[dict[str, Any]] = []
        while True:
            attempt = len(aborts) + 1
            started = time.perf_counter()
            self.trace.append(
                {
                    "event": "scoring_executable_started",
                    "timestamp_start": time.time(),
                    "attempt": attempt,
                    "hidden": True,
                    "input_manifest_redacted": True,
                    "limits": dataclasses.asdict(resolved_limits),
                    **({"sandbox": sandbox} if sandbox is not None else {}),
                }
            )
            try:
                # Each call is a new one-shot Pod or process with the same limits.
                result = self._executable_runner.run(
                    submission_ref,
                    input_manifest,
                    resolved_limits,
                    output_schema,
                )
            except Exception as exc:
                self._add_resource_usage({"executable_runs": 1, "executable_failures": 1})
                abort = _participant_abort(exc, attempt)
                self.trace.append(_executable_finished_event(
                    started, error=repr(exc), attempt=attempt, participant_abort=abort))
                if abort is None:
                    if aborts and isinstance(exc, ExecutionAborted):
                        exc.metadata["participant_abort_retry"] = _retry_record(
                            aborts, "held: retry ended without participant_process evidence", attempt)
                    raise
                aborts.append(abort)
                if len(aborts) <= PARTICIPANT_ABORT_RETRIES:
                    self._add_resource_usage({"executable_participant_abort_retries": 1})
                    continue
                return self._participant_abort_failure(aborts)
            if aborts:
                result = dataclasses.replace(result, metadata={
                    **result.metadata,
                    "participant_abort_retry": _retry_record(aborts, "retry result used as is", attempt),
                })
            usage = {"executable_runs": 1}
            if result.timed_out:
                usage["executable_timeouts"] = 1
            if result.exit_code != 0:
                usage["executable_failures"] = 1
            self._add_resource_usage(usage)
            self.trace.append(_executable_finished_event(started, result=result, attempt=attempt))
            return result

    def _participant_abort_failure(self, aborts: list[dict[str, Any]]) -> ExecutionResult:
        """Repeated self-abort of submitted code is a failed submission, not a hold."""
        record = _retry_record(aborts, "scored as failed submission", len(aborts))
        self._add_resource_usage({"executable_participant_abort_failures": 1})
        self.trace.append(
            {
                "event": "scoring_executable_participant_abort_failure",
                "timestamp_end": time.time(),
                "hidden": True,
                "participant_abort_retry": record,
            }
        )
        return ExecutionResult(
            exit_code=PARTICIPANT_ABORT_EXIT_CODE,
            stderr=(
                f"submitted code aborted in {len(aborts)} fresh sandboxes with participant_process "
                f"evidence ({', '.join(item['reason'] for item in aborts)}); failed submission"
            ),
            metadata={
                "outcome": "participant_failure",
                "abort_reason": aborts[-1]["reason"],
                "attribution_evidence": "participant_process",
                "participant_abort_retry": record,
            },
        )

    def _record_simulation_finished(
        self,
        result: SimulationResult,
        started: float,
    ) -> None:
        self._add_resource_usage(result.resource_usage)
        payload_ref = result.payload_ref.to_dict() if result.payload_ref is not None else None
        self.trace.append(
            {
                "event": "scoring_simulation_finished",
                "timestamp_end": time.time(),
                "wall_seconds": time.perf_counter() - started,
                "request_hash": result.request_hash,
                "hidden": True,
                "request_redacted": True,
                "backend_metadata": result.backend_metadata,
                "resource_delta": result.resource_usage,
                "payload_ref": payload_ref,
                "execution_trace_event_count": len(result.trace),
            }
        )

    def _trace_simulation_failures(
        self,
        requests: list[SimulationRequest],
        started: float,
        exc: Exception,
    ) -> None:
        for request in requests:
            self.trace.append(
                {
                    "event": "scoring_simulation_failed",
                    "timestamp_end": time.time(),
                    "wall_seconds": time.perf_counter() - started,
                    "request_hash": request.request_hash,
                    "hidden": True,
                    "request_redacted": True,
                    "error": repr(exc),
                }
            )

    def _add_resource_usage(self, delta: dict[str, Any]) -> None:
        for key, value in delta.items():
            if isinstance(value, (int, float)):
                self.resource_usage[key] = self.resource_usage.get(key, 0) + value


@runtime_checkable
class Scorer(Protocol):
    """Task scorer for one immutable task instance."""

    def score(
        self,
        *,
        task_spec: TaskSpec,
        instance_spec: InstanceSpec,
        submission: TaskSubmission,
        run_record: RunRecord,
        gold_store: GoldStore,
        scoring_services: ScoringServices | None = None,
    ) -> ValidationResult:
        """Score a submission using durable run data, gold, and optional services."""
        ...


def scoring_requirements(task_spec: TaskSpec) -> ScoringRequirements:
    raw = task_spec.config.get("scoring", {})
    if not isinstance(raw, dict):
        return ScoringRequirements()
    return ScoringRequirements(
        requires_services=bool(raw.get("requires_services", False)),
        requires_executable_runner=bool(raw.get("requires_executable_runner", False)),
    )


def score_submission(
    scorer: Scorer,
    *,
    task_spec: TaskSpec,
    instance_spec: InstanceSpec,
    submission: TaskSubmission,
    run_record: RunRecord,
    gold_store: GoldStore,
    scoring_services: ScoringServices | None = None,
) -> ValidationResult:
    kwargs = {
        "task_spec": task_spec,
        "instance_spec": instance_spec,
        "submission": submission,
        "run_record": run_record,
        "gold_store": gold_store,
    }
    if _score_accepts_scoring_services(scorer):
        kwargs["scoring_services"] = scoring_services
    elif scoring_services is not None:
        raise TypeError(
            "task scoring config requested ScoringServices, but scorer.score does not "
            "accept a scoring_services keyword"
        )
    return scorer.score(**kwargs)


def scoring_context_from_config(task_spec: TaskSpec) -> RuntimeContext | None:
    scoring = task_spec.config.get("scoring", {})
    if not isinstance(scoring, dict) or not isinstance(scoring.get("runtime_context"), dict):
        return None
    raw = dict(scoring["runtime_context"])
    return RuntimeContext(
        artifact=dict(raw.get("artifact") or {}),
        payload=dict(raw.get("payload") or {}),
        security=dict(raw.get("security") or {}),
        env=dict(raw.get("env") or {}),
    )


def _score_accepts_scoring_services(scorer: Scorer) -> bool:
    try:
        signature = inspect.signature(scorer.score)
    except (TypeError, ValueError):
        return False
    return (
        "scoring_services" in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    )


def _scoring_started_event(request: SimulationRequest, timestamp: float) -> dict[str, Any]:
    return {
        "event": "scoring_simulation_started",
        "timestamp_start": timestamp,
        "request_hash": request.request_hash,
        "hidden": True,
        "request_redacted": True,
    }


def _coerce_limits(limits: ExecutionLimits | dict[str, Any] | None) -> ExecutionLimits:
    if limits is None:
        return ExecutionLimits()
    if isinstance(limits, ExecutionLimits):
        return limits
    if isinstance(limits, dict):
        return ExecutionLimits(**limits)
    raise TypeError(f"Unsupported executable limits type: {type(limits).__name__}")


def _participant_abort(exc: Exception, attempt: int) -> dict[str, Any] | None:
    """Compact record of an abort whose only evidence is the participant's process."""
    if not isinstance(exc, ExecutionAborted):
        return None
    metadata = exc.metadata
    if metadata.get("attribution_evidence") != "participant_process":
        return None
    return {
        "attempt": attempt,
        "reason": str(exc.reason),
        "phase": exc.phase,
        "attribution_evidence": "participant_process",
        # Full diagnostics stay in the runner's result store under this id.
        "job_id": metadata.get("job_id") or metadata.get("work_id"),
        "exit_code": metadata.get("exit_code"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
    }


def _retry_record(aborts: list[dict[str, Any]], resolution: str, attempts: int) -> dict[str, Any]:
    return {
        "rule": "executable-forced-hold (a): retry once in a fresh one-shot sandbox; "
        "a second participant_process abort is a failed submission",
        "participant_aborts": [dict(item) for item in aborts],
        "attempts": attempts,
        "resolution": resolution,
    }


def _executable_finished_event(
    started: float,
    *,
    result: ExecutionResult | None = None,
    error: str | None = None,
    attempt: int = 1,
    participant_abort: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "event": "scoring_executable_finished",
        "timestamp_end": time.time(),
        "wall_seconds": time.perf_counter() - started,
        "attempt": attempt,
        "hidden": True,
        "input_manifest_redacted": True,
    }
    if participant_abort is not None:
        event["participant_abort"] = participant_abort
    if result is not None:
        event.update(
            {
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stdout_bytes": len(result.stdout.encode("utf-8", errors="replace")),
                "stderr_bytes": len(result.stderr.encode("utf-8", errors="replace")),
                "metadata": result.metadata,
            }
        )
    if error is not None:
        event["error"] = error
    return event
