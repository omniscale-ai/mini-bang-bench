from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from mini_bang.framework.task import TaskSubmission, ValidationResult
from mini_bang.agent_types import to_serializable


@dataclass
class RunRecord:
    """Durable, fully serializable artifact for one benchmark run."""

    run_id: str
    task_id: str
    instance_id: str
    submission: dict[str, Any]
    trace: list[dict[str, Any]] = field(default_factory=list)
    scoring_trace: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    resource_usage: dict[str, Any] = field(default_factory=dict)
    scoring_resource_usage: dict[str, Any] = field(default_factory=dict)
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    score_summary: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        *,
        run_id: str,
        task_id: str,
        instance_id: str,
        submission: TaskSubmission,
        trace: list[dict[str, Any]] | None = None,
        scoring_trace: list[dict[str, Any]] | None = None,
        timing: dict[str, Any] | None = None,
        resource_usage: dict[str, Any] | None = None,
        scoring_resource_usage: dict[str, Any] | None = None,
        backend_metadata: dict[str, Any] | None = None,
        versions: dict[str, Any] | None = None,
        validation: ValidationResult | None = None,
    ) -> "RunRecord":
        """Create a run record from harness-owned run components."""
        score_summary: dict[str, Any] = {}
        if validation is not None:
            score_summary = {
                "success": validation.success,
                "details": validation.details,
                "metrics": validation.metrics or {},
            }
        return cls(
            run_id=run_id,
            task_id=task_id,
            instance_id=instance_id,
            submission=to_serializable(
                {"answer": submission.answer, "artifacts": submission.artifacts}
            ),
            trace=to_serializable(trace or []),
            scoring_trace=to_serializable(scoring_trace or []),
            timing=to_serializable(timing or {}),
            resource_usage=to_serializable(resource_usage or {}),
            scoring_resource_usage=to_serializable(scoring_resource_usage or {}),
            backend_metadata=to_serializable(backend_metadata or {}),
            score_summary=to_serializable(score_summary),
            versions=to_serializable(versions or {}),
        )

    @classmethod
    def from_result(
        cls,
        *,
        run_id: str,
        task_id: str,
        instance_id: str,
        submission: TaskSubmission,
        validation: ValidationResult,
        trace: list[dict[str, Any]] | None = None,
        scoring_trace: list[dict[str, Any]] | None = None,
        timing: dict[str, Any] | None = None,
        resource_usage: dict[str, Any] | None = None,
        scoring_resource_usage: dict[str, Any] | None = None,
        backend_metadata: dict[str, Any] | None = None,
        versions: dict[str, Any] | None = None,
    ) -> "RunRecord":
        """Create a scored run record."""
        return cls.from_parts(
            run_id=run_id,
            task_id=task_id,
            instance_id=instance_id,
            submission=submission,
            validation=validation,
            trace=trace,
            scoring_trace=scoring_trace,
            timing=timing,
            resource_usage=resource_usage,
            scoring_resource_usage=scoring_resource_usage,
            backend_metadata=backend_metadata,
            versions=versions,
        )

    def with_score(self, validation: ValidationResult) -> "RunRecord":
        """Return a copy with score summary attached."""
        data = self.to_dict()
        data["score_summary"] = to_serializable(
            {
                "success": validation.success,
                "details": validation.details,
                "metrics": validation.metrics or {},
            }
        )
        return RunRecord(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return this record as a JSON-compatible dictionary."""
        return to_serializable(dataclasses.asdict(self))
