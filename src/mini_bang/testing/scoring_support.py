from __future__ import annotations

from typing import Any

from mini_bang.execution.models import SimulationRequest
from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.policy import BaseTaskPolicy
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring import ScoringServices
from mini_bang.framework.task import (
    InstanceSpec,
    TaskSpec,
    TaskSubmission,
    ValidationResult,
)


class ToyAPI:
    def __init__(self, *, session: Any) -> None:
        self.session = session

    def instructions(self) -> str:
        return "toy instructions"


class ToyPolicy(BaseTaskPolicy):
    pass


class OldCompatScorer:
    def score(
        self,
        *,
        task_spec: TaskSpec,
        instance_spec: InstanceSpec,
        submission: TaskSubmission,
        run_record: RunRecord,
        gold_store: GoldStore,
    ) -> ValidationResult:
        return ValidationResult(success=True, metrics={"old_signature": True})


class RuntimeSimulationScorer:
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
        if scoring_services is None:
            raise RuntimeError("RuntimeSimulationScorer requires ScoringServices")
        request = SimulationRequest(
            simulator_id="artifact_echo_test",
            saturation=1,
            macro_params={"artifact_ref": gold_store.get("artifact_ref")},
        )
        result = scoring_services.run_simulation(request)
        payload = result.materialize_payload()
        observed = payload["trajectories"][0]["size_bytes"]
        return ValidationResult(success=observed == gold_store.get("expected_size"), metrics={"size": observed})


class ExecutableScorer:
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
        if scoring_services is None:
            raise RuntimeError("ExecutableScorer requires ScoringServices")
        result = scoring_services.run_executable(
            submission.artifacts["submission_ref"] if submission.artifacts else None,
            {"cases": gold_store.get("cases", [])},
            {"timeout_seconds": 1},
        )
        return ValidationResult(
            success=result.exit_code == 0,
            metrics={"exit_code": result.exit_code},
        )
