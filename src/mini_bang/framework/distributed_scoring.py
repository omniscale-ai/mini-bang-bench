from __future__ import annotations

import time
from typing import Any

from mini_bang.execution.runtime import RuntimeContext
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring import scoring_requirements
from mini_bang.framework.scoring_jobs import (
    ScoringJob,
    instance_spec_to_manifest,
    task_spec_to_manifest,
)
from mini_bang.framework.task import RunSession, TaskSubmission, ValidationResult


def score_with_scoring_job_backend(
    *,
    scoring_job_backend: Any,
    artifact_store: BenchmarkArtifactStore,
    session: RunSession,
    submission: TaskSubmission,
    provisional_record: RunRecord,
    scoring_context: RuntimeContext,
    force: bool = False,
) -> tuple[ValidationResult, RunRecord] | None:
    requirements = scoring_requirements(session.task_spec)
    scoring_cfg = session.task_spec.config.get("scoring", {})
    distributed = bool(isinstance(scoring_cfg, dict) and scoring_cfg.get("distributed"))
    if scoring_job_backend is None or not (force or distributed or requirements.requires_executable_runner):
        return None
    gold_store = artifact_store.load_gold(session.instance_spec)
    job = ScoringJob(
        run_id=session.run_id,
        task_id=session.task_id,
        instance_id=session.instance_id,
        provisional_record=provisional_record,
        submission=submission,
        gold={
            "data": gold_store.data,
            "version": gold_store.version,
            "path": str(gold_store.path),
        },
        scorer_class_path=session.task_spec.scorer_class_path or "",
        task_spec=task_spec_to_manifest(session.task_spec),
        instance_spec=instance_spec_to_manifest(session.instance_spec),
        scoring_context=scoring_context,
        executable_runner_config=scoring_cfg.get("executable_runner")
        if isinstance(scoring_cfg, dict)
        else None,
    )
    run = getattr(scoring_job_backend, "run", None)
    envelope = run(job) if callable(run) else scoring_job_backend.fetch(
        scoring_job_backend.submit(job)
    )
    validation_raw = dict(envelope["validation"])
    validation = ValidationResult(
        success=bool(validation_raw.get("success")),
        details=validation_raw.get("details"),
        metrics=dict(validation_raw.get("metrics") or {}),
    )
    return validation, RunRecord(**dict(envelope["run_record"]))


def mark_distributed_score_timing(
    session: RunSession,
    *,
    score_started: float,
    run_started: float,
) -> None:
    session.timing["score_seconds"] = time.perf_counter() - score_started
    session.timing["wall_seconds"] = time.perf_counter() - run_started
    session.trace.append({"event": "scored", "timestamp": time.time()})
