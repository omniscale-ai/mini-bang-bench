"""Hidden scoring in a warm interpreter with fresh authority for every work."""
from __future__ import annotations

import os
from pathlib import Path

from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring import score_submission
from mini_bang.importing import import_from_path
from mini_bang.pool.observation import prepare_work, retained_result
from mini_bang.worker.storage_retry import commit_prepared


def process(work):
    from mini_bang.worker import k8s_scoring_worker as worker

    manifest, store = prepare_work(work)
    cached = retained_result(work)
    if cached is not None:
        return {**cached, "worker_pid": os.getpid()}
    created_at = worker._now()
    try:
        task_spec = worker._task_spec(manifest["task_spec"])
        instance_spec = worker._instance_spec(manifest["instance_spec"])
        submission = worker._submission(manifest["submission"])
        record = RunRecord(**manifest["provisional_record"])
        raw_gold = manifest.get("gold", {})
        gold = GoldStore(task_id=manifest["task_id"], instance_id=manifest["instance_id"],
            path=Path(raw_gold.get("path") or "/hidden/gold.json"),
            data=dict(raw_gold.get("data") or raw_gold), version=str(raw_gold.get("version", "1")))
        context = worker._runtime_context(manifest["scoring_context"])
        # These mutable authority-bearing services are deliberately never cached.
        trace, usage = [], {}
        services = worker._scoring_services(task_spec, context, manifest, trace, usage)
        validation = score_submission(import_from_path(manifest["scorer_class_path"])(),
            task_spec=task_spec, instance_spec=instance_spec, submission=submission,
            run_record=record, gold_store=gold, scoring_services=services)
        final_record = record.with_score(validation)
        final_record.scoring_trace, final_record.scoring_resource_usage = trace, usage
        envelope = {"job_id": manifest["job_id"], "run_id": manifest["run_id"],
            "task_id": manifest["task_id"], "instance_id": manifest["instance_id"],
            "status": "succeeded", "validation": {"success": validation.success,
                "details": validation.details, "metrics": validation.metrics or {}},
            "run_record": final_record.to_dict(), "scoring_trace": trace,
            "resource_usage": usage, "error": None,
            "created_at": created_at, "finished_at": worker._now()}
    except Exception as error:
        # Unknown scorer/transport errors remain infrastructure, never a scored
        # participant failure. Preserve the established scoring envelope shape.
        envelope = {"job_id": manifest["job_id"], "run_id": manifest["run_id"],
            "task_id": manifest.get("task_id"), "instance_id": manifest.get("instance_id"),
            "status": "failed", "validation": None, "run_record": manifest.get("provisional_record"),
            "scoring_trace": [], "resource_usage": {}, "error": {
                "type": type(error).__name__, "message": str(error),
                "category": "infrastructure_failed", "reason": "scoring_worker_failure"},
            "created_at": created_at, "finished_at": worker._now()}
    commit_prepared(store, manifest, envelope)
    return {**retained_result(work), "retained": False, "worker_pid": os.getpid()}
