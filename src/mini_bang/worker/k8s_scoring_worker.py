from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.payloads import create_payload_storage_from_context
from mini_bang.execution.result_store import result_store_from_config
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.services import BackendExecutionServices, ExecutionServices
from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring import ScoringServices, score_submission, scoring_requirements
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission
from mini_bang.importing import import_from_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-uri", default=os.getenv("MINI_BANG_K8S_JOB_INPUT_URI"))
    parser.add_argument(
        "--result-store-config",
        default=os.getenv("MINI_BANG_K8S_RESULT_STORE_CONFIG"),
    )
    args = parser.parse_args(argv)
    if not args.input_uri or not args.result_store_config:
        print("missing scoring input URI or result store config", file=sys.stderr)
        return 2
    store = result_store_from_config(_json_dict(args.result_store_config))
    manifest: dict[str, Any] | None = None
    created_at = _now()
    try:
        manifest = store.get_job_input(args.input_uri)
        task_spec = _task_spec(manifest["task_spec"])
        instance_spec = _instance_spec(manifest["instance_spec"])
        submission = _submission(manifest["submission"])
        run_record = RunRecord(**manifest["provisional_record"])
        gold = GoldStore(
            task_id=manifest["task_id"],
            instance_id=manifest["instance_id"],
            path=Path(manifest.get("gold", {}).get("path") or "/hidden/gold.json"),
            data=dict(manifest.get("gold", {}).get("data") or manifest.get("gold") or {}),
            version=str(manifest.get("gold", {}).get("version", "1")),
        )
        scoring_context = _runtime_context(manifest["scoring_context"])
        scoring_trace: list[dict[str, Any]] = []
        scoring_usage: dict[str, Any] = {}
        services = _scoring_services(task_spec, scoring_context, manifest, scoring_trace, scoring_usage)
        scorer = import_from_path(manifest["scorer_class_path"])()
        validation = score_submission(
            scorer,
            task_spec=task_spec,
            instance_spec=instance_spec,
            submission=submission,
            run_record=run_record,
            gold_store=gold,
            scoring_services=services,
        )
        final_record = run_record.with_score(validation)
        final_record.scoring_trace = scoring_trace
        final_record.scoring_resource_usage = scoring_usage
        store.put_job_result(
            manifest["run_id"],
            manifest["job_id"],
            {
                "job_id": manifest["job_id"],
                "run_id": manifest["run_id"],
                "task_id": manifest["task_id"],
                "instance_id": manifest["instance_id"],
                "status": "succeeded",
                "validation": {
                    "success": validation.success,
                    "details": validation.details,
                    "metrics": validation.metrics or {},
                },
                "run_record": final_record.to_dict(),
                "scoring_trace": scoring_trace,
                "resource_usage": scoring_usage,
                "error": None,
                "created_at": created_at,
                "finished_at": _now(),
            },
        )
        return 0
    except Exception as exc:
        if manifest is not None:
            _write_failure(store, manifest, created_at, exc)
        traceback.print_exc()
        return 1


def _scoring_services(
    task_spec: TaskSpec,
    context: RuntimeContext,
    manifest: dict[str, Any],
    trace: list[dict[str, Any]],
    usage: dict[str, Any],
) -> ScoringServices | None:
    requirements = scoring_requirements(task_spec)
    if not requirements.requires_runtime:
        return None
    runner = None
    runner_config = manifest.get("executable_runner_config")
    if requirements.requires_executable_runner:
        from mini_bang.framework.k8s_executable import K8sExecutableRunner

        runner = K8sExecutableRunner.from_config(dict(runner_config or {}))
    execution_services = _execution_services_for_scoring(context, manifest)
    return ScoringServices(
        execution_services,
        context,
        executable_runner=runner,
        trace=trace,
        resource_usage=usage,
    )


def _execution_services_for_scoring(context: RuntimeContext, manifest: dict[str, Any]):
    backend_cfg = manifest.get("scoring_simulation_backend")
    if isinstance(backend_cfg, dict) and backend_cfg.get("kind") == "k8s":
        from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig

        raw_config = dict(backend_cfg.get("config") or {})
        if isinstance(raw_config.get("default_resources"), dict):
            from mini_bang.execution.resources import ResourceSpec

            raw_config["default_resources"] = ResourceSpec.from_dict(raw_config["default_resources"])
        backend = K8sBackend(
            config=K8sBackendConfig(**raw_config),
            result_store=result_store_from_config(raw_config.get("result_store_config") or {"kind": "filesystem"}),
        )
        return BackendExecutionServices(backend)
    executor = SimulationExecutor(
        payload_storage=create_payload_storage_from_context(context),
        context=context,
    )
    return ExecutionServices(executor=executor)


def _write_failure(store, manifest: dict[str, Any], created_at: str, exc: Exception) -> None:
    try:
        store.put_job_result(
            manifest["run_id"],
            manifest["job_id"],
            {
                "job_id": manifest.get("job_id"),
                "run_id": manifest.get("run_id"),
                "task_id": manifest.get("task_id"),
                "instance_id": manifest.get("instance_id"),
                "status": "failed",
                "validation": None,
                "run_record": manifest.get("provisional_record"),
                "scoring_trace": [],
                "resource_usage": {},
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "created_at": created_at,
                "finished_at": _now(),
            },
        )
    except Exception:
        traceback.print_exc()


def _task_spec(data: dict[str, Any]) -> TaskSpec:
    payload = dict(data)
    if payload.get("package_dir"):
        payload["package_dir"] = Path(payload["package_dir"])
    return TaskSpec(**payload)


def _instance_spec(data: dict[str, Any]) -> InstanceSpec:
    payload = dict(data)
    for key in ("gold_path", "config_path"):
        if payload.get(key):
            payload[key] = Path(payload[key])
    return InstanceSpec(**payload)


def _submission(data: dict[str, Any]) -> TaskSubmission:
    return TaskSubmission(answer=data.get("answer"), artifacts=data.get("artifacts"))


def _runtime_context(data: dict[str, Any]) -> RuntimeContext:
    return RuntimeContext(
        run_id=data.get("run_id"),
        task_id=data.get("task_id"),
        instance_id=data.get("instance_id"),
        artifact=dict(data.get("artifact") or {}),
        payload=dict(data.get("payload") or {}),
        security=dict(data.get("security") or {}),
        env=dict(data.get("env") or {}),
    )


def _json_dict(value: str) -> dict[str, Any]:
    import json

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
