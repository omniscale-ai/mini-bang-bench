from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import traceback
from typing import Any

from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.payloads import create_payload_storage_from_context
from mini_bang.execution.result_store import result_store_from_config
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.transport import (
    request_from_transport,
    result_to_transport,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-only", action="store_true")
    parser.add_argument("--input-uri", default=os.getenv("MINI_BANG_K8S_JOB_INPUT_URI"))
    parser.add_argument(
        "--result-store-config",
        default=os.getenv("MINI_BANG_K8S_RESULT_STORE_CONFIG"),
    )
    args = parser.parse_args(argv)
    if not args.input_uri:
        print("missing job input URI", file=sys.stderr)
        return 2
    if not args.result_store_config:
        print("missing result store config", file=sys.stderr)
        return 2
    from mini_bang.worker.storage_retry import retrying_store, commit_prepared
    store = retrying_store(result_store_from_config(_json_dict(args.result_store_config)))
    # The CLI owns supervision. Embedded callers may have in-memory simulator
    # registrations that intentionally cannot cross an interpreter boundary.
    if not args.execute_only and argv is None:
        from mini_bang.worker.observation_watchdog import supervise
        return supervise(store, args.input_uri, args.result_store_config)
    manifest: dict[str, Any] | None = None
    created_at = _now()
    try:
        manifest = store.get_job_input(args.input_uri)
        store = retrying_store(store, manifest)
        from mini_bang.worker.observation_startup import retained_outcome
        retained = retained_outcome(store, manifest)
        if retained is not None:
            return retained
        context = _runtime_context_from_dict(dict(manifest["runtime_context"]))
        requests = [request_from_transport(item) for item in manifest["requests"]]
        executor = SimulationExecutor(
            payload_storage=create_payload_storage_from_context(context),
            context=context,
        )
        from mini_bang.execution.progress import observe_progress
        def progress(kind, values):
            import json
            from mini_bang.worker.observation_watchdog import PREFIX
            selected = {key: values[key] for key in ("phase", "completed", "total", "unit") if key in values}
            print(PREFIX + json.dumps({"kind": kind, **selected}), file=sys.stderr, flush=True)
        with observe_progress(progress):
            retained = retained_outcome(store, manifest)
            if retained is not None:
                return retained
            results = executor.run_batch(requests, context=context)
        prepared = {
                "job_id": manifest["job_id"],
                "run_id": manifest.get("run_id"),
                "task_id": manifest.get("task_id"),
                "instance_id": manifest.get("instance_id"),
                "status": "succeeded",
                "results": [result_to_transport(result) for result in results],
                "trace": _combined_trace(results),
                "resource_usage": _combined_usage(results),
                "error": None,
                "created_at": created_at,
                "finished_at": _now(),
            }
        return commit_prepared(store, manifest, prepared)
    except Exception as exc:
        if manifest is not None:
            _write_failure(store, manifest, created_at, exc)
        traceback.print_exc()
        return 1


def _write_failure(store, manifest: dict[str, Any], created_at: str, exc: Exception) -> None:
    from mini_bang.framework.policy import RequestRejectedError
    # A builtin ValueError/TypeError may come from organizer payload setup,
    # numerical code or serialization. Only explicit request validation is
    # positive evidence against the participant; every unknown stays infra.
    genuine = isinstance(exc, RequestRejectedError)
    from mini_bang.worker.storage_retry import commit_prepared
    try:
        prepared = {
                "job_id": manifest.get("job_id"),
                "run_id": manifest.get("run_id"),
                "task_id": manifest.get("task_id"),
                "instance_id": manifest.get("instance_id"),
                "status": "failed",
                "results": [],
                "trace": [],
                "resource_usage": {},
                "error": {"type": type(exc).__name__, "message": str(exc),
                          "category": "request_rejected" if genuine else "infrastructure_failed",
                          **({"reason": "observation_worker_failure"} if not genuine else {})},
                "created_at": created_at,
                "finished_at": _now(),
            }
        commit_prepared(store, manifest, prepared)
    except Exception:
        traceback.print_exc()


def _runtime_context_from_dict(data: dict[str, Any]) -> RuntimeContext:
    return RuntimeContext.from_dict(data)


def _combined_trace(results) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for result in results:
        trace.extend(result.trace)
    return trace


def _combined_usage(results) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for result in results:
        for key, value in result.resource_usage.items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    return usage


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
