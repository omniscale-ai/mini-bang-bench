from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mini_bang.client.paths import RUNS_ROOT
from mini_bang.client.cli.config import load_config, nested_get
from mini_bang.client.cli.errors import DEV_LOCAL_WARNING, CliError, EXIT_RUN_FAILED, EXIT_SECURITY, EXIT_SUCCESS, EXIT_USAGE
from mini_bang.client.cli.formatting import render_result_show_text, render_run_summary_text, render_table, score_label
from mini_bang.client.cli.io import emit, normalize_table_format, read_json, utc_now, looks_like_raw_file_ref
from mini_bang.client.cli.docker_official import require_insecure_host_gateway
from mini_bang.client.cli.official import run_docker_official_submission
from mini_bang.client.cli.records import (
    find_run_dir,
    materialize_artifact_entry,
    read_manifest,
    run_output_summary,
    write_run_zip,
)
from mini_bang.client.cli.runtime import build_backend, build_official_payload_storage, build_payload_storage, load_agent
from mini_bang.client.cli.security import check_docker_runtime_or_raise, command_compatibility, validate_or_raise


def cmd_run_local(args: argparse.Namespace) -> int:
    config = load_config()
    validate_or_raise(
        command_compatibility(
            mode="dev-local",
            agent_placement="in-process",
            backend=args.backend,
            engine=args.engine if args.backend == "local" else None,
            payload_store=args.payload_mode,
            config=config,
            command="run-local",
        )
    )
    print(DEV_LOCAL_WARNING, file=sys.stderr)
    backend = build_backend(
        kind=args.backend,
        engine=args.engine,
        workers=args.workers,
        reuse_strategy=args.reuse,
        payload_storage=build_payload_storage(
            mode=args.payload_mode,
            payload_dir=args.payload_dir,
            inline_threshold=args.payload_inline_threshold,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_endpoint_url=args.s3_endpoint_url,
        ),
    )
    agent = load_agent(args.agent, args.agent_file)
    from mini_bang.framework.runtime import RunManager
    from mini_bang.tasks.loader import TaskLoader

    manager = RunManager(loader=TaskLoader(), backend=backend)
    result = manager.run(agent, args.task, instance_id=args.instance)
    record = result.record
    record.backend_metadata.setdefault("cli", {})
    record.backend_metadata["cli"].update(
        {
            "mode": "dev-local",
            "agent": args.agent or args.agent_file,
            "created_at": utc_now(),
            "security": {"trusted_dev_mode": True},
        }
    )
    run_dir = save_run_outputs(record, args.out, mode="dev-local")
    output = run_output_summary(record.to_dict(), run_dir)
    emit(output, args.format, lambda: render_run_summary_text(output, record.score_summary))
    close_backend(backend)
    return EXIT_SUCCESS if result.validation.success else EXIT_RUN_FAILED

def cmd_run_docker(args: argparse.Namespace) -> int:
    if not args.agent_image and not (args.agent or args.agent_file):
        raise CliError(
            "run-docker requires --agent-image, or --agent/--agent-file for auto-built local image",
            EXIT_USAGE,
        )
    if args.agent_command and (args.agent or args.agent_file):
        raise CliError("--agent-command cannot be combined with --agent/--agent-file", EXIT_USAGE)
    host_gateway_enabled = require_insecure_host_gateway(args)
    if host_gateway_enabled:
        print(
            "WARNING: --allow-host-gateway exposes host networking to the agent container; "
            "this run is local-debug only and not official-scoreable.",
            file=sys.stderr,
        )
    config = load_config()
    validate_or_raise(
        command_compatibility(
            mode="local-official",
            agent_placement="docker",
            backend="local",
            engine="sequential",
            payload_store=args.payload_store,
            config=config,
            command="run-docker",
            allow_insecure=args.i_know_this_is_insecure,
        )
    )
    check_docker_runtime_or_raise()

    payload_storage = build_official_payload_storage(args)
    result = run_docker_official_submission(
        args,
        payload_storage,
        allow_insecure_artifacts=args.i_know_this_is_insecure,
    )
    output = run_output_summary(result["record"], result["run_dir"])
    emit(output, args.format, lambda: render_run_summary_text(output, result["record"].get("score_summary", {})))
    return EXIT_SUCCESS if result["success"] else EXIT_RUN_FAILED

def cmd_results_list(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir or nested_get(load_config(), "runs.root", f"./{RUNS_ROOT.as_posix()}"))
    rows = []
    for run_dir in sorted(p for p in runs_dir.glob("*") if p.is_dir()):
        record_path = run_dir / "run_record.json"
        if not record_path.exists():
            continue
        record = read_json(record_path)
        cli_meta = nested_get(record, "backend_metadata.cli", {}) or {}
        rows.append(
            {
                "run_name": run_dir.name,
                "run_id": record.get("run_id", run_dir.name),
                "task_id": record.get("task_id"),
                "instance_id": record.get("instance_id"),
                "agent": cli_meta.get("agent", ""),
                "score": score_label(record.get("score_summary", {})),
                "success": nested_get(record, "score_summary.success"),
                "created_at": cli_meta.get("created_at", ""),
                "mode": cli_meta.get("mode", ""),
            }
        )
    emit(
        {"runs": rows},
        normalize_table_format(args.format),
        lambda: render_table(
            rows,
            ["run_name", "task_id", "instance_id", "agent", "score", "success", "created_at", "mode"],
        ),
    )
    return EXIT_SUCCESS

def cmd_results_show(args: argparse.Namespace) -> int:
    run_dir = find_run_dir(args.run_id)
    record_path = run_dir / "run_record.json"
    if not record_path.exists():
        raise CliError(f"run_record.json not found for run {args.run_id}", EXIT_USAGE)
    record = read_json(record_path)
    data = {
        "run_name": run_dir.name,
        "run_id": record.get("run_id"),
        "task_id": record.get("task_id"),
        "instance_id": record.get("instance_id"),
        "score_summary": record.get("score_summary", {}),
        "timing": record.get("timing", {}),
        "resource_usage": record.get("resource_usage", {}),
        "backend_metadata": record.get("backend_metadata", {}),
        "artifact_refs": read_manifest(run_dir),
        "run_record_path": str(record_path),
    }
    if args.trace:
        data["trace"] = record.get("trace", [])
    if args.artifacts:
        data["submission_artifacts"] = nested_get(record, "submission.artifacts", {})
    emit(data, args.format, lambda: render_result_show_text(data, include_trace=args.trace))
    return EXIT_SUCCESS

def cmd_results_export(args: argparse.Namespace) -> int:
    run_dir = find_run_dir(args.run_id)
    record_path = run_dir / "run_record.json"
    if not record_path.exists():
        raise CliError(f"run_record.json not found for run {args.run_id}", EXIT_USAGE)
    if args.format == "json":
        print(json.dumps(read_json(record_path), ensure_ascii=False, indent=2))
        return EXIT_SUCCESS
    if args.format == "jsonl":
        trace_path = run_dir / "trace.jsonl"
        if trace_path.exists():
            print(trace_path.read_text(encoding="utf-8"), end="")
        else:
            record = read_json(record_path)
            for event in record.get("trace", []):
                print(json.dumps(event, sort_keys=True))
        return EXIT_SUCCESS
    out_path = Path(args.out or run_dir / f"{args.run_id}.zip")
    write_run_zip(run_dir, out_path, include_artifacts=args.include_artifacts)
    print(str(out_path))
    return EXIT_SUCCESS

def cmd_artifact_get(args: argparse.Namespace) -> int:
    if looks_like_raw_file_ref(args.artifact_id):
        raise CliError("artifact_id must be a registered artifact id, not a raw file path", EXIT_SECURITY)
    run_dir = find_run_dir(args.run)
    manifest = read_manifest(run_dir)
    entry = next((item for item in manifest if item.get("artifact_id") == args.artifact_id), None)
    if entry is None:
        raise CliError(f"artifact {args.artifact_id} is not registered for run {args.run}", EXIT_USAGE)
    if args.format in {"metadata", "text", "json"} and not args.as_path:
        print(json.dumps(entry, indent=2, sort_keys=True))
        return EXIT_SUCCESS
    output_path = materialize_artifact_entry(entry, args.out)
    if args.as_path or args.format == "download":
        print(str(output_path))
    return EXIT_SUCCESS
