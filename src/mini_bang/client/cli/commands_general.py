from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
import time
from pathlib import Path
from typing import Any

from mini_bang.client.paths import CACHE_ROOT, init_directories
from mini_bang.client.cli.config import dump_simple_yaml, load_config, nested_get, preset_config
from mini_bang.client.cli.errors import CheckResult, CliError, EXIT_CONFIG, EXIT_SUCCESS, EXIT_USAGE
from mini_bang.client.cli.formatting import render_checks_text, render_table, render_task_show_text, task_summary
from mini_bang.client.cli.io import emit, normalize_table_format, parse_duration_seconds
from mini_bang.client.cli.runtime import build_payload_storage, prepare_public_view
from mini_bang.client.cli.security import (
    add_compatibility_findings,
    check_command,
    check_docker,
    has_s3_credential_hint,
    project_config_to_compatibility,
)
from mini_bang.config_validation import validate_compatibility


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config()
    checks: list[CheckResult] = []
    warnings = 0
    errors = 0

    def add(status: str, message: str, **details: Any) -> None:
        nonlocal warnings, errors
        checks.append(CheckResult(status=status, message=message, details=details))
        if status == "WARN":
            warnings += 1
        if status == "ERR":
            errors += 1

    if sys.version_info >= (3, 11):
        add("OK", f"Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        add("ERR", f"Python {sys.version_info.major}.{sys.version_info.minor}; need >= 3.11")

    try:
        import mini_bang  # noqa: F401

        add("OK", "mini_bang imports")
    except Exception as exc:
        add("ERR", "mini_bang import failed", error=repr(exc))
    for dependency in ("numpy", "pydantic"):
        try:
            importlib.import_module(dependency)
            add("OK", f"{dependency} imports")
        except Exception as exc:
            add("ERR", f"{dependency} import failed", error=repr(exc))

    try:
        from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore, DEV_SPLIT, instance_split
        from mini_bang.tasks.loader import TaskLoader

        loader = TaskLoader()
        specs = list(loader.specs())
        add("OK", f"{len(specs)} tasks discovered", task_ids=[spec.task_id for spec in specs])
        store = BenchmarkArtifactStore()
        for spec in specs:
            try:
                instance = store.load_instance(spec, spec.default_instance_id)
                if instance.gold_path is not None and instance.gold_path.is_file():
                    add(
                        "OK",
                        f"default task {spec.task_id} has instance {instance.instance_id}",
                        gold=str(instance.gold_path),
                    )
                elif instance_split(instance) != DEV_SPLIT:
                    add("OK", f"default task {spec.task_id} has test instance {instance.instance_id}; "
                        "the scoring server holds its gold")
                else:
                    add("ERR", f"default task {spec.task_id} is missing gold")
            except Exception as exc:
                add("ERR", f"default task {spec.task_id} instance check failed", error=repr(exc))
    except Exception as exc:
        add("ERR", "TaskLoader failed", error=repr(exc))

    backend = nested_get(config, "backend.kind", "local")
    engine = nested_get(config, "backend.engine", "sequential")
    payload_mode = nested_get(config, "payload.mode", "inline")
    add("OK", f"backend {backend}")
    add("OK", f"execution engine {engine}")
    try:
        payload_storage = build_payload_storage(
            mode=str(payload_mode),
            payload_dir=nested_get(config, "payload.filesystem.root"),
            inline_threshold=nested_get(config, "payload.inline_threshold_bytes"),
            s3_bucket=nested_get(config, "payload.s3.bucket"),
            s3_prefix=nested_get(config, "payload.s3.prefix"),
            s3_endpoint_url=nested_get(config, "payload.s3.endpoint_url"),
        )
        add("OK", f"payload storage {payload_storage.mode}", config=payload_storage.describe())
    except Exception as exc:
        add("ERR", "payload storage config failed", error=repr(exc))

    if str(payload_mode).lower() == "s3":
        bucket = nested_get(config, "payload.s3.bucket")
        if bucket:
            add("OK", f"S3 bucket configured: {bucket}")
        else:
            add("ERR", "S3 payload mode requires payload.s3.bucket")
        if has_s3_credential_hint():
            add("OK", "S3 credentials discoverable from environment")
        else:
            add("WARN", "S3 credentials not found in environment; boto3 may still use instance credentials")

    validation = validate_compatibility(project_config_to_compatibility(config))
    if validation:
        add_compatibility_findings(add, validation)
    else:
        add("OK", "mode/backend/agent/payload compatibility")

    mode = str(config.get("mode", "dev-local")).lower()
    if mode in {"compose-local", "local-official"}:
        check_docker(add)
    if mode in {"k8s-official", "cluster-official"}:
        check_command(add, "kubectl")
        check_command(add, "helm")

    payload = {
        "checks": [dataclasses.asdict(item) for item in checks],
        "summary": {"warnings": warnings, "errors": errors, "strict": args.strict},
    }
    emit(payload, args.format, lambda: render_checks_text(checks))
    if errors:
        return EXIT_CONFIG
    if warnings and args.strict:
        return EXIT_CONFIG
    return EXIT_SUCCESS

def cmd_init(args: argparse.Namespace) -> int:
    config_path = Path("mini-bang.yaml")
    if config_path.exists() and not args.force:
        raise CliError("mini-bang.yaml already exists; use --force to overwrite", EXIT_USAGE)

    config = preset_config(args.preset)
    config_path.write_text(dump_simple_yaml(config), encoding="utf-8")
    dirs = init_directories(config)
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "config_path": str(config_path),
        "preset": args.preset,
        "created_dirs": [str(path) for path in dirs],
    }
    emit(payload, args.format, lambda: "\n".join(
        [f"Created {config_path}", *[f"Created {path}/" for path in dirs]]
    ))
    return EXIT_SUCCESS

def cmd_support_matrix(args: argparse.Namespace) -> int:
    path = Path("docs/support_matrix.md")
    if not path.is_file():
        raise CliError("docs/support_matrix.md is missing", EXIT_CONFIG)
    rows = _parse_support_matrix(path.read_text(encoding="utf-8"))
    emit(
        {"claims": rows},
        args.format,
        lambda: path.read_text(encoding="utf-8"),
    )
    return EXIT_SUCCESS

def cmd_agent_client(args: argparse.Namespace) -> int:
    from mini_bang.client.deployment.public_agent_sdk import write_agent_client

    output = Path(args.output)
    paths = write_agent_client(output)
    usage = (f"pip install {output}   # or PYTHONPATH={output}\n"
             "python -m mini_bang.agent_runtime --agent my_package.agents:MyAgent")
    emit({"output": str(output), "files": paths, "usage": usage}, args.format,
         lambda: f"Agent client written to {output} (no simulators, tasks or gold):\n{usage}")
    return EXIT_SUCCESS

def cmd_task_list(args: argparse.Namespace) -> int:
    from mini_bang.tasks.loader import TaskLoader

    rows = []
    for spec in TaskLoader().specs():
        rows.append(task_summary(spec))
    rows.sort(key=lambda item: item["task_id"])
    emit(
        {"tasks": rows},
        normalize_table_format(args.format),
        lambda: render_table(rows, ["task_id", "level", "name", "default_instance_id"]),
    )
    return EXIT_SUCCESS


def _parse_support_matrix(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| Claim"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 3:
            rows.append({"claim": cells[0], "status": cells[1], "notes": cells[2]})
    return rows

def cmd_task_show(args: argparse.Namespace) -> int:
    from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
    from mini_bang.tasks.loader import TaskLoader

    loader = TaskLoader()
    try:
        spec = loader.get_spec(args.task_id)
    except KeyError as exc:
        raise CliError(str(exc), EXIT_USAGE) from exc
    store = BenchmarkArtifactStore()
    instance_id = args.instance or spec.default_instance_id
    instance_ids = store.list_instance_ids(spec)
    instance = store.load_instance(spec, instance_id)
    capabilities, instructions = prepare_public_view(spec.task_id, instance_id)
    gold_exists = instance.gold_path is not None and instance.gold_path.is_file()
    data = {
        "task_id": spec.task_id,
        "description": instance.description or spec.description,
        "instructions": instructions,
        "available_instances": list(instance_ids),
        "default_instance_id": spec.default_instance_id,
        "selected_instance_id": instance.instance_id,
        "public_api_capabilities": capabilities,
        "budget_limits": {
            "max_generate_calls": capabilities.get("max_generate_calls"),
            "remaining_generate_calls": capabilities.get("remaining_generate_calls"),
        },
        "allowed_extras": capabilities.get("allowed_extras", []),
        "scorer_class": spec.scorer_class_path,
        "policy_class": spec.policy_class_path,
        "gold_artifact_exists": gold_exists,
    }
    emit(data, args.format, lambda: render_task_show_text(data))
    return EXIT_SUCCESS

def cmd_cache_clear(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir or nested_get(load_config(), "cache.root", f"./{CACHE_ROOT.as_posix()}"))
    if not cache_dir.exists():
        emit({"cache_dir": str(cache_dir), "removed": 0}, args.format, lambda: "Cache directory does not exist")
        return EXIT_SUCCESS
    cutoff = None
    if args.older_than:
        cutoff = time.time() - parse_duration_seconds(args.older_than)
    if not args.all and cutoff is None:
        raise CliError("cache clear requires --all or --older-than", EXIT_USAGE)
    removed = 0
    for path in sorted(cache_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path == cache_dir:
            continue
        if cutoff is not None and path.stat().st_mtime > cutoff:
            continue
        if path.is_dir():
            try:
                path.rmdir()
                removed += 1
            except OSError:
                pass
        else:
            path.unlink()
            removed += 1
    emit({"cache_dir": str(cache_dir), "removed": removed}, args.format, lambda: f"Removed {removed} cache entries")
    return EXIT_SUCCESS
