from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mini_bang.client.paths import PAYLOADS_ROOT
from mini_bang.client.cli.config import load_config, nested_get
from mini_bang.client.cli.errors import CliError, EXIT_CONFIG, EXIT_USAGE
from mini_bang.client.cli.io import close_backend, get_cache_dir, safe_name
from mini_bang.framework.task import TaskSubmission
from mini_bang.importing import import_from_path


def build_payload_storage(
    *,
    mode: str,
    payload_dir: str | None = None,
    inline_threshold: int | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    s3_endpoint_url: str | None = None,
):
    from mini_bang.execution.payloads import PayloadStorageConfig
    from mini_bang.execution.payload_stores.filesystem import FileSystemPayloadStore
    from mini_bang.execution.payload_stores.memory import InMemoryPayloadStore
    from mini_bang.execution.payload_stores.s3 import S3PayloadStore

    normalized = mode.lower()
    if normalized in {"inline", "none", "off"}:
        return PayloadStorageConfig.inline()
    if normalized == "memory":
        return PayloadStorageConfig.reference(InMemoryPayloadStore(), mode="memory")
    if normalized == "filesystem":
        return PayloadStorageConfig.reference(
            FileSystemPayloadStore(payload_dir or f"./{PAYLOADS_ROOT.as_posix()}"),
            mode="filesystem",
        )
    if normalized in {"s3", "minio"}:
        bucket = s3_bucket or os.getenv("MINI_BANG_PAYLOAD_S3_BUCKET")
        if normalized == "minio":
            bucket = bucket or "mini-bang-payloads"
            s3_endpoint_url = s3_endpoint_url or "http://127.0.0.1:9000"
        return PayloadStorageConfig.reference(
            S3PayloadStore(
                bucket=bucket or "",
                prefix=s3_prefix or os.getenv("MINI_BANG_PAYLOAD_S3_PREFIX", ""),
                endpoint_url=s3_endpoint_url or os.getenv("MINI_BANG_PAYLOAD_S3_ENDPOINT_URL"),
            ),
            mode="s3",
        )
    if normalized == "auto":
        if s3_bucket:
            store = S3PayloadStore(
                bucket=s3_bucket,
                prefix=s3_prefix or "",
                endpoint_url=s3_endpoint_url,
            )
        else:
            store = FileSystemPayloadStore(payload_dir or f"./{PAYLOADS_ROOT.as_posix()}")
        return PayloadStorageConfig(
            mode="auto",
            store=store,
            inline_threshold_bytes=inline_threshold,
        )
    raise CliError(f"unsupported payload mode {mode}", EXIT_USAGE)

def build_backend(
    *,
    kind: str,
    engine: str,
    workers: int | None,
    reuse_strategy: str,
    payload_storage: Any,
):
    if kind == "mcp":
        from mini_bang.backends.mcp import MCPBackend

        return MCPBackend.from_env()
    if kind != "local":
        raise CliError(f"unsupported backend {kind}", EXIT_USAGE)
    from mini_bang.backends.local import LocalBackend
    from mini_bang.execution.engines import ProcessPoolEngine, SequentialEngine
    from mini_bang.execution.executor import SimulationExecutor
    from mini_bang.execution.runtime import RuntimeContext

    if engine == "process_pool":
        execution_engine = ProcessPoolEngine(max_workers=workers, reuse_strategy=reuse_strategy)
    elif engine == "sequential":
        execution_engine = SequentialEngine(reuse_strategy=reuse_strategy)
    else:
        raise CliError(f"unsupported execution engine {engine}", EXIT_USAGE)
    runtime_context = RuntimeContext.from_env()
    payload_context = _payload_context_from_storage(payload_storage)
    if payload_context:
        runtime_context = dataclasses.replace(runtime_context, payload=payload_context)
    return LocalBackend(
        executor=SimulationExecutor(
            engine=execution_engine,
            payload_storage=payload_storage,
            context=runtime_context,
        )
    )

def _payload_context_from_storage(payload_storage: Any) -> dict[str, Any]:
    store = getattr(payload_storage, "store", None)
    if store is None:
        return {}
    description = store.describe()
    kind = description.get("kind")
    if not kind:
        return {}
    return dict(description)

def build_official_payload_storage(args: argparse.Namespace):
    if args.payload_store == "filesystem":
        config = load_config()
        return build_payload_storage(
            mode="filesystem",
            payload_dir=nested_get(config, "payload.filesystem.root", f"./{PAYLOADS_ROOT.as_posix()}"),
        )
    return build_payload_storage(
        mode=args.payload_store,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint_url=args.s3_endpoint_url,
    )

def load_agent(agent_path: str | None, agent_file: str | None) -> Any:
    if agent_path:
        return instantiate_agent_object(import_from_path(agent_path))
    if not agent_file:
        raise CliError("missing agent", EXIT_USAGE)
    file_path, _, attr = agent_file.partition(":")
    module_path = Path(file_path)
    if not module_path.is_file():
        raise CliError(f"agent file not found: {module_path}", EXIT_USAGE)
    module_name = f"_mini_bang_agent_{hashlib.sha256(str(module_path.resolve()).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CliError(f"cannot import agent file: {module_path}", EXIT_USAGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if attr:
        return instantiate_agent_object(getattr(module, attr))
    for name in ("agent", "Agent", "create_agent", "solve"):
        if hasattr(module, name):
            obj = getattr(module, name)
            if name == "create_agent" and callable(obj):
                obj = obj()
            return instantiate_agent_object(obj)
    raise CliError(
        "agent file must define agent, Agent, create_agent(), solve(), or use path.py:Name",
        EXIT_USAGE,
    )

def instantiate_agent_object(obj: Any) -> Any:
    if isinstance(obj, type):
        return obj()
    return obj

def prepare_public_view(task_id: str, instance_id: str) -> tuple[dict[str, Any], str]:
    """Capabilities and instructions that GET /v1/run shows an agent for this instance."""
    from mini_bang.execution.payloads import PayloadStorageConfig
    from mini_bang.framework.runtime import RunManager
    from mini_bang.tasks.loader import TaskLoader

    backend = build_backend(
        kind="local",
        engine="sequential",
        workers=None,
        reuse_strategy="none",
        payload_storage=PayloadStorageConfig.inline(),
    )
    try:
        session = RunManager(loader=TaskLoader(), backend=backend).prepare(
            task_id,
            instance_id=instance_id,
        )
        environment = session.environment
        return session.capabilities(), environment.api.instructions() if environment else ""
    finally:
        close_backend(backend)

def build_local_agent_image(args: argparse.Namespace) -> str:
    tag = f"mini-bang-agent-local:{safe_name(args.task)}-{int(time.time())}"
    cache_dir = get_cache_dir() / "agent-build"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = cache_dir / "Dockerfile"
    dockerfile.write_text(
        """FROM python:3.12-slim
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .
ENV PYTHONPATH=/workspace/src
CMD ["python", "-m", "mini_bang.agent_runtime"]
""",
        encoding="utf-8",
    )
    command = ["docker", "build", "-f", str(dockerfile), "-t", tag, "."]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        raise CliError(
            "failed to build local official agent image: "
            f"{tail_text(process.stderr or process.stdout)}",
            EXIT_CONFIG,
        )
    return tag

def container_agent_file(agent_file: str) -> str:
    file_path, sep, attr = agent_file.partition(":")
    path = Path(file_path)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            return agent_file
    else:
        relative = path
    container_path = f"/workspace/{relative.as_posix()}"
    return f"{container_path}:{attr}" if sep else container_path

def parse_submission_from_stdout(stdout: str) -> TaskSubmission | None:
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return submission_from_json(value)
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return submission_from_json(value) if isinstance(value, dict) else TaskSubmission(answer=value)

def tail_text(value: str, *, max_chars: int = 1000) -> str:
    cleaned = value.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return "..." + cleaned[-max_chars:]

def submission_from_json(value: dict[str, Any]) -> TaskSubmission:
    if "answer" in value or "artifacts" in value:
        return TaskSubmission(answer=value.get("answer"), artifacts=value.get("artifacts"))
    if "submission" in value and isinstance(value["submission"], dict):
        return submission_from_json(value["submission"])
    return TaskSubmission(answer=value)
