from __future__ import annotations

import argparse
import os
import http.server
import subprocess
import threading
import time
from typing import Any

from mini_bang.client.cli.errors import CliError, EXIT_INTERNAL, EXIT_RUN_FAILED
from mini_bang.client.cli.docker_official import (
    DockerNetwork,
    build_agent_docker_command,
    require_insecure_host_gateway,
)
from mini_bang.client.cli.io import close_backend, find_free_port, read_json, utc_now
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.client.cli.records import save_run_outputs
from mini_bang.client.cli.runtime import (
    build_backend,
    build_local_agent_image,
    parse_submission_from_stdout,
    submission_from_json,
    tail_text,
)
from mini_bang.client.cli.scoring import build_official_scoring_services
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring import score_submission


def run_docker_official_submission(
    args: argparse.Namespace,
    payload_storage: Any,
    *,
    allow_insecure_artifacts: bool = False,
) -> dict[str, Any]:
    from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
    from mini_bang.framework.runtime import RunManager
    from mini_bang.tasks.loader import TaskLoader

    host_gateway_enabled = require_insecure_host_gateway(args)
    backend = build_backend(
        kind="local",
        engine="sequential",
        workers=None,
        reuse_strategy="none",
        payload_storage=payload_storage,
    )
    loader = TaskLoader()
    manager = RunManager(loader=loader, backend=backend, artifact_store=BenchmarkArtifactStore())
    session = manager.prepare(args.task, instance_id=args.instance)
    if session.environment is None:
        raise CliError("failed to prepare official run environment", EXIT_INTERNAL)

    server = OfficialRunServer(session)
    runner_token = server.session_token(session.run_id)
    port = find_free_port()
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), server.handler_class())
    thread = threading.Thread(target=httpd.serve_forever, name="mini-bang-official-api", daemon=True)
    thread.start()
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    exit_code = 0
    try:
        agent_image = args.agent_image or build_local_agent_image(args)
        runner_url = f"http://host.docker.internal:{port}"
        timeout = args.timeout or args.max_wall_clock or 600
        with DockerNetwork(session.run_id) as network_name:
            docker_cmd = build_agent_docker_command(
                args=args,
                agent_image=agent_image,
                network_name=network_name,
                runner_url=runner_url,
                runner_token=runner_token,
                task_id=session.task_id,
                instance_id=session.instance_id,
                run_id=session.run_id,
            )
            completed = subprocess.run(
                docker_cmd,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        exit_code = 124
        session.trace.append(
            {
                "event": "agent_container_timeout",
                "timestamp": time.time(),
                "timeout_seconds": args.timeout,
            }
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        close_backend(backend)

    agent_seconds = time.perf_counter() - started
    submission = server.submission or parse_submission_from_stdout(stdout)
    if submission is None:
        stdout_tail = tail_text(stdout)
        stderr_tail = tail_text(stderr)
        detail = f"agent exit code {exit_code}"
        if stdout_tail:
            detail += f"; stdout tail: {stdout_tail}"
        if stderr_tail:
            detail += f"; stderr tail: {stderr_tail}"
        raise CliError(
            "agent container did not POST a submission and stdout did not contain "
            f"submission JSON ({detail})",
            EXIT_RUN_FAILED,
        )
    session.timing["agent_seconds"] = agent_seconds
    session.timing["wall_seconds"] = agent_seconds
    session.resource_usage["agent_exit_code"] = exit_code
    session.resource_usage["agent_stdout_bytes"] = len(stdout.encode("utf-8", errors="replace"))
    session.resource_usage["agent_stderr_bytes"] = len(stderr.encode("utf-8", errors="replace"))
    session.trace.append(
        {
            "event": "agent_container_completed",
            "timestamp": time.time(),
            "exit_code": exit_code,
        }
    )

    provisional = RunRecord.from_parts(
        run_id=session.run_id,
        task_id=session.task_id,
        instance_id=session.instance_id,
        submission=submission,
        trace=session.trace,
        timing=session.timing,
        resource_usage=session.resource_usage,
        backend_metadata=session.backend_metadata,
        versions={
            "task_version": session.task_spec.version,
            "instance_version": session.instance_spec.version,
        },
    )
    scorer = session.task_spec.create_scorer()
    gold_store = BenchmarkArtifactStore().load_gold(session.instance_spec)
    scoring_trace: list[dict[str, Any]] = []
    scoring_resource_usage: dict[str, Any] = {}
    scoring_services = build_official_scoring_services(
        backend=backend,
        session=session,
        trace=scoring_trace,
        resource_usage=scoring_resource_usage,
    )
    validation = score_submission(
        scorer,
        task_spec=session.task_spec,
        instance_spec=session.instance_spec,
        submission=submission,
        run_record=provisional,
        gold_store=gold_store,
        scoring_services=scoring_services,
    )
    record = provisional.with_score(validation)
    record.scoring_trace = scoring_trace
    record.scoring_resource_usage = scoring_resource_usage
    record.backend_metadata.setdefault("cli", {})
    record.backend_metadata["cli"].update(
        {
            "mode": "local-official",
            "agent": args.agent or args.agent_file or args.agent_image,
            "agent_image": agent_image,
            "agent_command": args.agent_command,
            "created_at": utc_now(),
            "security": {
                "agent_isolated": not host_gateway_enabled,
                "gold_mounted_into_agent": False,
                "runner_storage_credentials_passed_to_agent": False,
                "raw_file_refs_exposed_to_agent": False,
                "dedicated_docker_network": True,
                "host_gateway_enabled": host_gateway_enabled,
                "official_scoreable": not host_gateway_enabled,
            },
        }
    )
    run_dir = save_run_outputs(
        record,
        args.out,
        mode="local-official",
        allow_pickle_payloads=allow_insecure_artifacts,
    )
    return {
        "record": read_json(run_dir / "run_record.json"),
        "run_dir": run_dir,
        "success": validation.success and exit_code == 0,
    }
