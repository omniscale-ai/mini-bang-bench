from __future__ import annotations

import argparse
import dataclasses
import http.server
import os
import sys

from mini_bang.backends import create_backend_from_env
from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.client.cli.official import OfficialRunServer
from mini_bang.client.cli.scoring_environment import private_judge_environment
from mini_bang.execution.result_store import result_store_from_env
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.framework.runtime import RunManager
from mini_bang.framework.scoring_jobs import K8sScoringJobBackend
from mini_bang.tasks.loader import TaskLoader


def build_parser() -> argparse.ArgumentParser:
    """Command line of the official runner used by the k8s, pool and compose stacks."""
    parser = argparse.ArgumentParser(prog="mini-bang-official-runner")
    parser.add_argument("--host", default=os.getenv("MINI_BANG_RUNNER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_runner_port())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    admin_token = os.getenv("MINI_BANG_RUNNER_ADMIN_TOKEN")
    if not admin_token:
        print("MINI_BANG_RUNNER_ADMIN_TOKEN is required", file=sys.stderr)
        return 2

    result_store = result_store_from_env()
    backend = create_backend_from_env()
    scoring_backend = None
    if isinstance(backend, K8sBackend):
        scoring_backend = K8sScoringJobBackend(
            config=_scoring_config_from_backend(backend),
            result_store=result_store,
            use_k8s_for_hidden_simulations=_env_bool(
                "MINI_BANG_K8S_SCORING_HIDDEN_SIMULATIONS_USE_K8S",
                True,
            ),
        )
    manager = RunManager(
        loader=TaskLoader(),
        backend=backend,
        artifact_store=BenchmarkArtifactStore(),
        scoring_job_backend=scoring_backend,
        scoring_runtime_context=_scoring_runtime_context_from_env(),
    )
    server = OfficialRunServer(
        run_manager=manager,
        mode="multi_session_official",
        admin_token=admin_token,
        artifact_store=BenchmarkArtifactStore(),
        scoring_job_backend=scoring_backend,
        result_store=result_store,
        auto_score=True,
    )
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), server.handler_class())
    print(f"mini-bang official runner listening on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return 0


def _scoring_config_from_backend(backend: K8sBackend) -> K8sBackendConfig:
    return dataclasses.replace(backend.config)


def _scoring_runtime_context_from_env() -> RuntimeContext:
    """Explicit private scoring authority; never put it in the public/simulation context."""
    context = RuntimeContext.for_official()
    return dataclasses.replace(context, security={**context.security, "official_mode": True},
        env=private_judge_environment())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _runner_port() -> int:
    raw = os.getenv("MINI_BANG_RUNNER_LISTEN_PORT") or os.getenv("MINI_BANG_RUNNER_PORT") or "8765"
    try:
        return int(raw)
    except ValueError:
        return 8765


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
