from __future__ import annotations

import argparse

from mini_bang.client.cli.commands_deployment import cmd_compose_down, cmd_compose_up, cmd_k8s_install, cmd_k8s_render
from mini_bang.client.cli.extensions import load_cli_extensions
from mini_bang.client.cli.commands_general import (
    cmd_agent_client,
    cmd_cache_clear,
    cmd_doctor,
    cmd_init,
    cmd_support_matrix,
    cmd_task_list,
    cmd_task_show,
)
from mini_bang.client.cli.commands_run import cmd_artifact_get, cmd_results_export, cmd_results_list, cmd_results_show, cmd_run_docker, cmd_run_local
from mini_bang.client.cli.commands_storage import cmd_storage_test
from mini_bang.client.cli.commands_submit import cmd_server_submit
from mini_bang.client.cli.server_submission import SERVER_TOKEN_ENV, SERVER_URL_ENV


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-bang",
        description="CLI for running mini-bang benchmark tasks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local benchmark environment")
    doctor.add_argument("--format", choices=("text", "json"), default="text")
    doctor.add_argument("--strict", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    support = subparsers.add_parser("support-matrix", help="Show supported infrastructure modes")
    support.add_argument("--format", choices=("text", "json"), default="text")
    support.set_defaults(func=cmd_support_matrix)

    init = subparsers.add_parser("init", help="Create mini-bang.yaml and local dirs")
    init.add_argument(
        "--preset",
        choices=("dev-local", "compose-local", "k8s"),
        default="dev-local",
    )
    init.add_argument("--force", action="store_true")
    init.add_argument("--format", choices=("text", "json"), default="text")
    init.set_defaults(func=cmd_init)

    agent_client = subparsers.add_parser(
        "agent-client", help="Write the stdlib-only agent client (no simulators, tasks or gold)")
    agent_client.add_argument("--output", required=True, help="Directory to write; pip-installable")
    agent_client.add_argument("--format", choices=("text", "json"), default="text")
    agent_client.set_defaults(func=cmd_agent_client)

    task = subparsers.add_parser("task", help="Inspect benchmark tasks")
    task_sub = task.add_subparsers(dest="task_command", required=True)
    task_list = task_sub.add_parser("list", help="List discovered tasks")
    task_list.add_argument("--format", choices=("table", "text", "json"), default="table")
    task_list.set_defaults(func=cmd_task_list)
    task_show = task_sub.add_parser("show", help="Show task details")
    task_show.add_argument("task_id")
    task_show.add_argument("--instance")
    task_show.add_argument("--format", choices=("text", "json"), default="text")
    task_show.set_defaults(func=cmd_task_show)

    run_local = subparsers.add_parser("run-local", help="Run a trusted dev-local eval")
    run_local.add_argument("--task", required=True)
    run_local.add_argument("--instance")
    agent_group = run_local.add_mutually_exclusive_group(required=True)
    agent_group.add_argument("--agent")
    agent_group.add_argument("--agent-file")
    run_local.add_argument("--backend", choices=("local", "mcp"), default="local")
    run_local.add_argument(
        "--engine",
        choices=("sequential", "process_pool"),
        default="sequential",
    )
    run_local.add_argument("--workers", type=int)
    run_local.add_argument("--reuse", choices=("none", "grouped"), default="none")
    run_local.add_argument(
        "--payload-mode",
        choices=("inline", "memory", "filesystem", "s3", "auto"),
        default="inline",
    )
    run_local.add_argument("--payload-dir")
    run_local.add_argument("--payload-inline-threshold", type=int)
    run_local.add_argument("--s3-bucket")
    run_local.add_argument("--s3-prefix")
    run_local.add_argument("--s3-endpoint-url")
    run_local.add_argument("--out")
    run_local.add_argument("--format", choices=("text", "json"), default="text")
    run_local.set_defaults(func=cmd_run_local)

    compose = subparsers.add_parser("compose", help="Manage local official compose stack")
    compose_sub = compose.add_subparsers(dest="compose_command", required=True)
    compose_up = compose_sub.add_parser("up", help="Start Docker Compose stack")
    compose_up.add_argument("--with-minio", action="store_true")
    compose_up.add_argument(
        "--payload-store",
        choices=("filesystem", "s3", "minio"),
        default="filesystem",
    )
    compose_up.add_argument("--executor-workers", type=int, default=1)
    compose_up.add_argument("--worker-image", default="mini-bang-worker:latest",
                            help="Image with mini-bang installed; runs the official runner")
    compose_up.add_argument("--agent-image")
    compose_up.add_argument("--port", type=int, default=8765)
    compose_up.add_argument("--runtime-setting", action="append", default=[], metavar="NAME=VALUE",
                            help="Non-secret MINI_BANG_* runner environment entry, for example a "
                                 "simulator's runtime_environment setting; repeatable")
    compose_up.add_argument("--detach", action="store_true")
    compose_up.add_argument("--format", choices=("text", "json"), default="text")
    compose_up.set_defaults(func=cmd_compose_up)
    compose_down = compose_sub.add_parser("down", help="Stop Docker Compose stack")
    compose_down.add_argument("--volumes", action="store_true")
    compose_down.add_argument("--remove-artifacts", action="store_true")
    compose_down.add_argument("--format", choices=("text", "json"), default="text")
    compose_down.set_defaults(func=cmd_compose_down)

    server_submit = subparsers.add_parser("submit", help="Send a finished run to the scoring server")
    server_submit.add_argument("run", help="run directory, run id under the runs root, or run_record.json")
    server_submit.add_argument("--server", help=f"scoring server URL (default: ${SERVER_URL_ENV})")
    server_submit.add_argument("--token", help=f"participant token (default: ${SERVER_TOKEN_ENV})")
    server_submit.add_argument("--agent-log", action="append", default=[],
                               help="agent log file attached to the run log (repeatable)")
    server_submit.add_argument("--format", choices=("text", "json"), default="text")
    server_submit.set_defaults(func=cmd_server_submit)

    submit = subparsers.add_parser("run-docker", help="Run an agent container against a local runner (debug only)")
    submit.add_argument("--task", required=True)
    submit.add_argument("--instance")
    submit.add_argument("--agent-image")
    submit_agent_group = submit.add_mutually_exclusive_group()
    submit_agent_group.add_argument("--agent")
    submit_agent_group.add_argument("--agent-file")
    submit.add_argument("--agent-command")
    submit.add_argument("--timeout", type=int, default=600)
    submit.add_argument("--max-wall-clock", type=int)
    submit.add_argument(
        "--payload-store",
        choices=("filesystem", "s3", "minio"),
        default="filesystem",
    )
    submit.add_argument("--s3-bucket")
    submit.add_argument("--s3-prefix")
    submit.add_argument("--s3-endpoint-url")
    submit.add_argument("--out")
    submit.add_argument("--format", choices=("text", "json"), default="text")
    submit.add_argument("--allow-host-gateway", action="store_true")
    submit.add_argument("--insecure-host-network", action="store_true")
    submit.add_argument("--i-know-this-is-insecure", action="store_true")
    submit.set_defaults(func=cmd_run_docker)

    results = subparsers.add_parser("results", help="Inspect saved runs")
    results_sub = results.add_subparsers(dest="results_command", required=True)
    results_list = results_sub.add_parser("list", help="List saved runs")
    results_list.add_argument("--runs-dir")
    results_list.add_argument("--format", choices=("table", "text", "json"), default="table")
    results_list.set_defaults(func=cmd_results_list)
    results_show = results_sub.add_parser("show", help="Show one run")
    results_show.add_argument("run_id")
    results_show.add_argument("--trace", action="store_true")
    results_show.add_argument("--artifacts", action="store_true")
    results_show.add_argument("--format", choices=("text", "json"), default="text")
    results_show.set_defaults(func=cmd_results_show)
    results_export = results_sub.add_parser("export", help="Export one run")
    results_export.add_argument("run_id")
    results_export.add_argument("--format", choices=("json", "jsonl", "zip"), default="json")
    results_export.add_argument("--include-artifacts", action="store_true")
    results_export.add_argument("--out")
    results_export.set_defaults(func=cmd_results_export)

    artifact = subparsers.add_parser("artifact", help="Materialize run artifacts")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    artifact_get = artifact_sub.add_parser("get", help="Get an artifact")
    artifact_get.add_argument("artifact_id")
    artifact_get.add_argument("--run", required=True)
    artifact_get.add_argument("--out")
    artifact_get.add_argument("--as-path", action="store_true")
    artifact_get.add_argument(
        "--format",
        choices=("metadata", "download", "text", "json"),
        default="metadata",
    )
    artifact_get.set_defaults(func=cmd_artifact_get)

    storage = subparsers.add_parser("storage", help="Payload storage commands")
    storage_sub = storage.add_subparsers(dest="storage_command", required=True)
    storage_test = storage_sub.add_parser("test", help="Test payload storage")
    storage_test.add_argument(
        "--mode",
        choices=("inline", "memory", "filesystem", "s3", "auto"),
        default="inline",
    )
    storage_test.add_argument("--payload-dir")
    storage_test.add_argument("--s3-bucket")
    storage_test.add_argument("--s3-prefix")
    storage_test.add_argument("--s3-endpoint-url")
    storage_test.add_argument("--write-test", action="store_true")
    storage_test.add_argument("--read-test", action="store_true")
    storage_test.add_argument("--delete-test", action="store_true")
    storage_test.add_argument("--format", choices=("text", "json"), default="text")
    storage_test.set_defaults(func=cmd_storage_test)

    k8s = subparsers.add_parser("k8s", help="Kubernetes official-mode commands")
    k8s_sub = k8s.add_subparsers(dest="k8s_command", required=True)
    k8s_render = k8s_sub.add_parser("render", help="Render Kubernetes manifests")
    add_k8s_common_args(k8s_render)
    k8s_render.add_argument("--output")
    k8s_render.add_argument("--format", choices=("text", "json"), default="text")
    k8s_render.set_defaults(func=cmd_k8s_render)
    k8s_install = k8s_sub.add_parser("install", help="Install Kubernetes stack")
    add_k8s_common_args(k8s_install)
    k8s_install.add_argument("--create-namespace", action="store_true")
    k8s_install.add_argument("--with-minio", action="store_true")
    k8s_install.add_argument("--values")
    k8s_install.add_argument("--format", choices=("text", "json"), default="text")
    k8s_install.add_argument("--i-know-this-is-insecure", action="store_true")
    k8s_install.set_defaults(func=cmd_k8s_install)

    # Commands and deployment options that simulator, task and client manifests declare ("cli").
    extension = load_cli_extensions(subparsers, k8s_sub)
    for deployment_parser in (k8s_render, k8s_install):
        extension.add_deployment_options_to(deployment_parser)

    cache = subparsers.add_parser("cache", help="Manage CLI cache")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_clear = cache_sub.add_parser("clear", help="Clear cache files")
    cache_clear.add_argument("--cache-dir")
    cache_clear.add_argument("--older-than")
    cache_clear.add_argument("--all", action="store_true")
    cache_clear.add_argument("--format", choices=("text", "json"), default="text")
    cache_clear.set_defaults(func=cmd_cache_clear)

    return parser


def add_k8s_common_args(parser: argparse.ArgumentParser) -> None:
    from mini_bang.client.deployment.persistent_pool import add_pool_startup_arguments
    parser.add_argument("--namespace", default="mini-bang")
    add_pool_startup_arguments(parser)
    parser.add_argument("--worker-image", default="mini-bang-worker:latest")
    parser.add_argument("--minio-image", default="mini-bang-storage:2025-09-07",
                        help="Storage image built from deploy/shared/Dockerfile.storage")
    parser.add_argument(
        "--payload-store",
        choices=("s3", "minio", "filesystem"),
        default="s3",
    )
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--runtime-setting", action="append", default=[], metavar="NAME=VALUE",
                        help="Non-secret MINI_BANG_* ConfigMap entry for runner and workers, "
                             "for example a simulator's runtime_environment setting; repeatable")
    parser.add_argument("--judge-environment",
                        help="JSON file with MINI_BANG_JUDGE_* settings for runner-side description scoring")
