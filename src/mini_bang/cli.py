from __future__ import annotations

from mini_bang.client.cli.app import main
from mini_bang.client.cli.config import load_config, preset_config
from mini_bang.client.cli.errors import (
    CliError,
    EXIT_CONFIG,
    EXIT_INTERNAL,
    EXIT_RUN_FAILED,
    EXIT_SECURITY,
    EXIT_STORAGE,
    EXIT_SUCCESS,
    EXIT_USAGE,
)
from mini_bang.client.cli.parser import build_parser
from mini_bang.client.cli.records import save_run_outputs
from mini_bang.client.deployment.compose import render_compose_yaml
from mini_bang.client.deployment.k8s import effective_k8s_payload_store, render_k8s_agent_job, render_k8s_manifests

__all__ = [
    "main",
    "build_parser",
    "load_config",
    "preset_config",
    "save_run_outputs",
    "render_compose_yaml",
    "render_k8s_manifests",
    "render_k8s_agent_job",
    "effective_k8s_payload_store",
    "CliError",
    "EXIT_SUCCESS",
    "EXIT_RUN_FAILED",
    "EXIT_USAGE",
    "EXIT_CONFIG",
    "EXIT_SECURITY",
    "EXIT_STORAGE",
    "EXIT_INTERNAL",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
