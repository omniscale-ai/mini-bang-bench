"""CLI extensions declared by simulator, task and optional client packages.

A manifest entry `"cli": "package.module:register"` names a function that the
core CLI calls with one `CliExtension`. Through it a package adds its own
top-level or `k8s` commands, its own `k8s render` / `k8s install` options, and
a hook that turns those options into non-secret ConfigMap settings. Manifests
are the simulator and task `config.json` files and the optional client
manifests `client/extensions/*.json`; a client manifest may also name
`"runner_gateway": "package.module:gateway"`, the organizer routes `/v1/pool`
and `/v1/agents` of the official runner. The core CLI knows no simulator, task
or optional client module by name: an installation without the manifest has
neither the commands nor the routes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from mini_bang.importing import import_from_path

DeploymentSettings = Callable[..., dict[str, str]]
CLIENT_EXTENSIONS_DIR = Path(__file__).resolve().parents[1] / "extensions"


class CliExtension:
    """What one `register(cli)` call may add to the `mini-bang` CLI."""

    def __init__(self, commands: Any, k8s_commands: Any = None) -> None:
        self._commands = commands
        self._k8s_commands = k8s_commands
        self.deployment_options: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.deployment_hooks: list[DeploymentSettings] = []

    def add_command(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Parser of a new top-level command `mini-bang <name>`."""
        return self._commands.add_parser(name, **kwargs)

    def add_k8s_command(self, name: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Parser of a new command `mini-bang k8s <name>`."""
        if self._k8s_commands is None:
            self._k8s_commands = argparse.ArgumentParser().add_subparsers()
        return self._k8s_commands.add_parser(name, **kwargs)

    def add_deployment_option(self, *flags: str, **kwargs: Any) -> None:
        """Option of `mini-bang k8s render` and `mini-bang k8s install` (argparse arguments)."""
        self.deployment_options.append((flags, kwargs))

    def add_deployment_settings(self, hook: DeploymentSettings) -> None:
        """`hook(args, s3_endpoint_url=...)` returns ConfigMap entries for the runner and workers."""
        self.deployment_hooks.append(hook)

    def add_deployment_options_to(self, parser: argparse.ArgumentParser) -> None:
        for flags, kwargs in self.deployment_options:
            parser.add_argument(*flags, **kwargs)


def client_extension_manifests() -> list[dict[str, Any]]:
    """Optional client manifests (`client/extensions/*.json`), sorted by file name; none when absent."""
    if not CLIENT_EXTENSIONS_DIR.is_dir():
        return []
    return [json.loads(path.read_text()) for path in sorted(CLIENT_EXTENSIONS_DIR.glob("*.json"))]


def cli_extension_paths() -> list[str]:
    """`cli` entries of the simulator, task and client manifests, in discovery order."""
    from mini_bang.simulators.registry import list_simulator_entries
    from mini_bang.tasks.loader import TaskLoader

    paths = [spec.config.get("cli") for spec in list_simulator_entries()]
    paths += [spec.config.get("cli") for spec in TaskLoader().specs()]
    paths += [manifest.get("cli") for manifest in client_extension_manifests()]
    return list(dict.fromkeys(path for path in paths if path))


def load_cli_extensions(commands: Any = None, k8s_commands: Any = None) -> CliExtension:
    """Call every declared `register(cli)`; without ``commands`` new commands go to a scratch parser."""
    extension = CliExtension(commands if commands is not None else argparse.ArgumentParser().add_subparsers(),
                             k8s_commands)
    for path in cli_extension_paths():
        import_from_path(path)(extension)
    return extension


def runner_gateway(server: Any) -> Any:
    """Organizer routes of the first client manifest that declares `runner_gateway`, else None."""
    for manifest in client_extension_manifests():
        if manifest.get("runner_gateway"):
            return import_from_path(manifest["runner_gateway"])(server)
    return None


def extension_deployment_settings(args: argparse.Namespace, *, s3_endpoint_url: str) -> dict[str, str]:
    """ConfigMap entries that the declared extensions derive from their deployment options."""
    entries: dict[str, str] = {}
    for hook in load_cli_extensions().deployment_hooks:
        entries.update(hook(args, s3_endpoint_url=s3_endpoint_url))
    return entries
