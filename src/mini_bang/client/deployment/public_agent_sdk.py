"""Stdlib-only agent client: the task API wrapper without simulators, tasks, scorers or gold.

The same files are mounted into Kubernetes agent pods (ConfigMap) and written by
`mini-bang agent-client --output DIR` as a pip-installable directory for agent environments.
"""
from importlib.resources import files
import json
from pathlib import Path

CLIENT_MODULES = ("agent_runtime.py", "agent_types.py", "importing.py")
CLIENT_PYPROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mini-bang-agent-client"
version = "1.0"
description = "mini-bang agent client: task API wrapper only, without simulators, tasks or gold"
requires-python = ">=3.11"
dependencies = []

[project.scripts]
mini-bang-agent = "mini_bang.agent_runtime:main"

[tool.setuptools]
packages = ["mini_bang", "mini_bang.framework"]
py-modules = $modules
"""


def task_agent_files():
    """Public files that task manifests promise agents: name -> resource.

    A task lists them in `config.json` as `"agent_files": ["package:file.py"]`;
    each file is published at the client root under its own name.
    """
    from mini_bang.tasks.loader import TaskLoader
    found = {}
    for spec in TaskLoader().specs():
        for entry in spec.config.get("agent_files", ()):
            package, _, name = entry.partition(":")
            if not package or not name or "/" in name or not name.endswith(".py"):
                raise ValueError(f"Task '{spec.task_id}' agent_files entry must be 'package:file.py'")
            found[name] = files(package).joinpath(name)
    return found


def sdk_files():
    """ConfigMap key -> (path inside the client, text)."""
    root = files("mini_bang")
    exports = "from mini_bang.agent_types import AgentAPI, TaskEnvironment, TaskSubmission\n"
    return {"package.py": ("mini_bang/__init__.py", '"""Public task transport; benchmark internals are absent."""\n'),
            "framework.py": ("mini_bang/framework/__init__.py", exports),
            "task.py": ("mini_bang/framework/task.py", exports),
            **{name: (name, resource.read_text()) for name, resource in task_agent_files().items()},
            **{name: ("mini_bang/" + name, root.joinpath(name).read_text()) for name in CLIENT_MODULES}}


def public_agent_sdk(namespace):
    data = {key: text for key, (_, text) in sdk_files().items()}
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {
        "name": "mini-bang-public-agent-sdk", "namespace": namespace}, "immutable": True, "data": data}


def sdk_volume():
    return {"name": "public-agent-sdk", "configMap": {"name": "mini-bang-public-agent-sdk",
        "defaultMode": 0o444, "items": [{"key": key, "path": path} for key, (path, _) in sdk_files().items()]}}


def write_agent_client(directory):
    """Write the client files and a pyproject; return the written paths relative to ``directory``."""
    directory = Path(directory)
    written = {path: text for path, text in sdk_files().values()}
    modules = sorted(name.removesuffix(".py") for name in task_agent_files())
    written["pyproject.toml"] = CLIENT_PYPROJECT.replace("$modules", json.dumps(modules))
    for path, text in written.items():
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return sorted(written)
