"""Shared layers name no simulator or task; specific code lives in its own package.

Simulator and task ids come from the registries, and the packages that hold their
code come from the manifest entries (factory, API, policy, scorer, CLI, ...). Every
text file of the shared layers is scanned for those ids and package names, and
every Python file for imports of those packages. Discovery reads `config.json`
manifests, so no shared file needs a simulator or task name.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import mini_bang
from mini_bang.simulators.registry import list_simulator_entries
from mini_bang.tasks.loader import TaskLoader

PACKAGE = Path(mini_bang.__file__).resolve().parent
SHARED_LAYERS = ("framework", "execution", "backends", "client", "worker", "pool", "artifacts", "mcp",
                 "testing", "visualization", "agents", "rubric_judge")
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".sh", ".toml", ".txt", ".md", ".cfg"}
# Shared file (relative to the mini_bang package) -> why it may name a simulator or task.
# Empty: simulators and tasks are found through their manifests.
ALLOWED: dict[str, str] = {}


def _packages(paths, root):
    """`mini_bang.<root>.<package>` components of manifest import paths."""
    prefix = f"mini_bang.{root}."
    return {path[len(prefix):].split(".")[0].split(":")[0]
            for path in paths if isinstance(path, str) and path.startswith(prefix)}


def specific_packages() -> dict[str, set[str]]:
    simulators = list_simulator_entries()
    tasks = TaskLoader().specs()
    simulator_paths = [path for spec in simulators for path in (
        spec.factory_path, spec.response_builder_path, spec.aggregator_path, spec.config.get("cli"))]
    task_paths = [path for spec in tasks for path in (
        spec.package + ".", spec.task_class_path, spec.api_class_path, spec.policy_class_path,
        spec.scorer_class_path, spec.config.get("cli"), spec.config.get("acceptance"),
        *spec.config.get("agent_files", ()))]
    return {"simulators": _packages(simulator_paths, "simulators"), "tasks": _packages(task_paths, "tasks")}


def forbidden_names() -> set[str]:
    names = {spec.simulator_id for spec in list_simulator_entries()}
    for spec in TaskLoader().specs():
        names |= {spec.task_id, spec.task_id.replace("/", "-"), spec.task_id.split("/")[0]}
    for packages in specific_packages().values():
        names |= packages
    return names


def name_pattern(names) -> re.Pattern[str]:
    # "flow_lenia" also matches "flow-lenia" and "Flow Lenia"; ids never match inside longer words.
    variants = sorted({re.sub(r"[-_ ]", "[-_ ]", re.escape(name).replace(r"\ ", " ")) for name in names},
                      key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(variants) + r")(?![A-Za-z0-9])", re.IGNORECASE)


def shared_files():
    files = [path for path in PACKAGE.glob("*.py")]
    for layer in SHARED_LAYERS:
        files += [path for path in (PACKAGE / layer).rglob("*")
                  if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts]
    return sorted(files)


def test_registries_supply_the_names_this_check_looks_for():
    names = forbidden_names()
    assert {spec.simulator_id for spec in list_simulator_entries()} <= names
    assert all(spec.task_id in names for spec in TaskLoader().specs())
    assert specific_packages()["simulators"] and specific_packages()["tasks"]
    assert "base" not in names and "registry" not in names and "loader" not in names


def test_shared_layers_name_no_simulator_or_task():
    pattern = name_pattern(forbidden_names())
    hits = []
    for path in shared_files():
        relative = str(path.relative_to(PACKAGE))
        if relative in ALLOWED:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = pattern.search(line)
            if match:
                hits.append(f"{relative}:{number}: {match.group(0)!r} in {line.strip()[:120]}")
    assert not hits, "Move simulator- or task-specific code into its package:\n" + "\n".join(hits)


def test_shared_layers_import_no_simulator_or_task_package():
    packages, pattern = specific_packages(), name_pattern(forbidden_names())
    hits = []
    for path in shared_files():
        relative = str(path.relative_to(PACKAGE))
        if path.suffix != ".py" or relative in ALLOWED:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                parts = module.split(".")
                if (len(parts) > 2 and parts[0] == "mini_bang" and parts[1] in packages
                        and (parts[2] in packages[parts[1]] or pattern.search(parts[2]))):
                    hits.append(f"{relative}:{node.lineno}: imports {module}")
    assert not hits, "Shared layers must not import simulator or task packages:\n" + "\n".join(hits)


@pytest.mark.parametrize("text", ["from mini_bang.tasks.raf_common import x", "Flow Lenia pool", "raf-timing gold",
                                  "flow-lenia-data", "RAF network", "tasks/flow_lenia_common/probe.py"])
def test_name_pattern_catches_spelling_variants(text):
    assert name_pattern(forbidden_names()).search(text)


@pytest.mark.parametrize("text", ["draft", "graft", "MiniBang", "loader", "simulators.base", "registry"])
def test_name_pattern_ignores_generic_words(text):
    assert not name_pattern(forbidden_names()).search(text)
