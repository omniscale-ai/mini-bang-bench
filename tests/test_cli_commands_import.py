"""Every command the `mini-bang` CLI registers can run in this installation.

Each registered command has an importable handler, and the CLI code names no
`mini_bang` module (import, lazy import or dotted string) that this tree lacks,
so optional commands that live outside the tree are absent instead of dangling.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import re
from pathlib import Path

import pytest

import mini_bang.client.cli as cli_package
from mini_bang.client.cli.parser import build_parser

IMPORT_PATH = re.compile(r"mini_bang(?:\.\w+)+:\w+")
IMPORTERS = {"import_module", "import_from_path", "find_spec", "__import__"}


def leaf_commands(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield from leaf_commands(child, path + (name,))
            return
    yield path, parser


def named_modules(source: Path) -> set[str]:
    """Imports, `module:attribute` strings and strings (or constants holding them) passed to importers."""
    tree = ast.parse(source.read_text())
    constants = {target.id: node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                 for target in node.targets if isinstance(target, ast.Name)}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and IMPORT_PATH.fullmatch(node.value):
            names.add(node.value.split(":")[0])
        elif isinstance(node, ast.Call) and node.args and getattr(node.func, "attr", getattr(node.func, "id", None)) in IMPORTERS:
            argument = node.args[0]
            value = argument.value if isinstance(argument, ast.Constant) else constants.get(getattr(argument, "id", None))
            if isinstance(value, str):
                names.add(value.split(":")[0])
    return {name for name in names if name.split(".")[0] == "mini_bang"}


def resolves(name: str) -> bool:
    """A module of this tree, or an attribute of one (dotted strings may end in a name)."""
    try:
        if importlib.util.find_spec(name) is not None:
            return True
    except ModuleNotFoundError:
        pass
    parent, _, attribute = name.rpartition(".")
    try:
        return hasattr(importlib.import_module(parent), attribute)
    except ImportError:
        return False


COMMANDS = list(leaf_commands(build_parser()))
CLI_SOURCES = sorted(Path(cli_package.__file__).parent.glob("*.py"))


@pytest.mark.parametrize("path,parser", COMMANDS, ids=[" ".join(path) for path, _ in COMMANDS])
def test_every_registered_command_has_an_importable_handler(path, parser):
    handler = parser.get_default("func")
    assert callable(handler), f"mini-bang {' '.join(path)} has no handler"
    module = importlib.import_module(handler.__module__)
    assert getattr(module, handler.__name__) is handler
    missing = sorted(name for name in named_modules(Path(module.__file__)) if not resolves(name))
    assert not missing, f"mini-bang {' '.join(path)}: {handler.__module__} names missing modules {missing}"


@pytest.mark.parametrize("source", CLI_SOURCES, ids=[source.name for source in CLI_SOURCES])
def test_cli_code_names_only_modules_of_this_tree(source):
    missing = sorted(name for name in named_modules(source) if not resolves(name))
    assert not missing, f"{source.name} names missing modules {missing}"
