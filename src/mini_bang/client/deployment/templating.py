from __future__ import annotations

import json
import re
from importlib.resources import files
from string import Template
from typing import Any


TEMPLATE_PACKAGE = "mini_bang.client.deployment.templates"


def render_template(relative_path: str, values: dict[str, Any] | None = None) -> str:
    template = files(TEMPLATE_PACKAGE).joinpath(*relative_path.split("/")).read_text(encoding="utf-8")
    rendered = Template(template).safe_substitute(_template_values(values or {}))
    return rendered.rstrip() + "\n"


def yaml_string(value: Any) -> str:
    return json.dumps(str(value))


def _template_values(values: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in values.items()}


def runtime_setting_entries(items: Any, reserved_text: str = "") -> dict[str, str]:
    """Validated `--runtime-setting NAME=VALUE` options of a deployment.

    A simulator's non-secret deployment settings (its config.json
    `runtime_environment`) reach the runner and workers this way. Names that the
    rendered stack already sets (``reserved_text``) are refused.
    """
    from mini_bang.simulators.registry import _public_setting_name
    reserved = set(re.findall(r"^\s+(MINI_BANG_[A-Z0-9_]+):", reserved_text, re.M))
    entries: dict[str, str] = {}
    for item in items or ():
        name, separator, value = str(item).partition("=")
        if not separator or not _public_setting_name(name):
            raise ValueError("--runtime-setting must be NAME=VALUE with a non-secret MINI_BANG_* name")
        if name in reserved or name in entries:
            raise ValueError(f"--runtime-setting {name} would repeat an existing setting")
        entries[name] = value
    return entries
