from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from mini_bang.client.paths import ARTIFACTS_ROOT, CACHE_ROOT, PAYLOADS_ROOT, RUNS_ROOT
from mini_bang.client.cli.errors import CliError, EXIT_CONFIG, EXIT_USAGE


DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "dev-local",
    "task": {
        "default_instance_id": "default",
    },
    "backend": {
        "kind": "local",
        "engine": "sequential",
        "workers": 1,
        "reuse_strategy": "none",
    },
    "agent": {
        "placement": "in-process",
    },
    "payload": {
        "mode": "inline",
        "inline_threshold_bytes": 1048576,
        "filesystem": {
            "root": f"./{PAYLOADS_ROOT.as_posix()}",
        },
        "s3": {
            "bucket": None,
            "prefix": "dev/runs",
            "endpoint_url": None,
        },
    },
    "runs": {
        "root": f"./{RUNS_ROOT.as_posix()}",
    },
    "artifacts": {
        "root": f"./{ARTIFACTS_ROOT.as_posix()}",
    },
    "cache": {
        "root": f"./{CACHE_ROOT.as_posix()}",
    },
    "security": {
        "trusted_dev_mode": True,
        "allow_pickle_payloads": False,
        "allow_raw_file_refs": False,
        "mcp_endpoint_exposed_to_agent": False,
    },
}


def load_config(path: str | os.PathLike[str] = "mini-bang.yaml") -> dict[str, Any]:
    config = deep_copy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        return config
    loaded = load_simple_yaml(config_path)
    if not isinstance(loaded, dict):
        raise CliError(f"invalid config file {config_path}: expected mapping", EXIT_CONFIG)
    return deep_merge(config, loaded)

def preset_config(preset: str) -> dict[str, Any]:
    config = deep_copy(DEFAULT_CONFIG)
    if preset == "dev-local":
        return config
    if preset == "compose-local":
        config["mode"] = "compose-local"
        config["agent"]["placement"] = "docker"
        config["payload"]["mode"] = "filesystem"
        config["security"]["trusted_dev_mode"] = False
        return config
    if preset == "k8s":
        config["mode"] = "k8s-official"
        config["agent"]["placement"] = "k8s-pool"
        config["payload"]["mode"] = "minio"
        config["payload"]["s3"]["bucket"] = "mini-bang-payloads"
        config["payload"]["s3"]["endpoint_url"] = "http://mini-bang-minio:9000"
        config["backend"]["kind"] = "cluster"
        config["backend"]["engine"] = "executor_replicas"
        config["security"]["trusted_dev_mode"] = False
        return config
    raise CliError(f"unknown preset {preset}", EXIT_USAGE)

def load_simple_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data
    except ImportError:
        pass
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent_level = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise CliError(f"cannot parse config line: {raw_line}", EXIT_CONFIG)
        while stack and indent_level <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent_level, child))
        else:
            parent[key] = parse_scalar(value.strip())
    return root

def dump_simple_yaml(data: dict[str, Any], level: int = 0) -> str:
    lines = []
    prefix = " " * level
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(dump_simple_yaml(value, level + 2).rstrip())
        else:
            lines.append(f"{prefix}{key}: {format_scalar(value)}")
    return "\n".join(lines) + "\n"

def parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value.strip("'\"")

def format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)

def deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))

def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deep_copy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def nested_get(data: Any, path: str, default: Any = None) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current
