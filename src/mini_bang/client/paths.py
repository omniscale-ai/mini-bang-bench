from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from mini_bang.naming import readable_slug


WORKSPACE_ROOT = Path("workdir")
RUNS_ROOT = WORKSPACE_ROOT / "runs"
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
ARTIFACTS_ROOT = DATASETS_ROOT / "task-artifacts"
CACHE_ROOT = WORKSPACE_ROOT / "cache"
PAYLOADS_ROOT = RUNS_ROOT / "observations"
INFRASTRUCTURE_ROOT = WORKSPACE_ROOT / "infrastructure"
KUBERNETES_VALIDATION_STATUS = INFRASTRUCTURE_ROOT / "kubernetes-validation" / "latest-status.json"


def new_run_directory(purpose: str, *, config=None, now=None) -> Path:
    """Use the same configured storage root for every benchmark entry point."""
    root = Path(_nested_get(config or {}, "runs.root", str(RUNS_ROOT)))
    instant = now or datetime.now(timezone.utc)
    name = "__".join(readable_slug(part) for part in purpose.split("__"))
    return root / f"{instant.strftime('%Y-%m-%d__%H-%M-%S-%f-utc')}__{name}"


def default_runtime_roots() -> dict[str, str]:
    return {
        "runs": f"./{RUNS_ROOT.as_posix()}",
        "artifacts": f"./{ARTIFACTS_ROOT.as_posix()}",
        "cache": f"./{CACHE_ROOT.as_posix()}",
        "payloads": f"./{PAYLOADS_ROOT.as_posix()}",
    }


def init_directories(config: dict[str, Any]) -> list[Path]:
    """Return the runtime directories that init should own.

    `mini-bang.yaml` stays in the project root because it is source-controlled
    configuration. Generated data lives under `workdir/`.
    """
    return unique_paths(
        [
            Path(_nested_get(config, "runs.root", str(RUNS_ROOT))),
            Path(_nested_get(config, "artifacts.root", str(ARTIFACTS_ROOT))),
            Path(_nested_get(config, "cache.root", str(CACHE_ROOT))),
            Path(_nested_get(config, "payload.filesystem.root", str(PAYLOADS_ROOT))),
            INFRASTRUCTURE_ROOT,
        ]
    )


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _nested_get(value: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
