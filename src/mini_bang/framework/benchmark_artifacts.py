from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_bang.artifacts import ArtifactResolver
from mini_bang.framework.task import InstanceSpec, TaskSpec

# Folder that holds hidden gold outside the installed package (the scoring server
# sets it); gold of task package P then lives in <root>/P/ with the layout of P/gold/.
GOLD_ROOT_ENV = "MINI_BANG_GOLD_ROOT"
# Instances with "split": "dev" ship their gold with the open package; any other
# split (absent means test) is scored by the scoring server.
DEV_SPLIT = "dev"
TEST_SPLIT = "test"


class GoldUnavailable(FileNotFoundError):
    """The gold of an instance is not in this installation (test gold stays on the scoring server)."""


def gold_root() -> Path | None:
    value = os.environ.get(GOLD_ROOT_ENV, "").strip()
    return Path(value) if value else None


def task_gold_dir(package_dir: Path, root: Path | None = None) -> Path:
    """Gold folder of one task package: ``<root>/<package>`` with a gold root, else ``<package>/gold``."""
    root = root if root is not None else gold_root()
    return Path(root) / Path(package_dir).name if root else Path(package_dir) / "gold"


def instance_split(instance: Any) -> str:
    """``dev`` or ``test`` for an InstanceSpec or a raw instance mapping."""
    config = getattr(instance, "config", instance) or {}
    return str(config.get("split") or TEST_SPLIT)


@dataclass(frozen=True)
class GoldStore:
    """Immutable accessor for hidden gold artifacts of one instance."""

    task_id: str
    instance_id: str
    path: Path
    data: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    artifact_resolver: ArtifactResolver | None = field(default=None, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a deep copy of a gold artifact field."""
        return copy.deepcopy(self.data.get(key, default))

    def to_metadata(self) -> dict[str, Any]:
        """Return non-secret metadata about the loaded gold artifact."""
        return {
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "gold_version": self.version,
            "gold_path": str(self.path),
        }


class BenchmarkArtifactStore:
    """Filesystem loader for visible instances and hidden gold JSON artifacts."""

    def __init__(self, *, artifact_resolver: ArtifactResolver | None = None,
                 gold_root: Path | str | None = None) -> None:
        self.artifact_resolver = artifact_resolver
        self.gold_root = Path(gold_root) if gold_root else None

    def load_instance(self, task_spec: TaskSpec, instance_id: str | None = None) -> InstanceSpec:
        """Load the visible artifact for a task instance."""
        resolved_id = instance_id or task_spec.default_instance_id
        package_dir = self._package_dir(task_spec)
        instance_path = package_dir / "instances" / f"{resolved_id}.json"
        if not instance_path.is_file():
            raise FileNotFoundError(
                f"Task '{task_spec.task_id}' has no visible instance artifact '{instance_path}'"
            )
        with instance_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        source = _bundled_gold_source(config)
        gold_id = source['instance_id'] if source else resolved_id
        gold_path = task_gold_dir(package_dir, self.gold_root) / f"{gold_id}.json"
        return InstanceSpec.from_config(
            task_id=task_spec.task_id,
            package=task_spec.package,
            instance_id=resolved_id,
            config=config,
            config_path=instance_path,
            gold_path=gold_path,
            task_metadata=task_spec.metadata,
        )

    def has_gold(self, instance_spec: InstanceSpec) -> bool:
        """Whether this installation can score the instance: dev gold, or test gold on the scoring server."""
        try:
            self.load_gold(instance_spec)
        except GoldUnavailable:
            return False
        return True

    def load_gold(self, instance_spec: InstanceSpec) -> GoldStore:
        """Load hidden gold artifacts for a concrete instance."""
        if instance_spec.gold_path is None:
            raise ValueError(
                f"Instance '{instance_spec.task_id}/{instance_spec.instance_id}' has no gold path"
            )
        if not instance_spec.gold_path.is_file():
            if instance_split(instance_spec) != DEV_SPLIT:
                raise GoldUnavailable(
                    f"Instance {instance_spec.task_id}/{instance_spec.instance_id} is a test instance; "
                    "its gold is held by the scoring server (see `mini-bang submit`)"
                )
            raise GoldUnavailable(
                f"Missing gold artifact for '{instance_spec.task_id}/{instance_spec.instance_id}': "
                f"{instance_spec.gold_path}"
            )
        body = instance_spec.gold_path.read_bytes()
        if instance_spec.config_path is not None:
            config = json.loads(instance_spec.config_path.read_text())
            source = _bundled_gold_source(config)
            if source:
                if (instance_spec.gold_path.name != source['instance_id'] + '.json'
                        or hashlib.sha256(body).hexdigest() != source['sha256']):
                    raise ValueError('Versioned bundled gold source does not match its immutable binding')
        data = json.loads(body)
        return GoldStore(
            task_id=instance_spec.task_id,
            instance_id=instance_spec.instance_id,
            path=instance_spec.gold_path,
            data=copy.deepcopy(data),
            version=str(data.get("gold_version", data.get("version", "1"))),
            artifact_resolver=self.artifact_resolver,
        )

    def list_instance_ids(self, task_spec: TaskSpec) -> tuple[str, ...]:
        """Return visible instance ids available for a task."""
        instances_dir = self._package_dir(task_spec) / "instances"
        if not instances_dir.is_dir():
            return ()
        return tuple(sorted(path.stem for path in instances_dir.glob("*.json")))

    def _package_dir(self, task_spec: TaskSpec) -> Path:
        if task_spec.package_dir is None:
            raise ValueError(f"Task spec '{task_spec.task_id}' has no package directory")
        return task_spec.package_dir

class ArtifactStore(BenchmarkArtifactStore):
    """Deprecated alias for BenchmarkArtifactStore."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ArtifactStore is deprecated; use BenchmarkArtifactStore for benchmark "
            "instances/gold, ArtifactResolver for external source artifacts, and "
            "PayloadStore for runtime outputs.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _bundled_gold_source(config):
    """Select an existing bundled gold version without copying its contents.

    This is a task-package version selection, not an external artifact URI or
    transport. Source identity stays private to the organizer's instance loader.
    """
    source = config.get('gold_source')
    if source is None:
        return None
    if not isinstance(source, dict) or set(source) != {'instance_id', 'sha256'}:
        raise ValueError('gold_source requires exactly instance_id and sha256')
    name, digest = source['instance_id'], source['sha256']
    if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
            or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('Invalid bundled gold source identity or checksum')
    return source
