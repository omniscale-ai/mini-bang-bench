from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_bang.artifacts import ArtifactCache, ArtifactResolver
from mini_bang.artifacts.stores import (
    FileArtifactSourceStore,
    GDriveArtifactSourceStore,
    HttpArtifactSourceStore,
    S3ArtifactSourceStore,
)
from mini_bang.execution.payload_stores import (
    FileSystemPayloadStore,
    InMemoryPayloadStore,
    PayloadStore,
    S3PayloadStore,
)
from mini_bang.execution.payload_stores.registry import payload_mode_and_store_from_env
from mini_bang.importing import import_from_path


@dataclass(frozen=True)
class RuntimeContext:
    """Serializable runtime settings that may cross process/container boundaries."""

    run_id: str | None = None
    task_id: str | None = None
    instance_id: str | None = None
    artifact: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    security: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        instance_id: str | None = None,
    ) -> "RuntimeContext":
        artifact: dict[str, Any] = {}
        cache_dir = os.getenv("MINI_BANG_ARTIFACT_CACHE_DIR")
        if cache_dir:
            artifact["cache_dir"] = cache_dir
        roots = _split_env("MINI_BANG_ARTIFACT_FILE_ALLOWED_ROOTS") or _split_env(
            "MINI_BANG_ARTIFACT_FILE_ROOTS"
        )
        if roots:
            artifact["allowed_file_roots"] = roots
        schemes = _split_env("MINI_BANG_ARTIFACT_ALLOWED_SCHEMES")
        if schemes:
            artifact["allowed_schemes"] = schemes
        raw_prefixes = os.getenv("MINI_BANG_ARTIFACT_ALLOWED_S3_PREFIXES", "").strip()
        s3_prefixes = json.loads(raw_prefixes) if raw_prefixes.startswith("[") else [
            item.strip() for item in raw_prefixes.replace("\n", ",").split(",") if item.strip()]
        if s3_prefixes:
            artifact["allowed_s3_prefixes"] = s3_prefixes
        for env_key, key in (("MINI_BANG_ARTIFACT_S3_ENDPOINT_URL", "s3_endpoint_url"),
                             ("MINI_BANG_ARTIFACT_S3_REGION", "s3_region")):
            if os.getenv(env_key):
                artifact[key] = os.environ[env_key]
        payload = _payload_context_from_env()
        security = {
            "network_enabled": _env_bool("MINI_BANG_ARTIFACT_NETWORK_ENABLED", True),
            "gdrive_enabled": _env_bool("MINI_BANG_ARTIFACT_GDRIVE_ENABLED", True),
            "allow_pickle_payloads": _env_bool("MINI_BANG_ALLOW_PICKLE_PAYLOADS", False),
        }
        return cls(
            run_id=run_id,
            task_id=task_id,
            instance_id=instance_id,
            artifact=artifact,
            payload=payload,
            security=security,
        )

    @classmethod
    def for_dev_local(cls, **kwargs: Any) -> "RuntimeContext":
        """Create a context from local development environment settings."""
        return cls.from_env(**kwargs)

    @classmethod
    def for_official(cls, **kwargs: Any) -> "RuntimeContext":
        """Create an official-mode context from explicit top-level config/env."""
        context = cls.from_env(**kwargs)
        security = {**context.security, "allow_pickle_payloads": False}
        return dataclasses.replace(context, security=security)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeContext":
        """Rebuild a context sent across a process or transport boundary by ``to_dict``."""
        return cls(
            run_id=data.get("run_id"),
            task_id=data.get("task_id"),
            instance_id=data.get("instance_id"),
            artifact=dict(data.get("artifact") or {}),
            payload=dict(data.get("payload") or {}),
            security=dict(data.get("security") or {}),
            env=dict(data.get("env") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(dataclasses.asdict(self))

    def stable_key(self) -> str:
        import hashlib

        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def with_run(
        self,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        instance_id: str | None = None,
    ) -> "RuntimeContext":
        return dataclasses.replace(
            self,
            run_id=run_id if run_id is not None else self.run_id,
            task_id=task_id if task_id is not None else self.task_id,
            instance_id=instance_id if instance_id is not None else self.instance_id,
        )


# ``RuntimeContext.security["authority"]`` of scorer-side work (hidden simulations, scorer-only refs).
SCORING_AUTHORITY = "scoring"


@dataclass
class RuntimeServices:
    """Live runtime clients and stores reconstructed from a RuntimeContext."""

    artifact_resolver: ArtifactResolver
    payload_store: PayloadStore | None = None
    # ``security.authority`` of the context: SCORING_AUTHORITY for scorer-side work, None for agent sessions.
    authority: str | None = None

    @classmethod
    def from_context(cls, context: RuntimeContext) -> "RuntimeServices":
        artifact_cfg = dict(context.artifact)
        security_cfg = dict(context.security)
        cache = ArtifactCache(artifact_cfg.get("cache_dir"))
        stores = _artifact_stores_from_config(artifact_cfg, security_cfg)
        resolver = ArtifactResolver(
            cache=cache,
            stores=stores,
            allowed_schemes=artifact_cfg.get("allowed_schemes"),
            allowed_file_roots=artifact_cfg.get("allowed_file_roots"),
            allowed_s3_prefixes=artifact_cfg.get("allowed_s3_prefixes"),
            network_enabled=bool(security_cfg.get("network_enabled", True)),
            gdrive_enabled=bool(security_cfg.get("gdrive_enabled", True)),
        )
        payload_store = _payload_store_from_config(dict(context.payload))
        return cls(artifact_resolver=resolver, payload_store=payload_store,
                   authority=security_cfg.get("authority"))


def _artifact_stores_from_config(
    artifact_cfg: dict[str, Any],
    security_cfg: dict[str, Any],
) -> list[Any]:
    stores: list[Any] = []
    for spec in artifact_cfg.get("stores") or ():
        if not isinstance(spec, dict) or not spec.get("class"):
            raise ValueError("artifact store specs must include a 'class' import path")
        cls = import_from_path(str(spec["class"]))
        stores.append(cls(**dict(spec.get("kwargs") or {})))
    if not stores:
        if security_cfg.get("gdrive_enabled", True):
            stores.append(GDriveArtifactSourceStore())
        stores.append(
            FileArtifactSourceStore(
                allowed_roots=artifact_cfg.get("allowed_file_roots")
            )
        )
        if security_cfg.get("network_enabled", True):
            stores.append(HttpArtifactSourceStore())
        stores.append(
            S3ArtifactSourceStore(
                endpoint_url=artifact_cfg.get("s3_endpoint_url"),
                region_name=artifact_cfg.get("s3_region"),
            )
        )
    return stores


def _payload_store_from_config(payload_cfg: dict[str, Any]) -> PayloadStore | None:
    kind = payload_cfg.get("store_kind") or payload_cfg.get("kind") or payload_cfg.get("mode")
    if not kind:
        return None
    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized in {"inline", "none", "off"}:
        return None
    allow_pickle = bool(payload_cfg.get("allow_pickle_payloads", False))
    config = dict(payload_cfg.get("config") or payload_cfg)
    if normalized in {"filesystem", "fs", "file", "k8s", "pvc", "filesystem_pvc"}:
        root = config.get("root") or config.get("dir") or config.get("path")
        if not root:
            raise ValueError("filesystem payload context requires root")
        return FileSystemPayloadStore(root, allow_pickle_payloads=allow_pickle)
    if normalized in {"memory", "in_memory"}:
        return InMemoryPayloadStore(namespace=config.get("namespace"))
    if normalized in {"s3", "minio", "aws_s3", "object", "object_store"}:
        bucket = config.get("bucket")
        if not bucket:
            raise ValueError("s3 payload context requires bucket")
        return S3PayloadStore(
            bucket=str(bucket),
            prefix=str(config.get("prefix") or ""),
            endpoint_url=config.get("endpoint_url"),
            region_name=config.get("region_name") or config.get("region"),
            download_dir=config.get("download_dir")
            or str(Path(tempfile.gettempdir()) / "mini_bang_payloads"),
            allow_pickle_payloads=allow_pickle,
        )
    if config.get("class"):
        cls = import_from_path(str(config["class"]))
        return cls(**dict(config.get("kwargs") or {}))
    raise ValueError(f"Unsupported payload store kind '{kind}' in RuntimeContext")


def _payload_context_from_env() -> dict[str, Any]:
    # Same mode/store rule as the executor's create_payload_storage_from_env.
    _mode, store_kind = payload_mode_and_store_from_env()
    kind = store_kind or ""
    if not kind and os.getenv("MINI_BANG_PAYLOAD_DIR"):
        kind = "filesystem"
    if not kind or kind in {"inline", "none", "off"}:
        return {}
    if kind in {"filesystem", "fs", "file", "k8s", "pvc", "filesystem-pvc"}:
        return {
            "kind": "filesystem",
            "root": os.getenv("MINI_BANG_PAYLOAD_DIR", "/tmp/mini_bang_payloads"),
            "allow_pickle_payloads": _env_bool("MINI_BANG_ALLOW_PICKLE_PAYLOADS", False),
        }
    if kind in {"memory", "in-memory", "in_memory"}:
        return {
            "kind": "memory",
            "namespace": os.getenv("MINI_BANG_PAYLOAD_NAMESPACE"),
        }
    if kind in {"s3", "minio", "object_store"}:
        return {
            "kind": "s3",
            "bucket": os.getenv("MINI_BANG_PAYLOAD_S3_BUCKET", ""),
            "prefix": os.getenv("MINI_BANG_PAYLOAD_S3_PREFIX", ""),
            "endpoint_url": os.getenv("MINI_BANG_PAYLOAD_S3_ENDPOINT_URL"),
            "region_name": os.getenv("MINI_BANG_PAYLOAD_S3_REGION"),
            "download_dir": os.getenv("MINI_BANG_PAYLOAD_DOWNLOAD_DIR"),
            "allow_pickle_payloads": _env_bool("MINI_BANG_ALLOW_PICKLE_PAYLOADS", False),
        }
    return {"kind": kind}


def _split_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item for item in raw.split(os.pathsep) if item]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        raise TypeError("RuntimeContext must contain strings, not Path objects")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"RuntimeContext contains non-serializable {type(value).__name__}")
