from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mini_bang.artifacts.refs import ArtifactRef, coerce_artifact_ref


@dataclass(frozen=True)
class ArtifactManifest:
    """Loaded arbitrary artifact manifest data."""

    ref: ArtifactRef
    data: Any


def load_manifest(ref: ArtifactRef | dict[str, Any], *, resolver: Any | None = None) -> Any:
    """Load a JSON/YAML manifest without interpreting its fields."""
    from mini_bang.artifacts.resolver import get_default_artifact_resolver

    artifact_ref = coerce_artifact_ref(ref)
    active_resolver = resolver or get_default_artifact_resolver()
    body = active_resolver.load_bytes(artifact_ref)
    return _decode_manifest(body, uri=artifact_ref.uri, media_type=artifact_ref.media_type)


def _decode_manifest(body: bytes, *, uri: str, media_type: str | None) -> Any:
    text = body.decode("utf-8")
    lowered = uri.lower()
    if media_type == "application/json" or lowered.endswith(".json"):
        return json.loads(text)
    if lowered.endswith((".yaml", ".yml")) or media_type in {
        "application/yaml",
        "application/x-yaml",
        "text/yaml",
    }:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML artifact manifests require PyYAML") from exc
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError(f"Artifact manifest is not valid JSON: {uri}") from exc
        return yaml.safe_load(text)
