from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mini_bang.artifacts import (
    ArtifactCache,
    ArtifactRef,
    ArtifactResolver,
    coerce_artifact_ref,
    get_default_artifact_resolver,
)


@dataclass(frozen=True)
class FlowLeniaCheckpoint:
    saturation: float
    params: np.ndarray
    step: int | None = None
    loss: float | None = None
    source: str | None = None
    index: int | None = None


class FlowLeniaCheckpointRepository:
    """Flow-Lenia-specific parser/selector over generic materialized artifacts.

    This class knows Flow Lenia checkpoint file shapes. It is not a storage
    client: URI schemes, downloads, cache reuse, and digest checks belong to
    ``ArtifactResolver``.
    """

    def __init__(
        self,
        *,
        checkpoints: list[dict[str, Any]] | None = None,
        checkpoint_ref: ArtifactRef | dict[str, Any] | None = None,
        checkpoint_uri: str | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_param_key: str = "params",
        checkpoint_loss_key: str = "loss",
        checkpoint_steps: list[int] | None = None,
        checkpoint_member_index: int = 0,
        manifest_ref: ArtifactRef | dict[str, Any] | None = None,
        manifest_uri: str | None = None,
        manifest_path: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        artifact_resolver: ArtifactResolver | None = None,
    ) -> None:
        self._inline = list(checkpoints or [])
        self.checkpoint_ref = _flow_lenia_artifact_ref(
            checkpoint_ref,
            uri=checkpoint_uri,
            sha256=checkpoint_sha256,
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_param_key = checkpoint_param_key
        self.checkpoint_loss_key = checkpoint_loss_key
        self.checkpoint_steps = list(checkpoint_steps or [])
        self.checkpoint_member_index = int(checkpoint_member_index)
        self.manifest_ref = _flow_lenia_artifact_ref(manifest_ref, uri=manifest_uri)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        cache_root = Path(cache_dir) if cache_dir is not None else None
        # Runtime/executor code passes artifact_resolver explicitly. The cache_dir
        # and default-resolver paths are local-dev compatibility conveniences.
        self.artifact_resolver = artifact_resolver or (
            ArtifactResolver(cache=ArtifactCache(cache_root))
            if cache_root is not None
            else get_default_artifact_resolver()
        )
        self._loaded: list[FlowLeniaCheckpoint] | None = None

    def nearest(self, saturation: float) -> FlowLeniaCheckpoint:
        value = float(saturation)
        if not 0.0 <= value <= 1.0:
            raise ValueError("flow_lenia saturation must be normalized in [0.0, 1.0]")
        checkpoints = self.load()
        if not checkpoints:
            raise ValueError("No Flow Lenia checkpoints are configured")
        return min(checkpoints, key=lambda item: abs(item.saturation - value))

    def load(self) -> list[FlowLeniaCheckpoint]:
        if self._loaded is None:
            loaded = []
            loaded.extend(self._load_inline(self._inline))
            loaded.extend(self._load_manifest())
            loaded.extend(self._load_checkpoint_file())
            self._loaded = sorted(loaded, key=lambda item: item.saturation)
        return list(self._loaded)

    def _load_inline(self, items: list[dict[str, Any]]) -> list[FlowLeniaCheckpoint]:
        result = []
        for index, item in enumerate(items):
            params = np.asarray(item["params"], dtype=np.float32)
            result.append(
                FlowLeniaCheckpoint(
                    saturation=float(item.get("saturation", item.get("normalized_step", 0.0))),
                    step=_optional_int(item.get("step")),
                    loss=_optional_float(item.get("loss")),
                    source="inline",
                    index=index,
                    params=params,
                )
            )
        return result

    def _load_manifest(self) -> list[FlowLeniaCheckpoint]:
        path = self.manifest_path
        if path is None and self.manifest_ref is not None:
            path = self._materialize_ref(self.manifest_ref)
        if path is None:
            return []
        with Path(path).open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        result = []
        for item in manifest.get("checkpoints") or []:
            if "params" in item:
                result.extend(self._load_inline([item]))
                continue
            loaded = self._load_one_file(
                uri=item.get("uri") or item.get("url"),
                path=item.get("path"),
                sha256=item.get("sha256"),
                params_key=item.get("param_key") or item.get("params_key") or self.checkpoint_param_key,
                loss_key=item.get("loss_key") or self.checkpoint_loss_key,
                steps=item.get("steps"),
                member_index=int(item.get("member_index", self.checkpoint_member_index)),
            )
            saturation = item.get("saturation", item.get("normalized_step"))
            step = item.get("step")
            loss = item.get("loss")
            if saturation is not None or step is not None or loss is not None:
                loaded = [
                    FlowLeniaCheckpoint(
                        saturation=float(saturation if saturation is not None else ckpt.saturation),
                        params=ckpt.params,
                        step=_optional_int(step) if step is not None else ckpt.step,
                        loss=_optional_float(loss) if loss is not None else ckpt.loss,
                        source=ckpt.source,
                        index=ckpt.index,
                    )
                    for ckpt in loaded
                ]
            result.extend(loaded)
        for source in manifest.get("files") or manifest.get("trajectories") or []:
            uri = source.get("uri") or source.get("url")
            source_path = source.get("path")
            sha256 = source.get("sha256")
            key = source.get("param_key") or source.get("params_key") or self.checkpoint_param_key
            loss_key = source.get("loss_key") or self.checkpoint_loss_key
            steps = source.get("steps")
            member_index = int(source.get("member_index", self.checkpoint_member_index))
            result.extend(
                self._load_one_file(
                    uri=uri,
                    path=source_path,
                    sha256=sha256,
                    params_key=key,
                    loss_key=loss_key,
                    steps=steps,
                    member_index=member_index,
                )
            )
        return result

    def _load_checkpoint_file(self) -> list[FlowLeniaCheckpoint]:
        if self.checkpoint_path is None and self.checkpoint_ref is None:
            return []
        return self._load_one_file(
            ref=self.checkpoint_ref,
            uri=None,
            path=self.checkpoint_path,
            sha256=None,
            params_key=self.checkpoint_param_key,
            loss_key=self.checkpoint_loss_key,
            steps=self.checkpoint_steps,
            member_index=self.checkpoint_member_index,
        )

    def _load_one_file(
        self,
        *,
        ref: ArtifactRef | None = None,
        uri: str | None,
        path: str | os.PathLike[str] | None,
        sha256: str | None,
        params_key: str,
        loss_key: str,
        steps: list[int] | None,
        member_index: int,
    ) -> list[FlowLeniaCheckpoint]:
        source_path = Path(path) if path is not None else None
        if source_path is None:
            if ref is not None:
                source_path = self._materialize_ref(ref)
            elif uri is not None:
                source_path = self._materialize_ref(_flow_lenia_artifact_ref(uri=uri, sha256=sha256))
            else:
                raise ValueError("Checkpoint manifest file entry needs uri or path")
        elif sha256:
            verify_sha256(source_path, sha256)
        with source_path.open("rb") as fh:
            payload = pickle.load(fh)
        source = uri or str(source_path)
        return checkpoints_from_payload(
            payload,
            params_key=params_key,
            loss_key=loss_key,
            steps=steps,
            source=source,
            member_index=member_index,
        )

    def _materialize_ref(self, ref: ArtifactRef) -> Path:
        handle = self.artifact_resolver.materialize(ref)
        if handle.local_path is None:
            raise RuntimeError(f"Artifact did not materialize to a local path: {ref.uri}")
        return handle.local_path


def checkpoints_from_payload(
    payload: Any,
    *,
    params_key: str,
    loss_key: str,
    steps: list[int] | None,
    source: str,
    member_index: int,
) -> list[FlowLeniaCheckpoint]:
    if isinstance(payload, dict):
        if params_key in payload:
            params = np.asarray(payload[params_key])
            loss = payload.get(loss_key)
            steps = steps or payload.get("steps") or payload.get("iter") or payload.get("iteration")
            return _checkpoints_from_arrays(params, loss, steps, source, member_index)
        if "checkpoints" in payload:
            return _checkpoints_from_items(payload["checkpoints"], source)
    if isinstance(payload, tuple) and payload:
        params = np.asarray(payload[0])
        loss = payload[1] if len(payload) > 1 else None
        return _checkpoints_from_arrays(params, loss, steps, source, member_index)
    if isinstance(payload, list):
        return _checkpoints_from_items(payload, source)
    return _checkpoints_from_arrays(np.asarray(payload), None, steps, source, member_index)


def _checkpoints_from_items(items: list[Any], source: str) -> list[FlowLeniaCheckpoint]:
    result = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            params = item.get("params")
            saturation = item.get("saturation", item.get("normalized_step"))
            step = item.get("step", item.get("iter", index))
            loss = item.get("loss")
        else:
            params = item
            saturation = None
            step = index
            loss = None
        result.append(
            FlowLeniaCheckpoint(
                saturation=float(saturation if saturation is not None else _norm(index, len(items))),
                params=np.asarray(params, dtype=np.float32),
                step=_optional_int(step),
                loss=_optional_float(loss),
                source=source,
                index=index,
            )
        )
    return result


def _checkpoints_from_arrays(
    params: np.ndarray,
    loss: Any,
    steps: list[int] | np.ndarray | None,
    source: str,
    member_index: int,
) -> list[FlowLeniaCheckpoint]:
    if params.ndim == 1:
        return [FlowLeniaCheckpoint(saturation=0.0, params=params.astype(np.float32), source=source, index=0)]
    if params.ndim == 3:
        loss_array = None if loss is None else np.asarray(loss)
        if loss_array is not None and loss_array.shape[:2] == params.shape[:2]:
            member_ids = np.argmin(loss_array, axis=1)
            params = np.asarray([params[i, member_ids[i]] for i in range(params.shape[0])])
            loss = np.asarray([loss_array[i, member_ids[i]] for i in range(loss_array.shape[0])])
        else:
            params = params[:, int(member_index), :]
    if params.ndim != 2:
        raise ValueError(f"Unsupported Flow Lenia checkpoint params shape: {params.shape}")
    step_values = _step_values(params.shape[0], steps)
    loss_values = np.asarray(loss) if loss is not None else None
    result = []
    for index, row in enumerate(params):
        result.append(
            FlowLeniaCheckpoint(
                saturation=_normalized_step(index, params.shape[0], step_values),
                params=np.asarray(row, dtype=np.float32),
                step=int(step_values[index]) if step_values is not None else index,
                loss=_loss_at(loss_values, index),
                source=source,
                index=index,
            )
        )
    return result


def verify_sha256(path: str | os.PathLike[str], expected: str) -> None:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")


def _step_values(count: int, steps: list[int] | np.ndarray | None) -> np.ndarray | None:
    if steps is None or len(steps) == 0:
        return None
    values = np.asarray(steps, dtype=np.int64)
    if values.shape[0] != count:
        raise ValueError(f"checkpoint_steps length {values.shape[0]} does not match checkpoint count {count}")
    return values


def _normalized_step(index: int, count: int, steps: np.ndarray | None) -> float:
    if steps is None:
        return _norm(index, count)
    lo = float(steps[0])
    hi = float(steps[-1])
    return 0.0 if hi == lo else (float(steps[index]) - lo) / (hi - lo)


def _norm(index: int, count: int | None) -> float:
    if count is None:
        return 0.0
    return 0.0 if count <= 1 else float(index) / float(count - 1)


def _loss_at(loss: np.ndarray | None, index: int) -> float | None:
    if loss is None or loss.shape[0] <= index:
        return None
    return _optional_float(loss[index])


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(np.asarray(value).item())


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(np.asarray(value).item())


def _flow_lenia_artifact_ref(
    value: ArtifactRef | dict[str, Any] | None = None,
    *,
    uri: str | None = None,
    sha256: str | None = None,
) -> ArtifactRef | None:
    """Normalize Flow-Lenia-specific artifact inputs to the generic ref type.

    ``checkpoint_uri`` / ``checkpoint_sha256`` and ``manifest_uri`` are legacy
    convenience fields for this simulator only. Storage mechanics still live in
    ``ArtifactResolver``.
    """
    if value is not None:
        return coerce_artifact_ref(value)
    if uri is None:
        return None
    return ArtifactRef(uri=uri, sha256=sha256)
