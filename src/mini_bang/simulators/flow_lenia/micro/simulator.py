from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mini_bang.artifacts import (
    ArtifactRef,
    ArtifactResolver,
    coerce_artifact_ref,
    get_default_artifact_resolver,
)
from mini_bang.execution.payloads import (
    PAYLOAD_REF_SENTINEL,
    PayloadRef,
    PayloadStorageConfig,
    load_payload_ref,
    payload_to_transport_payload,
)
from mini_bang.execution.runtime import RuntimeServices
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase


class FlowLeniaMicroSimulator(MicroSimulatorBase):
    def __init__(
        self,
        *,
        params: Any,
        substrate_config: dict[str, Any],
        checkpoint_metadata: dict[str, Any],
        artifact_resolver: ArtifactResolver | None = None,
        runtime_services: RuntimeServices | None = None,
    ) -> None:
        self.substrate_config = dict(substrate_config)
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.artifact_resolver = artifact_resolver or get_default_artifact_resolver()
        self.runtime_services = runtime_services
        jax, jnp = _jax()
        from mini_bang.simulators.flow_lenia.substrate_factory import create_flow_lenia_substrate

        self.jax = jax
        self.jnp = jnp
        self.substrate = create_flow_lenia_substrate(**self.substrate_config)
        self.params = jnp.asarray(params)
        expected = int(np.asarray(self.substrate.default_params(jax.random.PRNGKey(0))).size)
        actual = int(np.asarray(self.params).size)
        if actual != expected:
            raise ValueError(
                f"Flow Lenia params length {actual} does not match substrate expectation {expected}"
            )
        self.params = self.params.reshape((expected,))

        def advance(state, keys):
            def step_fn(carry, key):
                return self.substrate.step_state(key, carry, self.params), None
            state_final, _ = jax.lax.scan(step_fn, state, keys)
            return state_final

        self._advance_state = jax.jit(advance)

    def sample(
        self,
        seed: int | None = None,
        rollout_steps: int | None = None,
        start_step: int = 0,
        end_step: int | None = None,
        sample_rate: int | None = None,
        snapshot_steps: Iterable[int] | None = None,
        sample_ranges: list[dict[str, Any]] | None = None,
        fields: Iterable[str] | None = ("A", "P"),
        metrics: Iterable[str] | None = None,
        include_rgb: bool = False,
        img_size: int | None = None,
        array_format: str = "numpy",
        externalize_fields: bool | None = None,
        field_payload_mode: str | None = None,
        field_payload_store: str | None = None,
        field_inline_threshold_bytes: int | None = None,
        init_state: Any | None = None,
        init_state_ref: ArtifactRef | dict[str, Any] | None = None,
        init_state_uri: str | None = None,
        init_state_path: str | os.PathLike[str] | None = None,
        init_state_sha256: str | None = None,
        rng_offset: int = 0,
        init_rng_mode: str = "rollout",
        return_final_state: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        seed_value = 0 if seed is None else int(seed)
        rng = self.jax.random.PRNGKey(seed_value)
        steps = _resolve_snapshot_steps(
            rollout_steps=rollout_steps if rollout_steps is not None else self.substrate.rollout_steps,
            start_step=start_step,
            end_step=end_step,
            sample_rate=sample_rate,
            snapshot_steps=snapshot_steps,
            sample_ranges=sample_ranges,
        )
        state = self._initial_state(
            rng=rng,
            init_state=init_state,
            init_state_ref=init_state_ref,
            init_state_uri=init_state_uri,
            init_state_path=init_state_path,
            init_state_sha256=init_state_sha256,
            init_rng_mode=init_rng_mode,
        )
        requested_fields = list(dict.fromkeys([str(field) for field in (fields or [])]))
        if include_rgb and "rgb" not in requested_fields:
            requested_fields.append("rgb")
        requested_metrics = list(dict.fromkeys([str(metric) for metric in (metrics or [])]))
        max_step = max(steps) if steps else 0
        rng_offset = int(rng_offset)
        total_keys = rng_offset + max_step
        keys = self.jax.random.split(rng, total_keys) if total_keys > 0 else None
        snapshots: dict[str, list[Any]] = {field: [] for field in requested_fields}
        metric_values: dict[str, list[Any]] = {metric: [] for metric in requested_metrics}
        current = 0
        for target in steps:
            if target < current:
                raise ValueError("snapshot steps must be sorted")
            if target > current:
                if keys is None:
                    raise ValueError("internal key schedule error")
                state = self._advance_state(state, keys[rng_offset + current:rng_offset + target])
                current = target
            for field in requested_fields:
                snapshots[field].append(self._capture_field(state, field, img_size))
            for metric in requested_metrics:
                metric_values[metric].append(self._capture_metric(state, metric))
        fields_payload, field_refs = _materialize_snapshots(
            snapshots,
            array_format=array_format,
            storage=_field_storage(
                externalize_fields=externalize_fields,
                field_payload_mode=field_payload_mode,
                field_payload_store=field_payload_store,
                field_inline_threshold_bytes=field_inline_threshold_bytes,
                runtime_services=self.runtime_services,
            ),
        )
        payload = {
            "steps": steps,
            "absolute_steps": [int(step) + rng_offset for step in steps],
            "fields": fields_payload,
            "checkpoint": dict(self.checkpoint_metadata),
            "substrate_config": dict(self.substrate_config),
            "rng": {
                "seed": seed_value,
                "rng_offset": rng_offset,
                "init_rng_mode": init_rng_mode,
            },
        }
        if field_refs:
            payload["field_payloads"] = field_refs
        if requested_metrics:
            payload["metrics"] = _materialize_metrics(metric_values)
        if return_final_state:
            payload["final_state"] = _to_numpy_tree(state, array_format=array_format)
        return payload

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": dict(self.checkpoint_metadata),
            "substrate_config": dict(self.substrate_config),
        }

    def _initial_state(
        self,
        *,
        rng,
        init_state: Any | None,
        init_state_ref: ArtifactRef | dict[str, Any] | None,
        init_state_uri: str | None,
        init_state_path: str | os.PathLike[str] | None,
        init_state_sha256: str | None,
        init_rng_mode: str,
    ):
        state_obj = init_state
        if state_obj is None and init_state_ref is not None:
            if isinstance(init_state_ref, ArtifactRef):
                ref_data = init_state_ref
            elif init_state_ref.get(PAYLOAD_REF_SENTINEL):
                ref_data = init_state_ref.get("ref")
            else:
                ref_data = init_state_ref
            if isinstance(ref_data, dict) and "store_kind" in ref_data:
                state_obj = load_payload_ref(PayloadRef.from_dict(ref_data))
            elif isinstance(ref_data, ArtifactRef) or (isinstance(ref_data, dict) and "uri" in ref_data):
                handle = self.artifact_resolver.materialize(coerce_artifact_ref(ref_data))
                state_obj = handle.local_path
            else:
                raise ValueError("init_state_ref must be a payload ref or artifact ref")
        if state_obj is None and init_state_uri is not None:
            handle = self.artifact_resolver.materialize(
                ArtifactRef(uri=init_state_uri, sha256=init_state_sha256)
            )
            state_obj = handle.local_path
        if state_obj is None and init_state_path is not None:
            state_obj = Path(init_state_path)
        if state_obj is not None:
            return _to_jax_tree(_load_state_object(state_obj), self.jnp)
        if init_rng_mode == "split":
            _, rng_init = self.jax.random.split(rng)
            return self.substrate.init_state(rng_init, self.params)
        if init_rng_mode != "rollout":
            raise ValueError("init_rng_mode must be 'rollout' or 'split'")
        return self.substrate.init_state(rng, self.params)

    def _capture_field(self, state, field: str, img_size: int | None):
        if field == "rgb":
            return self.substrate.render_state(state, self.params, img_size=img_size)
        if field not in state:
            raise ValueError(f"Flow Lenia state does not contain requested field {field!r}")
        return state[field]

    def _capture_metric(self, state, metric: str):
        A = state.get("A")
        P = state.get("P")
        Food = state.get("Food")
        if metric == "mass":
            return self.jnp.sum(A)
        if metric == "mass_channels":
            return self.jnp.sum(A, axis=(0, 1))
        if metric == "food_mass":
            return self.jnp.sum(Food) if Food is not None else self.jnp.asarray(0.0)
        if metric == "p_max_mean":
            return self.jnp.mean(self.jnp.max(P, axis=-1))
        if metric == "p_mean":
            return self.jnp.mean(P)
        if metric == "a_mean":
            return self.jnp.mean(A)
        if metric == "a_max":
            return self.jnp.max(A)
        raise ValueError(f"Unsupported Flow Lenia metric {metric!r}")


def _resolve_snapshot_steps(
    *,
    rollout_steps: int,
    start_step: int,
    end_step: int | None,
    sample_rate: int | None,
    snapshot_steps: Iterable[int] | None,
    sample_ranges: list[dict[str, Any]] | None,
) -> list[int]:
    if snapshot_steps is not None:
        return _validate_steps(sorted({int(step) for step in snapshot_steps}))
    if sample_ranges:
        steps: set[int] = set()
        for item in sample_ranges:
            start = int(item.get("start", item.get("start_step", 0)))
            end = int(item.get("end", item.get("end_step")))
            rate = int(item.get("sample_rate", item.get("rate", item.get("stride", 1))))
            steps.update(_range_steps(start, end, rate, include_end=bool(item.get("include_end", True))))
        return _validate_steps(sorted(steps))
    end = int(rollout_steps if end_step is None else end_step)
    start = int(start_step)
    if sample_rate is None:
        return _validate_steps([end])
    return _validate_steps(_range_steps(start, end, int(sample_rate), include_end=True))


def _range_steps(start: int, end: int, rate: int, *, include_end: bool) -> list[int]:
    if rate <= 0:
        raise ValueError("sample_rate must be positive")
    if end < start:
        raise ValueError("end_step must be >= start_step")
    steps = list(range(start, end + 1, rate))
    if include_end and (not steps or steps[-1] != end):
        steps.append(end)
    return steps


def _validate_steps(steps: list[int]) -> list[int]:
    if not steps:
        raise ValueError("At least one Flow Lenia snapshot step is required")
    if steps[0] < 0:
        raise ValueError("Flow Lenia snapshot steps must be non-negative")
    return steps


def _load_state_object(value: Any) -> Any:
    if isinstance(value, Path) or isinstance(value, os.PathLike):
        path = Path(value)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        with path.open("rb") as fh:
            return pickle.load(fh)
    return value


def _materialize_snapshots(
    snapshots: dict[str, list[Any]],
    *,
    array_format: str,
    storage: PayloadStorageConfig | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = {}
    refs = {}
    for field, values in snapshots.items():
        arr = np.stack([np.asarray(_device_get(value)) for value in values], axis=0)
        metadata = {
            "simulator_id": "flow_lenia",
            "field": field,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        if storage is not None:
            inline, ref = storage.externalize(arr, key=f"flow_lenia_{field}", metadata=metadata)
            if ref is not None:
                out[field] = payload_to_transport_payload(None, ref)
                refs[field] = {**metadata, "ref": ref.to_dict()}
                continue
            arr = inline
        out[field] = arr.tolist() if array_format == "list" and hasattr(arr, "tolist") else arr
    return out, refs


def _materialize_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    out = {}
    for metric, values in metrics.items():
        arr = np.stack([np.asarray(_device_get(value)) for value in values], axis=0)
        out[metric] = arr.tolist()
    return out


def _field_storage(
    *,
    externalize_fields: bool | None,
    field_payload_mode: str | None,
    field_payload_store: str | None,
    field_inline_threshold_bytes: int | None,
    runtime_services: RuntimeServices | None,
) -> PayloadStorageConfig | None:
    if externalize_fields is False:
        return None
    requested_mode = _normalize_store_kind(field_payload_mode)
    if requested_mode in {"inline", "none", "off"}:
        return None
    store = runtime_services.payload_store if runtime_services is not None else None
    if store is None:
        if externalize_fields is True:
            raise RuntimeError("Flow Lenia field externalization requires RuntimeServices.payload_store")
        return None
    requested_store = _normalize_store_kind(field_payload_store) if field_payload_store else None
    actual_store = _normalize_store_kind(str(getattr(store, "store_kind", "")))
    if requested_mode and requested_mode != "auto" and requested_mode != actual_store:
        raise ValueError(
            f"field_payload_mode={field_payload_mode!r} does not match runtime payload store {actual_store!r}"
        )
    if requested_store and requested_store != actual_store:
        raise ValueError(
            f"field_payload_store={field_payload_store!r} does not match runtime payload store {actual_store!r}"
        )
    mode = "auto" if requested_mode == "auto" else actual_store
    return PayloadStorageConfig(
        mode=mode,
        store=store,
        inline_threshold_bytes=field_inline_threshold_bytes,
    )


def _normalize_store_kind(kind: str | None) -> str:
    normalized = str(kind or "").strip().lower().replace("-", "_")
    aliases = {
        "fs": "filesystem",
        "file": "filesystem",
        "pvc": "filesystem",
        "k8s": "filesystem",
        "filesystem_pvc": "filesystem",
        "in_memory": "memory",
        "in-memory": "memory",
        "aws_s3": "s3",
        "object": "s3",
        "object_store": "s3",
        "minio": "s3",
    }
    return aliases.get(normalized, normalized)


def _to_numpy_tree(value: Any, *, array_format: str) -> Any:
    if isinstance(value, dict):
        return {key: _to_numpy_tree(val, array_format=array_format) for key, val in value.items()}
    arr = np.asarray(_device_get(value))
    return arr.tolist() if array_format == "list" else arr


def _to_jax_tree(value: Any, jnp) -> Any:
    if isinstance(value, dict):
        return {key: _to_jax_tree(val, jnp) for key, val in value.items()}
    return jnp.asarray(value)


def _device_get(value: Any) -> Any:
    try:
        import jax
    except ImportError:
        return value
    return jax.device_get(value)


def _jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("flow_lenia requires the optional JAX dependencies") from exc
    return jax, jnp
