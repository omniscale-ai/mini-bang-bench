"""Numerical acceptance of replay mode against the original saved states."""
from __future__ import annotations

from pathlib import Path
import platform

import numpy as np

from mini_bang.artifacts import ArtifactResolver
from mini_bang.client.cli.io import read_json, write_json
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import validate_manifest
from mini_bang.simulators.flow_lenia_pool.observations import PoolObservations


def verify_replay(path: Path, *, output: Path, require_gpu=False, all_anchors=False):
    """Numerical acceptance test against original saved states, never a detector."""
    import jax
    from mini_bang.simulators.flow_lenia.substrate_factory import create_flow_lenia_substrate
    from mini_bang.simulators.flow_lenia_pool.replay import load_cursor

    if output.exists():
        raise FileExistsError("Verification output must be new")
    manifest = read_json(path)
    validate_manifest(manifest, mode="replay")
    devices = jax.devices()
    if require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("GPU verification requested but JAX cannot see a GPU")
    results = []
    resolver = ArtifactResolver()
    for name, trajectory in manifest["trajectories"].items():
        reader = PoolObservations(manifest, name, resolver, mode="replay")
        substrate = create_flow_lenia_substrate(**trajectory["substrate_config"])
        with np.load(reader.materialize(trajectory["params_ref"]), allow_pickle=False) as data:
            params = data["params"]
        restarts = trajectory["restarts"]
        indices = range(len(restarts) - 1) if all_anchors else sorted({0, (len(restarts) - 1) // 2, len(restarts) - 2})
        checks = []
        compiled_advance = None
        for index in indices:
            start, target = restarts[index:index + 2]
            cursor = load_cursor(reader, start, substrate, params, compiled_advance=compiled_advance)
            compiled_advance = cursor.advance
            state = cursor.at(target["step"])
            errors = {}
            with np.load(reader.materialize(target["ref"]), allow_pickle=False) as expected:
                for field in ("A", "P", "Food", "t", "mass_cycle_start"):
                    actual, wanted = np.asarray(state[field]), expected["state__" + field]
                    if actual.shape != wanted.shape or not np.isfinite(actual).all() or not np.isfinite(wanted).all():
                        raise ValueError(f"{name}: invalid numerical state at {target['step']}")
                    error = float(np.max(np.abs(actual - wanted))) if actual.size else 0.0
                    errors[field] = error
                    if not np.allclose(actual, wanted, rtol=1e-5, atol=1e-6):
                        raise ValueError(f"{name}: replay diverges at {target['step']}, {field}, max_abs={error}; use dense original observations")
            checks.append({"from": start["step"], "to": target["step"], "max_abs_error": errors})
        trajectory["replay_verified"] = True
        results.append({"trajectory": name, "checks": checks})
    manifest["replay_verification"] = {"jax": jax.__version__, "platform": platform.platform(),
        "devices": [str(device) for device in devices], "rtol": 1e-5, "atol": 1e-6,
        "coverage": "all consecutive restart pairs" if all_anchors else "three consecutive restart pairs per trajectory",
        "checks": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)
    return {"verified_manifest": str(output), "trajectories": len(results), "all_anchors": all_anchors,
            "devices": manifest["replay_verification"]["devices"]}
