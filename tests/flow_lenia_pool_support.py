"""A tiny observation pool on the first window the presence task shows agents.

The window comes from the visible presence instance, so no hidden annotation is
read. The simulator maps a case id to its trajectory through the hidden case
bindings; use_synthetic_bindings replaces them in this process.
"""
from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import tempfile

import numpy as np

from mini_bang.execution.models import SimulationRequest
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import file_ref
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.simulators.flow_lenia_pool import factory
from mini_bang.tasks.loader import TaskLoader

_TEMPORARIES = []
FIXTURE_TRAJECTORY = "traj_00000"


def visible_cases() -> list[dict]:
    spec = TaskLoader().get_spec("flow_lenia/presence")
    return BenchmarkArtifactStore().load_instance(spec).visible_data["cases"]


def use_synthetic_bindings(monkeypatch) -> None:
    """Bind every visible presence window to the fixture trajectory, in this process only."""
    bindings = {case["case_id"]: {"trajectory": FIXTURE_TRAJECTORY} for case in visible_cases()}
    monkeypatch.setattr(factory, "case_bindings", lambda: bindings)


def hidden_bindings_published() -> bool:
    """Spawned simulation workers can only read the real case bindings from gold."""
    package, name = factory.BINDINGS
    return files(package).joinpath(name).is_file()


def write_pool(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    case = visible_cases()[0]
    steps = [case["observation_window"][0] + offset for offset in (0, 100, 200)]
    data = root / "states.npz"
    np.savez_compressed(data, steps=np.array(steps), A=np.ones((3, 2, 2, 1), dtype=np.float32),
                        P=np.full((3, 2, 2, 1), 2, dtype=np.float32), Food=np.zeros((3, 2, 2, 1), dtype=np.float32))
    manifest = {"schema": "mini_bang.flow_pool.v1", "trajectories": {f"traj_{i:05d}": {
        "last_step": 500000, "observations": [{"steps": steps, "ref": file_ref(data)}]}
        for i in range(100)}}
    path = root / "pool.json"
    path.write_text(json.dumps(manifest))
    return path, case


def pool_fixture(root: Path, monkeypatch):
    use_synthetic_bindings(monkeypatch)
    return write_pool(root)


def pool_smoke_request(*, runs=2):
    """Smoke request on the pool; the caller provides the bindings (synthetic or real)."""
    temporary = tempfile.TemporaryDirectory(prefix="pool-smoke-")
    _TEMPORARIES.append(temporary)
    path, case = write_pool(Path(temporary.name))
    start = case["observation_window"][0]
    return SimulationRequest(simulator_id="flow_lenia_pool", saturation=0, runs=runs,
        macro_params={"pool_ref": file_ref(path), "case_id": case["case_id"], "mode": "checkpoints"},
        sample_params={"start_step": start, "end_step": start + 200, "step": 100, "fields": ["A", "P", "Food"]})
