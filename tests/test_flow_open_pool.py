"""The open Flow Lenia observation pool (simulators/flow_lenia_pool/open_pool.py).

A synthetic open pool on a visible presence window runs a full observation through
the task API without the hidden case bindings. The recorded pool opens bindings
of hidden cases from the gold root only for scorer-side work.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from mini_bang.artifacts import ArtifactResolver
from mini_bang.backends.local import LocalBackend
from mini_bang.execution.payloads import load_payload_ref, payload_from_transport_payload
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.framework.runtime import RunManager
from mini_bang.simulators.flow_lenia_pool import factory
from mini_bang.simulators.flow_lenia_pool.indexed_frames import write_shard
from mini_bang.simulators.flow_lenia_pool.open_pool import OPEN_SCHEMA, OpenPoolObservations, window_steps
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import file_ref
from tests.flow_lenia_pool_support import FIXTURE_TRAJECTORY, visible_cases, write_pool

def synthetic_pool(root: Path):
    case = visible_cases()[0]
    first = window_steps(case["observation_window"], 100)[0]
    window = [first, first + 300]
    steps = window_steps(window, 100)
    rng = np.random.default_rng(5)
    arrays = {"A": rng.normal(size=(4, 3, 3, 3)).astype(np.float16),
              "Food": rng.normal(size=(4, 3, 3)).astype(np.float32),
              "rgb": rng.integers(0, 255, size=(4, 3, 3, 3), dtype=np.uint8)}
    blocks = []
    for part in (steps[:2], steps[2:]):
        fields = {}
        for field, values in arrays.items():
            path = root / f"{part[0]}" / f"{field}.frames.zip"
            write_shard(path, part, values[steps.index(part[0]):steps.index(part[0]) + len(part)])
            fields[field] = file_ref(path)
        blocks.append({"steps": part, "fields": fields})
    case_entry = {"observation_window": window, "blocks": blocks}
    manifest = {"schema": OPEN_SCHEMA, "step_multiple": 100, "fields": list(arrays), "cases": {case["case_id"]: case_entry}}
    path = root / "open.json"
    path.write_text(json.dumps(manifest))
    return path, case, steps, arrays


def npz(observation):
    if hasattr(observation, "load"):
        body = observation.load()
    else:
        value, ref = payload_from_transport_payload(observation)
        body = load_payload_ref(ref) if ref is not None else value
    return np.load(io.BytesIO(body), allow_pickle=False)


def test_task_api_serves_the_open_pool_without_hidden_bindings(tmp_path, monkeypatch):
    path, case, steps, arrays = synthetic_pool(tmp_path / "pool")
    monkeypatch.setattr(factory, "case_bindings", lambda: pytest.fail("open pool read the hidden bindings"))
    monkeypatch.setenv("MINI_BANG_ARTIFACT_CACHE_DIR", str(tmp_path / "artifact-cache"))
    monkeypatch.setenv("MINI_BANG_FLOW_POOL_REF", path.as_uri())
    monkeypatch.setenv("MINI_BANG_FLOW_OBSERVATION_MODE", "checkpoints")
    monkeypatch.setenv("MINI_BANG_PAYLOAD_MODE", "filesystem")
    monkeypatch.setenv("MINI_BANG_PAYLOAD_DIR", str(tmp_path / "payloads"))
    session = RunManager(backend=LocalBackend()).prepare("flow_lenia/presence")
    session.instance_spec.visible_data["stabilization_annotations_complete"] = True
    response = session.environment.api.generate_samples(case_id=case["case_id"], start_step=steps[1],
                                                        end_step=steps[3], step=100, fields=["A", "rgb"])
    with npz(response["observation"]) as data:
        assert data["steps"].tolist() == steps[1:]
        assert data["A"].tobytes() == arrays["A"][1:].tobytes() and data["rgb"].dtype == np.uint8
        assert data["rgb"].tobytes() == arrays["rgb"][1:].tobytes()


def test_open_reader_rejects_steps_outside_the_recorded_window(tmp_path):
    path, case, steps, _ = synthetic_pool(tmp_path / "pool")
    reader = OpenPoolObservations(json.loads(path.read_text()), case["case_id"], ArtifactResolver(), mode="video")
    with pytest.raises(ValueError):
        reader.observe([steps[0], steps[-1] + 100], ["rgb"])
    with pytest.raises(ValueError):
        reader.observe([steps[0]], ["A"])   # video mode is RGB only
    with pytest.raises(ValueError):
        OpenPoolObservations(json.loads(path.read_text()), "unknown-case", ArtifactResolver(), mode="checkpoints")
    with pytest.raises(ValueError):
        OpenPoolObservations(json.loads(path.read_text()), case["case_id"], ArtifactResolver(), mode="replay")


def test_gold_root_bindings_open_only_for_scoring_authority(tmp_path, monkeypatch):
    """Bindings of hidden cases under MINI_BANG_GOLD_ROOT never reach agent sessions."""
    pool, _ = write_pool(tmp_path / "pool")
    hidden = tmp_path / "gold" / "flow_lenia_common"
    hidden.mkdir(parents=True)
    (hidden / "bindings.json").write_text(json.dumps({"hidden-case": {"trajectory": FIXTURE_TRAJECTORY}}))
    monkeypatch.setenv("MINI_BANG_GOLD_ROOT", str(tmp_path / "gold"))
    for authority in (None, "agent", "scoring"):
        services = RuntimeServices.from_context(RuntimeContext(security={"authority": authority}))
        build = lambda: factory.create_pool_macro(pool_ref=file_ref(pool), case_id="hidden-case", mode="checkpoints",
                                                  artifact_resolver=services.artifact_resolver,
                                                  runtime_services=services)
        if authority == "scoring":
            assert build().case_id == "hidden-case"
        else:
            with pytest.raises(ValueError, match="Unknown observation case"):
                build()

