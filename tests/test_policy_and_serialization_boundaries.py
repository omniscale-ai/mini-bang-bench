from __future__ import annotations

from pathlib import Path

import pytest

from mini_bang.artifacts import ArtifactRef
from mini_bang.artifacts.resolver import ArtifactResolver
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.payload_stores.base import PayloadRef
from mini_bang.execution.payload_stores.codecs import deserialize_payload
from mini_bang.execution.payloads import FileSystemPayloadStore
from mini_bang.execution.planner import SimulationPlanner
from mini_bang.framework.policy import CostBudgetPolicy, FieldPolicy, GranularityPolicy, SeedPolicy, WindowPolicy


def test_artifact_ref_representations_hash_equally(tmp_path) -> None:
    ref = ArtifactRef(uri=(tmp_path / "a.bin").as_uri(), sha256="1" * 64)
    dict_ref = {"sha256": "1" * 64, "uri": ref.uri}
    first = SimulationRequest("artifact_echo_test", 1, macro_params={"artifact_ref": ref})
    second = SimulationRequest("artifact_echo_test", 1, macro_params={"artifact_ref": dict_ref})

    assert first.request_hash == second.request_hash


def test_arbitrary_request_objects_are_rejected_before_execution() -> None:
    class NotSerializable:
        pass

    with pytest.raises(TypeError):
        SimulationRequest("raf", 1, macro_params={"bad": NotSerializable()})


def test_no_repr_fallback_in_request_hash(tmp_path) -> None:
    with pytest.raises(TypeError):
        SimulationRequest("raf", 1, sample_params={"path": Path("not-a-contract")}).request_hash


def test_seed_modes_are_explicit() -> None:
    planner = SimulationPlanner()
    supplied = SimulationRequest(
        "raf",
        1,
        sample_params={"seed": 7},
        seed_mode="agent_supplied",
    )
    derived = SimulationRequest("raf", 1, sample_params={"seed": 7})
    fixed = SimulationRequest("raf", 1, sample_params={"seed": 9}, seed_mode="fixed_by_instance")

    assert planner.plan(supplied)[0].sample_params["seed"] == 7
    assert planner.plan(fixed)[0].sample_params["seed"] == 9
    assert planner.plan(derived)[0].sample_params["seed"] != 7
    assert planner.plan(supplied)[0].seed_mode == "agent_supplied"
    assert planner.plan(supplied)[0].seed_source == "agent"
    assert planner.plan(fixed)[0].seed_source == "instance"
    with pytest.raises(ValueError, match="agent_supplied"):
        planner.plan(SimulationRequest("raf", 1, seed_mode="agent_supplied"))
    with pytest.raises(ValueError, match="fixed_by_instance"):
        planner.plan(SimulationRequest("raf", 1, seed_mode="fixed_by_instance"))
    SeedPolicy(fixed_seed=9).validate(fixed)
    with pytest.raises(ValueError, match="overridden"):
        SeedPolicy(fixed_seed=9).validate(
            SimulationRequest("raf", 1, sample_params={"seed": 10}, seed_mode="fixed_by_instance")
        )


def test_pickle_payload_rejected_unless_explicitly_allowed(tmp_path) -> None:
    store = FileSystemPayloadStore(tmp_path / "payloads")
    with pytest.raises(ValueError, match="pickle"):
        store.put({"bad": object()})

    trusted = FileSystemPayloadStore(tmp_path / "trusted", allow_pickle_payloads=True)
    ref = trusted.put({"bad": object()})
    body = trusted.get_bytes(ref)
    with pytest.raises(ValueError, match="pickle"):
        deserialize_payload(body, codec=ref.codec)
    assert trusted.get(ref)


def test_filesystem_payload_store_rejects_path_escape(tmp_path) -> None:
    root = tmp_path / "payloads"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    store = FileSystemPayloadStore(root)
    ref = PayloadRef(
        uri=outside.as_uri(),
        store_kind="filesystem",
        codec="json",
        media_type="application/json",
    )

    with pytest.raises(PermissionError):
        store.get(ref)


def test_filesystem_payload_store_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "payloads"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = root / "link.json"
    link.symlink_to(outside)
    store = FileSystemPayloadStore(root)
    ref = PayloadRef(
        uri=link.as_uri(),
        store_kind="filesystem",
        codec="json",
        media_type="application/json",
    )

    with pytest.raises(PermissionError):
        store.get(ref)


def test_official_artifact_resolver_restricts_schemes_roots_and_s3(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    blocked = tmp_path / "blocked"
    allowed.mkdir()
    blocked.mkdir()
    resolver = ArtifactResolver(
        allowed_schemes=["file", "s3"],
        allowed_file_roots=[allowed],
        allowed_s3_prefixes=["s3://bucket/allowed/"],
        network_enabled=False,
        gdrive_enabled=False,
    )

    with pytest.raises(PermissionError):
        resolver.stat(ArtifactRef(uri=(blocked / "x").as_uri()))
    with pytest.raises(PermissionError):
        resolver.stat(ArtifactRef(uri="http://example.com/x"))
    with pytest.raises(PermissionError):
        resolver.stat(ArtifactRef(uri="s3://bucket/blocked/x"))


def test_cost_policy_checks_enforce_limits() -> None:
    class Session:
        budget_counters: dict[str, int] = {}

    session = Session()
    policy = CostBudgetPolicy(limits={"raw_step_count": 10, "estimated_bytes": 20})
    policy.initialize_session(session)
    request = SimulationRequest("sim", 1, runs=2, sample_params={"rollout_steps": 6})
    with pytest.raises(RuntimeError, match="raw_step_count"):
        policy.validate_request(session, request)

    ok = SimulationRequest("sim", 1, sample_params={"estimated_bytes": 10})
    policy.validate_request(session, ok)
    policy.consume_request(session, ok)
    assert policy.capabilities(session)["remaining_costs"]["estimated_bytes"] == 10
    WindowPolicy(max_raw_steps=5).validate(SimulationRequest("sim", 1, sample_params={"end_step": 5}))
    FieldPolicy(allowed_fields={"A"}, max_fields=1).validate(SimulationRequest("sim", 1, sample_params={"fields": ["A"]}))
    GranularityPolicy(step_multiple=5).validate(SimulationRequest("sim", 1, sample_params={"rollout_steps": 10}))
