from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from mini_bang.artifacts import ArtifactRef
from mini_bang.artifacts.resolver import ArtifactResolver
from mini_bang.execution.engines import ProcessPoolEngine, SequentialEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.simulators import registry as simulator_registry


SIM_PACKAGE = "mini_bang.testing.artifact_echo_simulator"
SIM_ID = "artifact_echo_test"
FIELD_PACKAGE = "mini_bang.testing.field_payload_simulator"
FIELD_ID = "field_payload_test"


@pytest.fixture(autouse=True)
def _cleanup_echo_registration():
    yield
    simulator_registry._REGISTRY._specs.pop(SIM_ID, None)
    simulator_registry._REGISTRY._specs.pop(FIELD_ID, None)


def test_runtime_services_from_context_does_not_mutate_process_env(tmp_path, monkeypatch) -> None:
    env_root = tmp_path / "env-payloads"
    monkeypatch.setenv("MINI_BANG_PAYLOAD_DIR", str(env_root))
    before = dict(os.environ)
    context_a = RuntimeContext(
        payload={"kind": "filesystem", "root": str(tmp_path / "payload-a")},
        env={"MINI_BANG_PAYLOAD_DIR": str(tmp_path / "ignored-a")},
    )
    context_b = RuntimeContext(
        payload={"kind": "filesystem", "root": str(tmp_path / "payload-b")},
        env={"MINI_BANG_PAYLOAD_DIR": str(tmp_path / "ignored-b")},
    )

    services_a = RuntimeServices.from_context(context_a)
    services_b = RuntimeServices.from_context(context_b)

    assert os.environ == before
    assert services_a.payload_store is not None
    assert services_b.payload_store is not None
    assert services_a.payload_store.describe()["root"] == str((tmp_path / "payload-a").resolve())
    assert services_b.payload_store.describe()["root"] == str((tmp_path / "payload-b").resolve())


def test_field_payload_externalization_uses_runtime_context_not_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", FIELD_PACKAGE)
    monkeypatch.setenv("MINI_BANG_PAYLOAD_DIR", str(tmp_path / "env-payloads"))
    _register_field_payload_simulator_parent()
    context = RuntimeContext(payload={"kind": "filesystem", "root": str(tmp_path / "context-payloads")})

    result = SimulationExecutor(engine=SequentialEngine(), context=context).run(
        SimulationRequest(FIELD_ID, 1)
    )

    ref = _field_ref(result.payload)
    assert str((tmp_path / "context-payloads").resolve()) in ref["uri"]
    assert str(tmp_path / "env-payloads") not in ref["uri"]
    assert _field_metrics(result.payload) == {"mass": [6]}


def test_process_pool_field_payload_externalization_uses_runtime_context(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", FIELD_PACKAGE)
    monkeypatch.setenv("MINI_BANG_PAYLOAD_DIR", str(tmp_path / "env-payloads"))
    _register_field_payload_simulator_parent()
    context = RuntimeContext(payload={"kind": "filesystem", "root": str(tmp_path / "context-payloads")})
    engine = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    try:
        try:
            result = SimulationExecutor(engine=engine, context=context).run(
                SimulationRequest(FIELD_ID, 1)
            )
        except PermissionError as exc:
            if _process_pool_unavailable(exc):
                pytest.skip(str(exc))
            raise
    finally:
        engine.close()

    ref = _field_ref(result.payload)
    assert str((tmp_path / "context-payloads").resolve()) in ref["uri"]
    assert str(tmp_path / "env-payloads") not in ref["uri"]
    assert _field_metrics(result.payload) == {"mass": [6]}


def test_process_pool_uses_runtime_context_artifact_restrictions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", SIM_PACKAGE)
    _register_echo_simulator_parent()
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    source = outside / "payload.bin"
    source.write_bytes(b"blocked")
    context = _context(tmp_path, allowed_roots=[allowed])
    engine = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    try:
        try:
            SimulationExecutor(engine=engine, context=context).run(_request(source))
        except PermissionError as exc:
            if _process_pool_unavailable(exc):
                pytest.skip(str(exc))
        else:
            pytest.fail("process pool did not enforce RuntimeContext artifact roots")
    finally:
        engine.close()


def test_sequential_and_process_pool_materialize_same_artifact_from_context(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", SIM_PACKAGE)
    _register_echo_simulator_parent()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "payload.bin"
    source.write_bytes(b"same-body")
    context = _context(tmp_path, allowed_roots=[allowed])
    sequential = SimulationExecutor(engine=SequentialEngine(), context=context)
    process = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    try:
        try:
            process_result = SimulationExecutor(engine=process, context=context).run(_request(source)).payload
        except PermissionError as exc:
            if _process_pool_unavailable(exc):
                pytest.skip(str(exc))
            raise
    finally:
        process.close()

    sequential_result = sequential.run(_request(source)).payload
    assert sequential_result["trajectories"] == process_result["trajectories"]


def test_process_pool_reconstructs_importable_artifact_store_from_context(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", SIM_PACKAGE)
    _register_echo_simulator_parent()
    context = RuntimeContext(
        artifact={
            "cache_dir": str(tmp_path / "cache"),
            "allowed_schemes": ["count"],
            "stores": [
                {
                    "class": "mini_bang.testing.stores:StaticArtifactStore",
                    "kwargs": {"body": "mocked"},
                }
            ],
        }
    )
    request = SimulationRequest(
        simulator_id=SIM_ID,
        saturation=1,
        macro_params={"artifact_ref": ArtifactRef(uri="count://item").to_dict()},
    )
    engine = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    try:
        try:
            result = SimulationExecutor(engine=engine, context=context).run(request).payload
        except PermissionError as exc:
            if _process_pool_unavailable(exc):
                pytest.skip(str(exc))
            raise
    finally:
        engine.close()

    assert result["trajectories"][0]["size_bytes"] == len(b"mocked")


def test_sequential_engine_accepts_live_artifact_resolver(tmp_path) -> None:
    _register_echo_simulator_parent()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "payload.bin"
    source.write_bytes(b"ok")
    resolver = ArtifactResolver(allowed_schemes=["file"], allowed_file_roots=[allowed])

    result = SimulationExecutor(
        engine=SequentialEngine(),
        artifact_resolver=resolver,
    ).run(_request(source))

    assert result.payload["trajectories"][0]["size_bytes"] == 2


def test_process_pool_rejects_live_artifact_resolver_without_runtime_context(tmp_path) -> None:
    _register_echo_simulator_parent()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "payload.bin"
    source.write_bytes(b"ok")
    resolver = ArtifactResolver(allowed_schemes=["file"], allowed_file_roots=[allowed])
    engine = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    try:
        with pytest.raises(ValueError, match="serializable RuntimeContext"):
            SimulationExecutor(engine=engine, artifact_resolver=resolver).run(_request(source))
    finally:
        engine.close()


def test_persistent_process_pool_separates_runtime_contexts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", SIM_PACKAGE)
    _register_echo_simulator_parent()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    file_a = root_a / "payload.bin"
    file_b = root_b / "payload.bin"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")
    engine = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    executor = SimulationExecutor(engine=engine, context=_context(tmp_path / "cache-a", allowed_roots=[root_a]))
    try:
        try:
            assert executor.run(_request(file_a)).payload["trajectories"][0]["size_bytes"] == 1
        except PermissionError as exc:
            if _process_pool_unavailable(exc):
                pytest.skip(str(exc))
            raise
        executor.set_runtime_context(_context(tmp_path / "cache-b", allowed_roots=[root_b]))
        assert executor.run(_request(file_b)).payload["trajectories"][0]["size_bytes"] == 1
        with pytest.raises(PermissionError):
            executor.run(_request(file_a))
    finally:
        engine.close()


def _request(path: Path) -> SimulationRequest:
    return SimulationRequest(
        simulator_id=SIM_ID,
        saturation=1,
        macro_params={"artifact_ref": ArtifactRef(uri=path.as_uri()).to_dict()},
    )


def _context(root: Path, *, allowed_roots: list[Path]) -> RuntimeContext:
    return RuntimeContext(
        artifact={
            "cache_dir": str(root / "cache"),
            "allowed_schemes": ["file"],
            "allowed_file_roots": [str(path) for path in allowed_roots],
        },
        security={"network_enabled": False, "gdrive_enabled": False},
    )


def _register_echo_simulator_parent() -> None:
    simulator_registry._REGISTRY._specs.pop(SIM_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=SIM_ID,
            description="test-only artifact echo simulator",
            factory_path=f"{SIM_PACKAGE}.factory:create_macro",
            grouped_sampling_safe=True,
        )
    )


def _register_field_payload_simulator_parent() -> None:
    simulator_registry._REGISTRY._specs.pop(FIELD_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=FIELD_ID,
            description="test-only runtime payload simulator",
            factory_path=f"{FIELD_PACKAGE}.factory:create_macro",
            grouped_sampling_safe=True,
        )
    )


def _field_ref(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["trajectories"][0]["fields"]["field"]["ref"]


def _field_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["trajectories"][0]["metrics"]


def _process_pool_unavailable(exc: PermissionError) -> bool:
    return "Operation not permitted" in str(exc)
