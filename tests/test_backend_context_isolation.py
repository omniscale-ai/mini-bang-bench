from __future__ import annotations

import http.server
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest

from mini_bang.artifacts import ArtifactRef
from mini_bang.backends.cache import CachingSimulationBackend, FileSystemSimulationCache
from mini_bang.backends.local import LocalBackend
from mini_bang.backends.mcp import MCPBackend
from mini_bang.client.cli.official import OfficialRunServer
from mini_bang.execution.engines import SequentialEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.planner import SimulationPlanner
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.framework.policy import BaseTaskPolicy
from mini_bang.framework.task import InstanceSpec, RunSession, TaskEnvironment, TaskSpec
from mini_bang.simulators import registry as simulator_registry


MARKER_PACKAGE = "mini_bang.testing.context_marker_simulator"
MARKER_ID = "context_marker_test"


@pytest.fixture(autouse=True)
def _cleanup_marker_registration():
    yield
    simulator_registry._REGISTRY._specs.pop(MARKER_ID, None)


def test_local_backend_uses_context_per_call_without_mutating_default(tmp_path) -> None:
    _register_marker_simulator_parent()
    default_root, agent_root, scoring_root = _marker_roots(tmp_path, "default", "agent", "scoring")
    backend = LocalBackend(
        executor=SimulationExecutor(engine=SequentialEngine()),
        runtime_context=_context(tmp_path / "default-cache", default_root),
    )
    backend.set_runtime_context = _forbidden_set_runtime_context  # type: ignore[method-assign]

    agent = backend.run(_marker_request(agent_root), context=_context(tmp_path / "agent-cache", agent_root))
    scoring = backend.run(_marker_request(scoring_root), context=_context(tmp_path / "scoring-cache", scoring_root))
    default = backend.run(_marker_request(default_root))

    assert _marker(agent) == "agent"
    assert _marker(scoring) == "scoring"
    assert _marker(default) == "default"
    with pytest.raises(PermissionError):
        backend.run(_marker_request(agent_root))


def test_concurrent_local_backend_context_isolation(tmp_path) -> None:
    _register_marker_simulator_parent()
    agent_root, scoring_root = _marker_roots(tmp_path, "agent", "scoring")
    backend = LocalBackend(
        executor=SimulationExecutor(
            engine=SequentialEngine(),
            planner=_BarrierPlanner(parties=2),
        )
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        agent_future = pool.submit(
            _run_backend_marker,
            backend,
            agent_root,
            _context(tmp_path / "agent-cache", agent_root),
        )
        scoring_future = pool.submit(
            _run_backend_marker,
            backend,
            scoring_root,
            _context(tmp_path / "scoring-cache", scoring_root),
        )
        assert agent_future.result(timeout=10) == "agent"
        assert scoring_future.result(timeout=10) == "scoring"


def test_run_session_owns_context_with_shared_backend(tmp_path) -> None:
    _register_marker_simulator_parent()
    agent_root, scoring_root = _marker_roots(tmp_path, "agent", "scoring")
    backend = LocalBackend(
        executor=SimulationExecutor(
            engine=SequentialEngine(),
            planner=_BarrierPlanner(parties=2),
        )
    )
    session_a = _session(backend, agent_root, _context(tmp_path / "agent-cache", agent_root))
    session_b = _session(backend, scoring_root, _context(tmp_path / "scoring-cache", scoring_root))

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(_run_session_marker, session_a, agent_root)
        b = pool.submit(_run_session_marker, session_b, scoring_root)
        assert a.result(timeout=10) == "agent"
        assert b.result(timeout=10) == "scoring"


def test_official_multi_session_uses_session_context_per_call(tmp_path) -> None:
    _register_marker_simulator_parent()
    agent_root, scoring_root = _marker_roots(tmp_path, "agent", "scoring")
    backend = LocalBackend(
        executor=SimulationExecutor(
            engine=SequentialEngine(),
            planner=_BarrierPlanner(parties=2),
        )
    )
    backend.set_runtime_context = _forbidden_set_runtime_context  # type: ignore[method-assign]
    session_a = _session(backend, agent_root, _context(tmp_path / "agent-cache", agent_root))
    session_b = _session(backend, scoring_root, _context(tmp_path / "scoring-cache", scoring_root))
    server = OfficialRunServer(mode="multi_session_official", admin_token="admin")
    server.register_session(session_a, token="token-a")
    server.register_session(session_b, token="token-b")

    with _server(server) as base, ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(_post_json, f"{base}/v1/generate_samples", {"session_id": session_a.run_id}, "token-a")
        b = pool.submit(_post_json, f"{base}/v1/generate_samples", {"session_id": session_b.run_id}, "token-b")
        assert a.result(timeout=10)["payload"]["trajectories"][0]["marker"] == "agent"
        assert b.result(timeout=10)["payload"]["trajectories"][0]["marker"] == "scoring"


def test_caching_backend_forwards_explicit_context(tmp_path) -> None:
    request = SimulationRequest("context-recording", 1)
    backend = _RecordingBackend()
    cached = CachingSimulationBackend(backend, FileSystemSimulationCache(tmp_path / "cache"))

    a = cached.run_batch([request], context=RuntimeContext(security={"marker": "agent"}))
    b = cached.run_batch([request], context=RuntimeContext(security={"marker": "scoring"}))

    assert _marker(a[0]) == "agent"
    assert _marker(b[0]) == "scoring"
    assert backend.markers == ["agent", "scoring"]


def test_mcp_backend_sends_per_call_context_and_rejects_live_services() -> None:
    client = _FakeMCPClient()
    backend = MCPBackend(client=client)
    try:
        default = RuntimeContext(run_id="run", task_id="task", instance_id="case")
        assert backend.run(SimulationRequest("sim", 1), context=default).payload["ok"] is True
        restricted = RuntimeContext(artifact={"allowed_schemes": ["file"]})
        backend.run(SimulationRequest("sim", 1), context=restricted)
        assert client.contexts == [default.to_dict(), restricted.to_dict()]
        with pytest.raises(ValueError, match="cannot send live RuntimeServices"):
            backend.run(SimulationRequest("sim", 1), services=RuntimeServices.from_context(RuntimeContext()))
        with pytest.raises(ValueError, match="cannot send live RuntimeServices"):
            backend.check_runtime(context=default, services=RuntimeServices.from_context(RuntimeContext()))
        assert len(client.contexts) == 2
    finally:
        backend.close()


class MarkerAPI:
    def __init__(self, session: RunSession) -> None:
        self.session = session

    def instructions(self) -> str:
        return "marker"

    def generate_samples(self, **kwargs: Any) -> Any:
        kwargs.pop("materialize", None)
        return self.session.execute_simulation(
            method_name="generate_samples",
            request=_marker_request(Path(self.session.instance_spec.visible_data["marker_root"])),
        )

    def generate_samples_batch(self, requests: list[dict[str, Any]], **kwargs: Any) -> list[Any]:
        return [self.generate_samples(**request) for request in requests]


def _session(backend: LocalBackend, marker_root: Path, context: RuntimeContext) -> RunSession:
    spec = TaskSpec(task_id="toy/context", package="mini_bang.testing", config={"api": {"max_generate_calls": 1}})
    instance = InstanceSpec(
        task_id=spec.task_id,
        instance_id=marker_root.name,
        package=spec.package,
        visible_data={"marker_root": str(marker_root)},
    )
    session = RunSession(task_spec=spec, instance_spec=instance, backend=backend, policy=BaseTaskPolicy(), runtime_context=context)
    session.policy.initialize_session(session)
    session.environment = TaskEnvironment(description="marker", api=MarkerAPI(session), metadata={})
    return session


def _run_backend_marker(backend: LocalBackend, marker_root: Path, context: RuntimeContext) -> str:
    return _marker(backend.run_batch([_marker_request(marker_root)], context=context)[0])


def _run_session_marker(session: RunSession, marker_root: Path) -> str:
    payload = session.execute_simulation(method_name="generate_samples", request=_marker_request(marker_root))
    return str(payload["trajectories"][0]["marker"])


def _register_marker_simulator_parent() -> None:
    simulator_registry._REGISTRY._specs.pop(MARKER_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=MARKER_ID,
            description="test-only context marker simulator",
            factory_path=f"{MARKER_PACKAGE}.factory:create_macro",
            grouped_sampling_safe=True,
        )
    )


def _marker_roots(tmp_path: Path, *markers: str) -> tuple[Path, ...]:
    roots = []
    for marker in markers:
        root = tmp_path / marker
        root.mkdir()
        (root / "marker.txt").write_text(marker, encoding="utf-8")
        roots.append(root)
    return tuple(roots)


def _context(cache_root: Path, allowed_root: Path) -> RuntimeContext:
    return RuntimeContext(
        artifact={
            "cache_dir": str(cache_root / "cache"),
            "allowed_schemes": ["file"],
            "allowed_file_roots": [str(allowed_root)],
        },
        security={"network_enabled": False, "gdrive_enabled": False},
    )


def _marker_request(marker_root: Path) -> SimulationRequest:
    return SimulationRequest(
        simulator_id=MARKER_ID,
        saturation=1,
        macro_params={"marker_ref": ArtifactRef(uri=(marker_root / "marker.txt").as_uri()).to_dict()},
        sample_params={"delay_seconds": 0.001},
    )


def _marker(result: SimulationResult) -> str:
    return str(result.payload["trajectories"][0]["marker"])


def _forbidden_set_runtime_context(context: RuntimeContext) -> None:
    raise AssertionError("official/multi-session paths must pass RuntimeContext per call")


class _BarrierPlanner:
    def __init__(self, *, parties: int) -> None:
        self._inner = SimulationPlanner()
        self._barrier = threading.Barrier(parties)

    def plan(self, request: SimulationRequest, *, request_index: int = 0):
        self._barrier.wait(timeout=10)
        return self._inner.plan(request, request_index=request_index)


class _RecordingBackend:
    def __init__(self) -> None:
        self.markers: list[str] = []

    def run(self, request: SimulationRequest, *, context=None, services=None) -> SimulationResult:
        marker = context.security.get("marker") if context else "none"
        self.markers.append(marker)
        return SimulationResult(request=request, payload={"trajectories": [{"marker": marker}]}, request_hash=request.request_hash)

    def run_batch(self, requests, *, context=None, services=None):
        return [self.run(request, context=context, services=services) for request in requests]

    def describe(self) -> dict[str, Any]:
        return {"backend": "recording"}


class _FakeMCPClient:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def call_get_simulation(self, simulator_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.contexts.append(payload.get("runtime_context"))
        return {"ok": True}

    def describe(self) -> dict[str, Any]:
        return {"fake": True}


class _server:
    def __init__(self, server: OfficialRunServer) -> None:
        self.server = server

    def __enter__(self) -> str:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self.server.handler_class())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __exit__(self, *args) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _post_json(url: str, body: dict[str, Any], token: str) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))
