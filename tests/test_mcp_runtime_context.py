"""The session RuntimeContext crosses MCP; what cannot cross fails before any budget."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import mini_bang.mcp
from mini_bang.artifacts import ArtifactRef
from mini_bang.backends.cache import CachingSimulationBackend, FileSystemSimulationCache
from mini_bang.backends.local import LocalBackend
from mini_bang.backends.mcp import MCPBackend
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.framework.runtime import RunManager
from mini_bang.mcp.simulation_call import run_get_simulation
from mini_bang.simulators import registry as simulator_registry
from tests.synthetic_gold import SMOKE_INSTANCE, use_synthetic_gold

MARKER_ID = "context_marker_over_mcp_test"


class _InProcessMCPServer:
    """Stands in for the FastMCP transport: JSON on the wire, then the server tool body."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.contexts: list[Any] = []

    def call_get_simulation(self, simulator_id: str, payload: dict[str, Any]) -> Any:
        wire = json.loads(json.dumps(payload))
        self.contexts.append(wire.get("runtime_context"))
        response = run_get_simulation(
            self.backend, simulator_id=simulator_id, saturation=wire["saturation"],
            runs=wire.get("runs", 1), extras=wire.get("extras", []),
            macro_params=wire.get("macro_params", {}), micro_params=wire.get("micro_params", {}),
            sample_params=wire.get("sample_params", {}), instance_id=wire.get("instance_id"),
            seed_mode=wire.get("seed_mode"), runtime_context=wire.get("runtime_context"))
        return json.loads(json.dumps(response))

    def describe(self) -> dict[str, Any]:
        return {"transport": "in-process stand-in"}


@pytest.fixture
def marker_simulator():
    simulator_registry._REGISTRY._specs.pop(MARKER_ID, None)
    simulator_registry.register_simulator_entry(simulator_registry.SimulatorSpec(
        simulator_id=MARKER_ID, description="test-only context marker simulator",
        factory_path="mini_bang.testing.context_marker_simulator.factory:create_macro",
        grouped_sampling_safe=True))
    yield
    simulator_registry._REGISTRY._specs.pop(MARKER_ID, None)


def _marker_request(root: Path) -> SimulationRequest:
    return SimulationRequest(simulator_id=MARKER_ID, saturation=1,
                             macro_params={"marker_ref": ArtifactRef(uri=(root / "marker.txt").as_uri()).to_dict()})


def test_runtime_context_round_trips_through_json() -> None:
    context = RuntimeContext(run_id="run", task_id="raf/timing", instance_id="case",
                             artifact={"cache_dir": "/cache"}, payload={"kind": "memory"},
                             security={"authority": "agent"}, env={"A": "1"})
    assert RuntimeContext.from_dict(json.loads(json.dumps(context.to_dict()))) == context


def test_server_executes_under_the_context_sent_by_the_client(tmp_path, marker_simulator) -> None:
    allowed, other = tmp_path / "allowed", tmp_path / "other"
    for root in (allowed, other):
        root.mkdir()
        (root / "marker.txt").write_text(root.name, encoding="utf-8")
    server = _InProcessMCPServer(LocalBackend())
    backend = MCPBackend(client=server)
    context = RuntimeContext(task_id="marker-task", artifact={
        "cache_dir": str(tmp_path / "session-cache"), "allowed_schemes": ["file"],
        "allowed_file_roots": [str(allowed)]})
    try:
        result = backend.run(_marker_request(allowed), context=context)
        assert result.payload["trajectories"][0]["marker"] == "allowed"
        assert list((tmp_path / "session-cache").rglob("*.artifact"))
        with pytest.raises(PermissionError, match="outside allowed roots"):
            backend.run(_marker_request(other), context=context)
        assert server.contexts == [context.to_dict(), context.to_dict()]
    finally:
        backend.close()


def test_session_with_artifact_env_runs_over_mcp_and_charges_one_call(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MINI_BANG_ARTIFACT_CACHE_DIR", str(tmp_path / "artifact-cache"))
    server = _InProcessMCPServer(LocalBackend())
    backend = CachingSimulationBackend(MCPBackend(client=server), FileSystemSimulationCache(tmp_path / "cache"))
    try:
        use_synthetic_gold(monkeypatch)
        session = RunManager(backend=backend).prepare("raf/timing", instance_id=SMOKE_INSTANCE)
        session.environment.api.generate_samples(saturation=6, runs=1, snapshot_times=[0.1],
                                                 extras=["first_hits"], macro_params={"seed": 101})
        assert session.budget_counters["generate_samples"] == 1
        assert server.contexts[0]["artifact"]["cache_dir"] == str(tmp_path / "artifact-cache")
        assert server.contexts[0]["task_id"] == "raf/timing"
    finally:
        backend.close()


def test_runtime_that_cannot_cross_mcp_fails_at_session_setup(monkeypatch, tmp_path) -> None:
    server = _InProcessMCPServer(LocalBackend())
    backend = CachingSimulationBackend(MCPBackend(client=server), FileSystemSimulationCache(tmp_path / "cache"))
    monkeypatch.setattr(RunManager, "_session_runtime_services",
                        lambda self, context: RuntimeServices.from_context(context))
    try:
        with pytest.raises(ValueError, match="cannot send live RuntimeServices"):
            RunManager(backend=backend).prepare("raf/timing")
        assert server.contexts == []
    finally:
        backend.close()


@pytest.fixture
def server_tool(monkeypatch):
    """The real ``get_simulation`` tool function of ``mini_bang.mcp.server``.

    Without fastmcp installed a no-op ``FastMCP`` stands in, so the tool keeps
    its own Python signature: an argument it does not declare fails like an
    unknown FastMCP tool argument. The server modules are re-imported for the
    test and removed again afterwards.
    """
    if importlib.util.find_spec("fastmcp") is None:
        stub, stub_config = types.ModuleType("fastmcp"), types.ModuleType("fastmcp.mcp_config")
        stub.FastMCP = type("FastMCP", (), {"__init__": lambda self, **kwargs: None,
                                            "tool": lambda self, **kwargs: (lambda function: function)})
        stub_config.MCPConfig = stub_config.StdioMCPServer = type("Unused", (), {})
        monkeypatch.setitem(sys.modules, "fastmcp", stub)
        monkeypatch.setitem(sys.modules, "fastmcp.mcp_config", stub_config)
    for name in ("config", "server"):
        monkeypatch.setitem(sys.modules, f"mini_bang.mcp.{name}", None)
        monkeypatch.setattr(mini_bang.mcp, name, None, raising=False)
        del sys.modules[f"mini_bang.mcp.{name}"]
    server = importlib.import_module("mini_bang.mcp.server")
    monkeypatch.setattr(server, "_BACKEND", LocalBackend())
    return server.get_simulation


class _ServerToolClient:
    """JSON on the wire, then the server tool called with exactly the sent arguments."""

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    def call_get_simulation(self, simulator_id: str, payload: dict[str, Any]) -> Any:
        wire = json.loads(json.dumps(payload))
        return json.loads(json.dumps(self.tool(simulator_id=simulator_id, **wire)))

    def describe(self) -> dict[str, Any]:
        return {"transport": "server tool without FastMCP"}


def test_server_tool_accepts_and_applies_seed_mode(server_tool) -> None:
    macro = {"M0": 2, "K": 3, "p": 0.15, "complexity": 0.5, "k_lig": 1.0, "k_unlig": 0.05,
             "alpha": {"kind": "linear_cutoff", "scale": 0.05, "cutoff": 3, "floor": 0.0},
             "step_limit": 400, "seed": 101}

    def request(seed_mode: str) -> SimulationRequest:
        return SimulationRequest(simulator_id="raf", saturation=6, runs=1, extras=("first_hits",),
                                 seed_mode=seed_mode, macro_params=dict(macro),
                                 sample_params={"seed": 7001, "snapshot_times": [0.25, 1.0]})

    backend = MCPBackend(client=_ServerToolClient(server_tool))
    try:
        assert "seed_mode" in request("agent_supplied").to_payload()
        over_mcp = backend.run(request("agent_supplied")).payload["trajectories"]
    finally:
        backend.close()
    local = {mode: json.loads(json.dumps(LocalBackend().run(request(mode)).payload["trajectories"]))
             for mode in ("agent_supplied", "planner_derived")}
    assert over_mcp == local["agent_supplied"] != local["planner_derived"]
