"""RemoteTaskAPI accepts what the in-process task API of the session accepts.

A local multi-session official server hosts a session whose task API is the
real FlowLeniaAPI; only the session's simulation backend is a fixture that
stores one NPZ per observation in a filesystem payload store.
"""
from __future__ import annotations

import http.server
import io
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mini_bang.agent_runtime import AgentRuntimeError, RemotePayloadHandle, RemoteTaskAPI
from mini_bang.client.cli.official import OfficialRunServer
from mini_bang.execution.payload_stores import FileSystemPayloadStore
from mini_bang.execution.payloads import payload_to_transport_payload
from mini_bang.tasks.flow_lenia_common.api import FlowLeniaAPI


class FlowLeniaFixtureSession:
    """Session surface used by FlowLeniaAPI and the official server."""

    def __init__(self, store: FileSystemPayloadStore) -> None:
        self.run_id = "flow-lenia-fixture-session"
        self.task_id = "flow_lenia/presence"
        self.instance_id = "fixture-instance"
        self.store = store
        self.batches: list[list[tuple[str, dict]]] = []
        self.environment = SimpleNamespace(description="", metadata={}, api=FlowLeniaAPI(session=self))

    def capabilities(self) -> dict:
        return {}

    def execute_simulation_batch(self, *, method_name, requests, materialize):
        assert method_name == "generate_samples" and materialize is True
        self.batches.append([(request.macro_params["case_id"], dict(request.sample_params)) for request in requests])
        responses = []
        for request in requests:
            sample = request.sample_params
            steps = list(range(sample["start_step"], sample["end_step"] + 1, sample["step"]))
            buffer = io.BytesIO()
            np.savez(buffer, steps=np.asarray(steps), rgb=np.zeros((len(steps), 2, 2, 3), dtype=np.uint8))
            ref = self.store.put(buffer.getvalue(), metadata={"format": "npz"})
            responses.append({"steps": steps, "format": "npz",
                              "observation": payload_to_transport_payload(None, ref)})
        return responses


class RAFSignatureAPI:
    """Signature of the in-process RAF task APIs (first three arguments positional)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def instructions(self) -> str:
        return ""

    def generate_samples(self, saturation, runs=1, snapshot_times=None, *, extras=None,
                         macro_params=None, materialize=True):
        self.calls.append({"saturation": saturation, "runs": runs, "snapshot_times": snapshot_times,
                           "extras": extras, "macro_params": macro_params})
        return {"accepted": True}


@pytest.fixture
def official_server():
    server = OfficialRunServer(mode="multi_session_official", admin_token="organizer-token")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    def client(session) -> RemoteTaskAPI:
        server.register_session(session)
        return RemoteTaskAPI(runner_url=f"http://127.0.0.1:{httpd.server_address[1]}", instructions="",
                             capabilities={}, session_id=session.run_id,
                             session_token=server.session_token(session.run_id))
    try:
        yield client
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        server.observations.close()


@pytest.fixture
def flow_session(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_BANG_FLOW_POOL_REF", json.dumps({"uri": "file:///fixture/pool.json"}))
    return FlowLeniaFixtureSession(FileSystemPayloadStore(tmp_path / "payloads"))


def observed_steps(handle: RemotePayloadHandle) -> list[int]:
    with np.load(io.BytesIO(handle.load()), allow_pickle=False) as arrays:
        return arrays["steps"].tolist()


def test_flow_lenia_single_observation_through_official_http(official_server, flow_session) -> None:
    api = official_server(flow_session)

    result = api.generate_samples(case_id="case-a", start_step=1000, end_step=1200, step=100, fields=["rgb"])
    defaults = api.generate_samples(case_id="case-b", start_step=0, end_step=6000)

    assert flow_session.batches[0] == [("case-a", {"start_step": 1000, "end_step": 1200, "step": 100,
                                                   "fields": ["rgb"], "seed": 0, "resource_spec": {"gpu": 0}})]
    # Omitted arguments keep the in-process FlowLeniaAPI defaults.
    assert flow_session.batches[1][0][1]["step"] == 3000
    assert flow_session.batches[1][0][1]["fields"] == ["rgb"]
    assert result["steps"] == [1000, 1100, 1200]
    assert isinstance(result["observation"], RemotePayloadHandle)
    assert not result["observation"].is_materialized
    assert observed_steps(result["observation"]) == [1000, 1100, 1200]
    assert observed_steps(defaults["observation"]) == [0, 3000, 6000]


def test_flow_lenia_batch_is_one_atomic_task_api_call(official_server, flow_session) -> None:
    api = official_server(flow_session)

    results = api.generate_samples_batch([
        {"case_id": "case-c", "start_step": 100, "end_step": 300, "step": 100},
        {"case_id": "case-d", "start_step": 500, "end_step": 500, "step": 100, "fields": ("rgb",)},
    ])

    assert [[case for case, _ in batch] for batch in flow_session.batches] == [["case-c", "case-d"]]
    assert [observed_steps(item["observation"]) for item in results] == [[100, 200, 300], [500]]


def test_task_api_of_the_session_rejects_foreign_arguments(official_server, flow_session) -> None:
    api = official_server(flow_session)

    with pytest.raises(AgentRuntimeError, match="HTTP 400"):
        api.generate_samples(case_id="case-e", start_step=0, end_step=100, saturation=4)
    with pytest.raises(AgentRuntimeError, match="HTTP 400"):
        api.generate_samples_batch([{"case_id": "case-e", "start_step": 0, "end_step": 100, "runs": 2}])
    assert flow_session.batches == []


def test_raf_positional_arguments_keep_their_names(official_server) -> None:
    raf = RAFSignatureAPI()
    session = SimpleNamespace(run_id="raf-fixture-session", task_id="raf/timing", instance_id="fixture",
                              environment=SimpleNamespace(description="", metadata={}, api=raf),
                              capabilities=lambda: {})
    api = official_server(session)

    assert api.generate_samples(6, 2, (0.5, 1.0), extras=("is_raf",), macro_params={"seed": 2001}) == {"accepted": True}
    with pytest.raises(TypeError, match="multiple values for saturation"):
        api.generate_samples(6, saturation=7)
    assert raf.calls == [{"saturation": 6, "runs": 2, "snapshot_times": [0.5, 1.0], "extras": ["is_raf"],
                          "macro_params": {"seed": 2001}}]
