from __future__ import annotations

import http.server
import threading
from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.client.cli.official import OfficialRunServer
from mini_bang.execution.payloads import FileSystemPayloadStore


class DummyAPI:
    def instructions(self) -> str:
        return "instructions"

    def generate_samples(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "kwargs": kwargs}


class DummyEnv:
    description = "description"
    metadata: dict[str, Any] = {}
    api = DummyAPI()


class DummySession:
    run_id = "run-1"
    task_id = "task/test"
    instance_id = "default"
    environment = DummyEnv()
    # A task without a declared answer format: the runner checks only that the answer is a JSON object.
    task_spec = SimpleNamespace(task_id="task/test", config={})
    instance_spec = None

    def capabilities(self) -> dict[str, Any]:
        return {"remaining_generate_calls": 1}


class DummyManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def prepare(self, task_id: str, *, instance_id: str | None = None) -> DummySession:
        self.calls.append((task_id, instance_id))
        session = DummySession()
        session.run_id = f"run-{len(self.calls)}"
        return session


def test_multi_session_official_requires_admin_token_to_create_run() -> None:
    manager = DummyManager()
    server = OfficialRunServer(
        run_manager=manager,
        mode="multi_session_official",
        admin_token="admin",
    )
    with _server(server) as base:
        with pytest.raises(AgentRuntimeError, match="HTTP 401"):
            post_json(f"{base}/v1/runs", {"task_id": "task/test"})
        created = post_json(
            f"{base}/v1/runs",
            {"task_id": "task/test", "instance_id": "case-1"},
            token="admin",
        )
        with pytest.raises(AgentRuntimeError, match="HTTP 401"):
            post_json(f"{base}/v1/runs", {"task_id": "task/test"}, token=created["session_token"])
        response = post_json(
            f"{base}/v1/generate_samples",
            {"session_id": created["session_id"], "saturation": 1},
            token=created["session_token"],
        )

    assert manager.calls == [("task/test", "case-1")]
    assert response["payload"]["ok"] is True


def test_missing_session_id_falls_back_only_in_single_session_mode() -> None:
    multi = OfficialRunServer(run_manager=DummyManager(), mode="multi_session_official", admin_token="admin")
    with _server(multi) as base:
        created = post_json(f"{base}/v1/runs", {"task_id": "task/test"}, token="admin")
        token = created["session_token"]
        # Exactly one registered session: multi-session mode still routes explicitly.
        for call in (lambda: get_json(f"{base}/v1/run", token=token),
                     lambda: post_json(f"{base}/v1/generate_samples", {"saturation": 1}, token=token),
                     lambda: post_json(f"{base}/v1/submission", {"answer": {}}, token=token)):
            with pytest.raises(AgentRuntimeError, match="HTTP 400.*session_id is required"):
                call()
        routed = get_json(f"{base}/v1/run?session_id={created['session_id']}", token=token)
    single = OfficialRunServer(DummySession())
    with _server(single) as base:
        fallback = get_json(f"{base}/v1/run", token=single.session_token("run-1"))

    assert routed["session_id"] == created["session_id"]
    assert fallback["session_id"] == "run-1"


def test_multi_session_official_fails_fast_without_admin_token(monkeypatch) -> None:
    monkeypatch.delenv("MINI_BANG_RUNNER_ADMIN_TOKEN", raising=False)
    with pytest.raises(ValueError, match="ADMIN_TOKEN"):
        OfficialRunServer(run_manager=DummyManager(), mode="multi_session_official")


def test_artifact_metadata_and_download_do_not_use_legacy_json_body(tmp_path) -> None:
    store = FileSystemPayloadStore(tmp_path / "payloads")
    ref = store.put(b"x" * 1024, key="blob", metadata={"request_hash": "hash-1"})
    server = OfficialRunServer(DummySession())
    server._artifact_refs["run-1"]["artifact-1"] = ref
    token = server.session_token("run-1")
    with _server(server) as base:
        metadata = get_json(f"{base}/v1/artifacts/artifact-1/metadata", token=token)
        body = _get_bytes(f"{base}/v1/artifacts/artifact-1/download", token=token)
        with pytest.raises(AgentRuntimeError, match="HTTP 400"):
            get_json(f"{base}/v1/artifacts/artifact-1", token=token)

    assert metadata["size_bytes"] == 1024
    assert metadata["created_by"]["request_hash"] == "hash-1"
    assert body == b"x" * 1024


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


def _get_bytes(url: str, *, token: str) -> bytes:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    with urlopen(request, timeout=300) as response:
        return response.read()
