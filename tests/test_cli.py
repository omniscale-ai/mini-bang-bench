from __future__ import annotations

import http.server
import json
import threading
from argparse import Namespace
from typing import Any

import pytest

from mini_bang import cli
from mini_bang.agent_runtime import AgentRuntimeError, RemoteTaskAPI, get_json, post_json
from mini_bang.client.cli.official import OfficialRunServer
from mini_bang.config_validation import CompatibilityConfig, validate_compatibility
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import TaskSubmission


def test_cli_task_list_json(capsys) -> None:
    exit_code = cli.main(["task", "list", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {task["task_id"] for task in payload["tasks"]} >= {"raf/timing"}


def test_init_dev_local_writes_config_and_dirs(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["init", "--preset", "dev-local", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["preset"] == "dev-local"
    assert (tmp_path / "mini-bang.yaml").is_file()
    assert (tmp_path / "workdir" / "runs").is_dir()
    assert (tmp_path / "workdir" / "datasets" / "task-artifacts").is_dir()
    assert (tmp_path / "workdir" / "cache").is_dir()
    assert (tmp_path / "workdir" / "runs" / "observations").is_dir()
    assert (tmp_path / "workdir" / "infrastructure").is_dir()


def test_k8s_preset_uses_local_minio_payload_defaults() -> None:
    config = cli.preset_config("k8s")

    assert config["mode"] == "k8s-official"
    assert config["payload"]["mode"] == "minio"
    assert config["payload"]["s3"]["bucket"] == "mini-bang-payloads"
    assert config["payload"]["s3"]["endpoint_url"] == "http://mini-bang-minio:9000"


def test_k8s_render_minio_includes_storage_manifests() -> None:
    manifests = cli.render_k8s_manifests(
        Namespace(
            namespace="mini-bang",
            executor_replicas=1,
            payload_store="minio",
            s3_bucket=None,
            s3_prefix=None,
            s3_endpoint_url=None,
            agent_image=None,
        )
    )

    assert "minio.yaml" in manifests
    assert "minio-bootstrap.yaml" in manifests
    assert "pool-trusted-workers.yaml" in manifests
    assert "agent-job-template.yaml" not in manifests
    assert "MINI_BANG_PAYLOAD_S3_ENDPOINT_URL: http://mini-bang-minio:9000" in manifests["configmap.yaml"]
    assert 'AWS_ACCESS_KEY_ID: "mini-bang"' in manifests["secret-storage.yaml"]


def test_k8s_runner_deployment_is_real() -> None:
    manifests = cli.render_k8s_manifests(
        Namespace(
            namespace="mini-bang",
            executor_replicas=2,
            worker_image="mini-bang-worker:test",
            payload_store="minio",
            s3_bucket=None,
            s3_prefix=None,
            s3_endpoint_url=None,
            agent_image=None,
            with_minio=True,
        )
    )

    runner = manifests["runner.yaml"]
    assert "kind: Deployment" in runner
    assert "kind: Service" in runner
    assert "mini_bang.client.cli.official_runner" in runner
    assert "mini-bang-runner-status" not in runner
    assert "NOT SUPPORTED" not in runner


def test_k8s_agent_has_no_forbidden_env() -> None:
    rendered = cli.render_k8s_agent_job(
        Namespace(
            run_id="run-1",
            session_secret_name="mini-bang-runner-session-run-1",
            task="task/test",
            instance="default",
            namespace="mini-bang",
            agent_image="agent:latest",
            agent=None,
            agent_file=None,
            agent_command=None,
            cpu=None,
            memory=None,
            gpu=None,
        )
    )

    assert "MINI_BANG_RUNNER_URL" in rendered
    assert "MINI_BANG_RUNNER_TOKEN" in rendered
    assert "MINI_BANG_TASK_ID" in rendered
    assert "MINI_BANG_RUN_ID" in rendered
    assert "MCP" not in rendered
    assert "AWS_" not in rendered
    assert "S3" not in rendered
    assert "GOLD" not in rendered
    assert "ttlSecondsAfterFinished: 300" in rendered
    assert "automountServiceAccountToken: false" in rendered


def test_k8s_install_with_minio_forces_minio_payload_store() -> None:
    args = Namespace(payload_store="s3", with_minio=True)

    assert cli.effective_k8s_payload_store(args) == "minio"


def test_cli_builds_without_client_extension_manifests(monkeypatch, capsys) -> None:
    from mini_bang.client.cli import extensions, parser as cli_parser
    monkeypatch.setattr(extensions, "client_extension_manifests", lambda: [])
    parser = cli_parser.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["k8s", "submit", "--task", "raf/timing"])
    assert parser.parse_args(["k8s", "render"]).func is cli_parser.cmd_k8s_render
    assert extensions.runner_gateway(object()) is None
    assert cli.main(["task", "list", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)


def test_official_save_scrubs_raw_file_refs_from_run_record(tmp_path) -> None:
    record = RunRecord.from_parts(
        run_id="run-1",
        task_id="task",
        instance_id="default",
        submission=TaskSubmission(answer={"ok": True}),
        trace=[
            {
                "event": "simulation_request",
                "response_metadata": {
                    "payload_ref": {
                        "uri": "file:///tmp/secret-payload.json",
                        "store_kind": "filesystem",
                        "codec": "json",
                        "media_type": "application/json",
                    }
                },
            }
        ],
    )

    run_dir = cli.save_run_outputs(record, str(tmp_path / "run-1"), mode="local-official")

    saved = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert "file:///tmp/secret-payload.json" not in json.dumps(saved)
    assert "artifact://" in json.dumps(saved)
    assert manifest["artifacts"][0]["ref"]["uri"] == "file:///tmp/secret-payload.json"


def test_storage_test_memory(capsys) -> None:
    exit_code = cli.main(
        [
            "storage",
            "test",
            "--mode",
            "memory",
            "--write-test",
            "--read-test",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(item["message"] == "can put object" for item in payload["checks"])


def test_remote_task_api_batch_uses_batch_endpoint(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post_json(url: str, payload: dict, *, token: str | None = None) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["token"] = token
        return {"payloads": [{"ok": 1}, {"ok": 2}]}

    monkeypatch.setattr("mini_bang.agent_runtime.post_json", fake_post_json)
    api = RemoteTaskAPI(
        runner_url="http://runner",
        instructions="",
        capabilities={},
        session_id="run-123",
        session_token="token-123",
    )

    result = api.generate_samples_batch(
        [
            {"saturation": 6, "runs": 1, "extras": ["is_raf"]},
            {"saturation": 7, "runs": 2, "extras": ["is_raf"]},
        ],
        materialize=False,
    )

    assert result == [{"ok": 1}, {"ok": 2}]
    assert captured["url"] == "http://runner/v1/generate_samples_batch"
    assert captured["payload"] == {
        "requests": [
            {"saturation": 6, "runs": 1, "extras": ["is_raf"]},
            {"saturation": 7, "runs": 2, "extras": ["is_raf"]},
        ],
        "materialize": False,
        "session_id": "run-123",
    }
    assert captured["token"] == "token-123"


class DummyOfficialAPI:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def instructions(self) -> str:
        return f"instructions:{self.name}"

    def generate_samples(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"session": self.name, "kwargs": dict(kwargs)}

    def generate_samples_batch(self, requests, *, materialize: bool = True):
        self.calls.append({"batch": list(requests), "materialize": materialize})
        return [{"session": self.name, "request": dict(item)} for item in requests]


class DummyOfficialEnvironment:
    def __init__(self, api: DummyOfficialAPI) -> None:
        self.description = f"description:{api.name}"
        self.api = api
        self.metadata = {"name": api.name}


class DummyOfficialSession:
    def __init__(self, run_id: str, api: DummyOfficialAPI) -> None:
        self.run_id = run_id
        self.task_id = "task/test"
        self.instance_id = "default"
        self.environment = DummyOfficialEnvironment(api)

    def capabilities(self) -> dict[str, Any]:
        return {"run_id": self.run_id}


class DummyRunManager:
    def __init__(self) -> None:
        self.created: list[tuple[str, str | None]] = []

    def prepare(self, task_id: str, *, instance_id: str | None = None) -> DummyOfficialSession:
        self.created.append((task_id, instance_id))
        api = DummyOfficialAPI(f"{task_id}:{instance_id or 'default'}")
        return DummyOfficialSession(f"run-{len(self.created)}", api)


@pytest.fixture
def isolated_official_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_BANG_OBSERVATION_RECEIPTS_DIR", str(tmp_path / "receipts"))


def test_official_run_server_routes_requests_by_session_id(isolated_official_receipts) -> None:
    api_1 = DummyOfficialAPI("one")
    api_2 = DummyOfficialAPI("two")
    server = OfficialRunServer(DummyOfficialSession("run-1", api_1))
    server.register_session(DummyOfficialSession("run-2", api_2))
    token_2 = server.session_token("run-2")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        info = get_json(f"{base}/v1/run?session_id=run-2", token=token_2)
        response = post_json(
            f"{base}/v1/generate_samples",
            {"session_id": "run-2", "saturation": 7, "runs": 1},
            token=token_2,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        server.observations.close()

    assert info["session_id"] == "run-2"
    assert response["payload"]["session"] == "two"
    assert api_1.calls == []
    assert api_2.calls == [{"saturation": 7, "runs": 1, "materialize": True}]


def test_official_run_server_can_create_sessions_on_one_endpoint(isolated_official_receipts) -> None:
    manager = DummyRunManager()
    server = OfficialRunServer(run_manager=manager)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        created = post_json(
            f"{base}/v1/runs",
            {"task_id": "task/test", "instance_id": "case-1"},
        )
        response = post_json(
            f"{base}/v1/generate_samples",
            {"session_id": created["session_id"], "saturation": 3, "runs": 1},
            token=created["session_token"],
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        server.observations.close()

    assert created["session_id"] == "run-1"
    assert isinstance(created["session_token"], str)
    assert manager.created == [("task/test", "case-1")]
    assert response["payload"]["session"] == "task/test:case-1"


def test_official_run_server_rejects_wrong_session_token(isolated_official_receipts) -> None:
    server = OfficialRunServer(DummyOfficialSession("run-1", DummyOfficialAPI("one")))
    server.register_session(DummyOfficialSession("run-2", DummyOfficialAPI("two")))
    wrong_token = server.session_token("run-1")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler_class())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with pytest.raises(AgentRuntimeError, match="HTTP 401"):
            post_json(
                f"{base}/v1/generate_samples",
                {"session_id": "run-2", "saturation": 7, "runs": 1},
                token=wrong_token,
            )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        server.observations.close()


def test_compatibility_rejects_inprocess_local_official() -> None:
    findings = validate_compatibility(
        CompatibilityConfig(
            mode="local-official",
            agent_placement="in-process",
            backend="local",
            engine="sequential",
            payload_store="filesystem",
        )
    )

    assert any(item.code == "local_official.agent_placement" for item in findings)


def test_compatibility_rejects_inline_official_payloads() -> None:
    findings = validate_compatibility(
        CompatibilityConfig(
            mode="local-official",
            agent_placement="docker",
            backend="local",
            engine="sequential",
            payload_store="inline",
        )
    )

    assert any(item.code == "official.payload_store" for item in findings)


def test_compatibility_accepts_k8s_stack() -> None:
    findings = validate_compatibility(
        CompatibilityConfig(
            mode="k8s-official",
            agent_placement="k8s-stack",
            backend="cluster",
            engine="executor_replicas",
            payload_store="s3",
        )
    )

    assert not [item for item in findings if item.level == "error"]
