"""The compose stack runs the real official runner; checked without Docker.

Docker is not required: the rendered runner command is parsed by the official
runner's own CLI parser and then started as a plain process with the rendered
environment (container paths moved to a temporary directory). An agent process
gets the rendered agent environment and completes one RAF session end to end.
"""
from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from mini_bang import cli
from mini_bang.client.cli.official_runner import build_parser

yaml = pytest.importorskip("yaml")

SRC = Path(__file__).resolve().parents[1] / "src"
RUNNER_MODULE = ["python", "-m", "mini_bang.client.cli.official_runner"]


def render(**overrides) -> dict:
    values = dict(with_minio=False, payload_store="filesystem", executor_workers=2,
                  agent_image="my-agent:latest", port=8765, worker_image=None)
    values.update(overrides)
    return yaml.safe_load(cli.render_compose_yaml(Namespace(**values)))


def test_runner_service_is_the_official_runner_with_documented_flags() -> None:
    stack = render(executor_workers=4, port=8899)
    runner = stack["services"]["runner"]
    k8s = cli.render_k8s_manifests(Namespace(
        namespace="mini-bang", executor_replicas=1, worker_image="mini-bang-worker:latest",
        payload_store="minio", s3_bucket=None, s3_prefix=None, s3_endpoint_url=None,
        agent_image=None, with_minio=True))
    k8s_runner = next(item for item in yaml.safe_load_all(k8s["runner.yaml"]) if item["kind"] == "Deployment")

    assert "http.server" not in json.dumps(stack)
    assert runner["image"] == k8s_runner["spec"]["template"]["spec"]["containers"][0]["image"]
    assert runner["command"][:3] == RUNNER_MODULE
    assert runner["command"][:3] == k8s_runner["spec"]["template"]["spec"]["containers"][0]["command"]
    arguments = build_parser().parse_args(runner["command"][3:])
    assert (arguments.host, arguments.port) == ("0.0.0.0", 8899)
    assert runner["ports"] == ["127.0.0.1:8899:8899"]
    environment = runner["environment"]
    assert environment["MINI_BANG_BACKEND"] == "local"
    assert environment["MINI_BANG_EXECUTION_ENGINE"] == "process_pool"
    assert environment["MINI_BANG_WORKERS"] == "4"
    assert environment["MINI_BANG_RUNNER_ADMIN_TOKEN"] is None  # taken from the operator shell
    assert environment["MINI_BANG_OBSERVATION_RECEIPTS_DIR"] == "/state/observations"
    assert environment["MINI_BANG_PAYLOAD_STORE"] == "filesystem"
    assert runner["depends_on"] == {"runner-state-permissions": {"condition": "service_completed_successfully"}}


def test_agent_reaches_only_the_runner_on_an_internal_network() -> None:
    stack = render()
    agent = stack["services"]["agent"]

    assert stack["networks"]["agent-boundary"] == {"internal": True}
    assert agent["networks"] == ["agent-boundary"]
    assert set(stack["services"]["runner"]["networks"]) == {"default", "agent-boundary"}
    assert agent["environment"] == {"MINI_BANG_RUNNER_URL": "http://runner:8765", "MINI_BANG_RUN_ID": None,
                                    "MINI_BANG_RUNNER_TOKEN": None, "MINI_BANG_AGENT": None,
                                    "MINI_BANG_AGENT_FILE": None}
    assert "minio" not in stack["services"]


def test_with_minio_stores_payloads_in_that_minio() -> None:
    stack = render(with_minio=True)
    runner = stack["services"]["runner"]

    assert runner["environment"]["MINI_BANG_PAYLOAD_STORE"] == "s3"
    assert runner["environment"]["MINI_BANG_PAYLOAD_S3_ENDPOINT_URL"] == "http://minio:9000"
    assert runner["environment"]["MINI_BANG_PAYLOAD_S3_BUCKET"] == "mini-bang-payloads"
    assert runner["depends_on"]["minio-init"] == {"condition": "service_completed_successfully"}
    assert "agent-boundary" not in stack["services"]["minio"].get("networks", ["default"])


def test_compose_up_writes_the_stack_and_requires_the_admin_token(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    module = "mini_bang.client.cli.commands_deployment"
    monkeypatch.setattr(f"{module}.check_docker_or_raise", lambda: None)
    monkeypatch.setattr(f"{module}.check_docker_runtime_or_raise", lambda: None)
    monkeypatch.setattr(f"{module}.docker_compose_cmd", lambda: ["docker", "compose"])
    calls = []
    monkeypatch.setattr(f"{module}.subprocess.call", lambda command: calls.append(command) or 0)
    arguments = ["compose", "up", "--with-minio", "--executor-workers", "4", "--detach", "--format", "json"]

    monkeypatch.delenv("MINI_BANG_RUNNER_ADMIN_TOKEN", raising=False)
    assert cli.main(arguments) != 0
    assert calls == []
    capsys.readouterr()
    monkeypatch.setenv("MINI_BANG_RUNNER_ADMIN_TOKEN", "organizer-token")
    assert cli.main(arguments) == 0

    output = json.loads(capsys.readouterr().out)
    stack = yaml.safe_load(Path(output["compose_file"]).read_text())
    assert calls == [["docker", "compose", "-f", output["compose_file"], "up", "-d"]]
    assert output["payload_store"] == "minio"
    assert stack["services"]["runner"]["command"][:3] == RUNNER_MODULE
    assert stack["services"]["runner"]["environment"]["MINI_BANG_WORKERS"] == "4"


AGENT = '''
def solve(task):
    families = task.metadata["network_families"]
    observation = task.api.generate_samples(saturation=4, runs=1, extras=["is_raf"],
                                            macro_params={"family": families[0], "network_index": 1})
    assert isinstance(observation, dict) and "p" not in observation["macro_params"], observation
    return {"probabilities": {name: {str(size): 0.5 for size in task.metadata["saturations"]}
                              for name in families}}
'''


def http(method, url, token, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def container_environment(values: dict, root: Path) -> dict:
    """Rendered environment with container paths moved under ``root``."""
    moved = {}
    for name, value in values.items():
        if value is not None:
            for mount in ("/state", "/payloads"):
                if str(value).startswith(mount):
                    value = str(root) + str(value)
            moved[name] = str(value)
    return moved


def test_rendered_runner_and_agent_complete_a_raf_session(tmp_path) -> None:
    port, admin = free_port(), "organizer-token"
    stack = render(port=port)
    runner_service, agent_service = stack["services"]["runner"], stack["services"]["agent"]
    pythonpath = os.pathsep.join(filter(None, [str(SRC), str(SRC.parent), os.environ.get("PYTHONPATH")]))
    base = {**os.environ, "PYTHONPATH": pythonpath}
    runner_env = {**base, **container_environment(runner_service["environment"], tmp_path),
                  "MINI_BANG_RUNNER_ADMIN_TOKEN": admin}
    # The rendered runner command, started with synthetic gold for its hidden scorer.
    command = [sys.executable, "-m", "tests.synthetic_gold", *runner_service["command"][1:]]
    command[command.index("0.0.0.0")] = "127.0.0.1"
    url = f"http://127.0.0.1:{port}"
    runner = subprocess.Popen(command, env=runner_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                urllib.request.urlopen(f"{url}/health", timeout=2).read()
                break
            except (urllib.error.URLError, ConnectionError):
                assert runner.poll() is None and time.monotonic() < deadline, runner.stdout.read()
                time.sleep(0.2)
        created = http("POST", f"{url}/v1/runs", admin, {"task_id": "raf/mechanism-prob"})
        agent_file = tmp_path / "agent.py"
        agent_file.write_text(AGENT)
        session_values = {"MINI_BANG_RUN_ID": created["session_id"],
                          "MINI_BANG_RUNNER_TOKEN": created["session_token"],
                          "MINI_BANG_AGENT_FILE": str(agent_file)}
        agent_env = {**base, **container_environment(agent_service["environment"], tmp_path),
                     **session_values, "MINI_BANG_RUNNER_URL": url}
        agent = subprocess.run([sys.executable, "-m", "mini_bang.agent_runtime"], env=agent_env,
                               capture_output=True, text=True, timeout=300)
        assert agent.returncode == 0, agent.stderr
        deadline = time.monotonic() + 120
        result = {"status": "pending"}
        while result["status"] in {"pending", "scoring"} and time.monotonic() < deadline:
            time.sleep(0.5)
            result = http("GET", f"{url}/v1/result?session_id={created['session_id']}", admin)
    finally:
        runner.terminate()
        runner.wait(timeout=30)

    assert result["status"] == "succeeded", result
    record = result["run_record"]
    assert record["task_id"] == "raf/mechanism-prob"
    assert record["submission"]["answer"]["probabilities"] == {
        f"family-{index}": {"20": 0.5, "30": 0.5, "40": 0.5} for index in range(1, 5)}
    # The hidden scorer ran; an answer shared by all families scores against the shared-answer anchor.
    assert "shared across families" in record["score_summary"]["details"], record["score_summary"]
    assert record["score_summary"]["metrics"]
    assert list((tmp_path / "state" / "observations" / created["session_id"]).glob("observation-*.json"))
