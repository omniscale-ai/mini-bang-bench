"""A new simulator and a new task need only their own two packages.

The test writes a toy simulator (a noisy logistic map) and a toy task for it
into a temporary directory, outside the benchmark source tree. Both packages
are discovered through MINI_BANG_SIMULATOR_PACKAGES and MINI_BANG_TASK_PACKAGES.
The task runs in-process, through the official HTTP runner in its own process
and through scoring. Nothing outside the two packages is edited; the guide
docs/adding_simulators_and_tasks.md describes the same files.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import fmean

import pytest

from mini_bang.agent_connector import AgentConnector
from mini_bang.agent_runtime import AgentRuntimeError, RemoteTaskAPI
from mini_bang.agent_types import TaskSubmission
from mini_bang.backends.local import LocalBackend
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.framework.policy import RequestRejectedError
from mini_bang.framework.records import RunRecord
from mini_bang.simulators import registry
from mini_bang.tasks.loader import TaskLoader

SOURCE = Path(__file__).resolve().parents[1] / "src"
TASK_ID = "toy/logistic-mean-v1"
RATES = [1.5, 2.2, 3.2, 3.9]
ADMIN_TOKEN = "toy-organizer-token"

SIMULATOR_CONFIG = {
    "id": "toy_logistic",
    "simulator_version": "1",
    "description": "Noisy logistic map x[t+1] = r x[t] (1 - x[t]) + noise; saturation is r",
    "factory": "toy_logistic_sim.simulator:create_macro",
    "grouped_sampling_safe": True,
    "runtime_environment": ["MINI_BANG_TOY_DATA_LABEL"],
    "executable_fields": ["x"],
    "cli": "toy_logistic_sim.cli:register",
}
CLI_SOURCE = '''
def register(cli):
    command = cli.add_command("toy-logistic", help="Print the growth rates of the toy task")
    command.set_defaults(func=lambda args: print("1.5 2.2 3.2 3.9") or 0)
    cli.add_deployment_option("--toy-data-label", help="Label of the toy data for runner and workers")
    cli.add_deployment_settings(settings)


def settings(args, *, s3_endpoint_url):
    label = getattr(args, "toy_data_label", None)
    return {"MINI_BANG_TOY_DATA_LABEL": label} if label else {}
'''
SIMULATOR_SOURCE = '''
import random

from pydantic import BaseModel, Field, create_model

from mini_bang.simulators.base import MacroSimulatorBase, MicroSimulatorBase


class LogisticMicro(MicroSimulatorBase):
    def __init__(self, r, noise):
        self.r, self.noise = r, noise

    def sample(self, *, seed, steps=50, x0=0.5):
        rng, x = random.Random(seed), float(x0)
        values = [x]
        for _ in range(int(steps)):
            x = min(1.0, max(0.0, self.r * x * (1.0 - x) + rng.uniform(-self.noise, self.noise)))
            values.append(x)
        return {"x": values}


class LogisticMacro(MacroSimulatorBase):
    def __init__(self, *, seed=0, noise=0.01):
        super().__init__(complexity=0.0)
        self.seed, self.noise = seed, noise

    def get_saturation_description(self):
        return create_model("LogisticRate", value=(float, Field(ge=0.0, le=4.0)))

    def get_micro_simulator(self, T, **kwargs):
        self.get_saturation_description()(value=T)
        return LogisticMicro(float(T), self.noise)


def create_macro(**params):
    return LogisticMacro(**params)
'''
TASK_CONFIG = {
    "task_id": TASK_ID,
    "task_version": "1",
    "description": "Estimate the long-run mean of the noisy logistic map at each listed rate.",
    "api_class": "toy_logistic_task.api:LogisticMeanAPI",
    "policy_class": "toy_logistic_task.policy:LogisticMeanPolicy",
    "scorer_class": "toy_logistic_task.scorer:LogisticMeanScorer",
    "default_instance_id": "default",
    "metadata": {"level": "1", "category": "Toy mean estimation"},
    "api": {"simulator_id": "toy_logistic", "max_generate_calls": 6, "max_runs": 8, "max_steps": 100},
}
INSTANCE = {
    "instance_version": "1",
    "description": "Estimate the mean of x over steps 50..100 (x0 = 0.5) for each listed rate r.",
    "instructions": "Call generate_samples(saturation, runs, steps). Submit {\"means\": {\"<r>\": value}}. "
                    "Six simulation requests are allowed; each batch entry counts as one.",
    "visible": {"test_saturations": RATES, "mean_window": [50, 100]},
}
GOLD = {"gold_version": "1", "means": {"1.5": 0.3331, "2.2": 0.5453, "3.2": 0.6538, "3.9": 0.5921}}
API_SOURCE = '''
from mini_bang.agent_types import AgentAPI
from mini_bang.execution import SimulationRequest


class LogisticMeanAPI(AgentAPI):
    def __init__(self, *, session):
        self._session = session

    def instructions(self):
        return self._session.instance_spec.instructions

    def capabilities(self):
        return self._session.capabilities()

    def remaining_calls(self):
        return self._session.remaining_calls()

    def generate_samples(self, saturation, runs=1, steps=50, *, materialize=True):
        return self._session.execute_simulation(
            method_name="generate_samples", request=self._request(saturation, runs, steps),
            materialize=materialize)

    def generate_samples_batch(self, requests, *, materialize=True):
        return self._session.execute_simulation_batch(
            method_name="generate_samples", requests=[self._request(**dict(item)) for item in requests],
            materialize=materialize)

    def _request(self, saturation, runs=1, steps=50):
        return SimulationRequest(
            simulator_id=self._session.task_spec.config["api"]["simulator_id"],
            saturation=float(saturation), runs=int(runs), sample_params={"steps": int(steps)},
            instance_id=self._session.instance_id)
'''
POLICY_SOURCE = '''
from mini_bang.framework.policy import AllowedParamPolicy, BaseTaskPolicy, PublicRequestError


class LogisticMeanPolicy(BaseTaskPolicy):
    def capabilities(self, session):
        api = session.task_spec.config["api"]
        return {**super().capabilities(session), "max_runs": api["max_runs"], "max_steps": api["max_steps"],
                "allowed_saturations": list(session.instance_spec.visible_data["test_saturations"])}

    def validate_request(self, *, task_spec, instance_spec, session, method_name, request):
        super().validate_request(task_spec=task_spec, instance_spec=instance_spec, session=session,
                                 method_name=method_name, request=request)
        api = task_spec.config["api"]
        if request.simulator_id != api["simulator_id"] or request.macro_params or request.micro_params:
            raise ValueError("simulator and its parameters are fixed by the task")
        if request.saturation not in instance_spec.visible_data["test_saturations"]:
            raise PublicRequestError("saturation must be one of the listed rates")
        if request.runs > api["max_runs"]:
            raise PublicRequestError(f"runs must be at most {api['max_runs']}")
        AllowedParamPolicy(sample_params={"steps", "seed"}).validate(request)
        if not 1 <= request.sample_params["steps"] <= api["max_steps"]:
            raise PublicRequestError(f"steps must be 1..{api['max_steps']}")
'''
SCORER_SOURCE = '''
from statistics import fmean

from mini_bang.framework.task import ValidationResult


class LogisticMeanScorer:
    """Score in [0, 1]: 1 - MAE / MAE of the best constant, clipped; the raw value stays a diagnostic."""

    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store, scoring_services=None):
        targets = {key: float(value) for key, value in gold_store.get("means").items()}
        answer = submission.answer if isinstance(submission.answer, dict) else {}
        predicted = answer.get("means") if isinstance(answer.get("means"), dict) else {}
        values = {key: predicted.get(key) for key in targets}
        missing = sorted(key for key, value in values.items()
                         if not isinstance(value, (int, float)) or isinstance(value, bool))
        mae = fmean(abs((0.0 if key in missing else float(values[key])) - target)
                    for key, target in targets.items())
        constant = fmean(targets.values())
        baseline = fmean(abs(constant - target) for target in targets.values())
        unclipped = 1.0 - mae / baseline
        score = min(1.0, max(0.0, unclipped))
        return ValidationResult(success=score >= 0.5, details=f"MAE {mae:.4f}",
                                metrics={"score": score, "unclipped_score": unclipped, "mae": mae,
                                         "best_constant_mae": baseline, "missing": missing})
'''


def _write_package(root: Path, name: str, files: dict[str, object]) -> None:
    for relative, content in files.items():
        path = root / name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content, indent=2) + "\n")


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("new-simulator-and-task")
    _write_package(root, "toy_logistic_sim", {
        "__init__.py": "", "config.json": SIMULATOR_CONFIG, "simulator.py": SIMULATOR_SOURCE,
        "cli.py": CLI_SOURCE})
    _write_package(root, "toy_logistic_task", {
        "__init__.py": "", "config.json": TASK_CONFIG, "instances/default.json": INSTANCE,
        "gold/default.json": GOLD, "api.py": API_SOURCE, "policy.py": POLICY_SOURCE,
        "scorer.py": SCORER_SOURCE})
    return root


def _package_environment() -> dict[str, str]:
    return {"MINI_BANG_SIMULATOR_PACKAGES": "toy_logistic_sim", "MINI_BANG_TASK_PACKAGES": "toy_logistic_task"}


@pytest.fixture
def toy(toy_root, monkeypatch):
    monkeypatch.syspath_prepend(str(toy_root))
    for name, value in _package_environment().items():
        monkeypatch.setenv(name, value)
    try:
        yield toy_root
    finally:
        registry._REGISTRY._specs.pop("toy_logistic", None)


def _means(results) -> dict[str, float]:
    means = {}
    for rate, result in results:
        samples = result["samples"] if "samples" in result else [result]
        means[str(rate)] = fmean(fmean(item["trajectories"][0]["x"][50:]) for item in samples)
    return means


def test_cli_discovers_both_packages_in_a_fresh_process(toy_root) -> None:
    environment = {**os.environ, **_package_environment(),
                   "PYTHONPATH": os.pathsep.join([str(SOURCE), str(toy_root)])}
    shown = subprocess.run([sys.executable, "-m", "mini_bang", "task", "show", TASK_ID, "--format", "json"],
                           env=environment, cwd=toy_root, capture_output=True, text=True, timeout=120)
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout)["task_id"] == TASK_ID
    simulator = subprocess.run(
        [sys.executable, "-c", "from mini_bang.simulators.registry import get_simulator_entry as g; "
                               "print(g('toy_logistic').factory_path)"],
        env=environment, cwd=toy_root, capture_output=True, text=True, timeout=120)
    assert simulator.stdout.strip() == SIMULATOR_CONFIG["factory"], simulator.stderr


def test_in_process_run_enforces_budget_and_scores_in_unit_interval(toy) -> None:
    def agent(task):
        assert task.metadata["test_saturations"] == RATES and "means" not in json.dumps(task.metadata)
        results = task.api.generate_samples_batch([{"saturation": r, "runs": 4, "steps": 100} for r in RATES])
        assert task.api.remaining_calls() == 2
        task.api.generate_samples(1.5, 1, 10)
        task.api.generate_samples(1.5, 1, 20)
        with pytest.raises(RequestRejectedError, match="call limit"):
            task.api.generate_samples(1.5, 1, 30)
        return TaskSubmission(answer={"means": _means(zip(RATES, results))})

    result = AgentConnector(loader=TaskLoader(), backend=LocalBackend()).run(agent, TASK_ID)

    assert 0.9 <= result.validation.metrics["score"] <= 1.0
    assert result.run_record.task_id == TASK_ID


def test_scorer_gives_one_to_the_oracle_and_zero_to_constants(toy) -> None:
    spec = TaskLoader().get_spec(TASK_ID)
    store = BenchmarkArtifactStore()
    instance = store.load_instance(spec)
    gold = store.load_gold(instance)
    scorer = spec.create_scorer()
    record = RunRecord.from_parts(run_id="toy", task_id=TASK_ID, instance_id="default",
                                  submission=TaskSubmission(answer={}), trace=[], timing={},
                                  resource_usage={}, backend_metadata={}, versions={})

    def score(answer):
        return scorer.score(task_spec=spec, instance_spec=instance, submission=TaskSubmission(answer=answer),
                            run_record=record, gold_store=gold).metrics

    constant = fmean(GOLD["means"].values())
    assert score({"means": GOLD["means"]})["score"] == 1.0
    assert score({"means": {key: constant for key in GOLD["means"]}})["score"] == pytest.approx(0.0, abs=1e-12)
    inverted = score({"means": {key: 1.0 - value for key, value in GOLD["means"].items()}})
    assert inverted["score"] == 0.0 and inverted["unclipped_score"] < 0.0
    assert score("not an answer")["missing"] == sorted(GOLD["means"])


def _http(base: str, method: str, path: str, body=None, token=None):
    request = urllib.request.Request(base + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


@pytest.fixture
def official_runner(toy_root, tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {**os.environ, **_package_environment(),
                   "PYTHONPATH": os.pathsep.join([str(SOURCE), str(toy_root)]),
                   "MINI_BANG_RUNNER_ADMIN_TOKEN": ADMIN_TOKEN, "MINI_BANG_BACKEND": "local",
                   "MINI_BANG_OBSERVATION_RECEIPTS_DIR": str(tmp_path / "state" / "observations"),
                   "MINI_BANG_SCORING_RECEIPTS_DIR": str(tmp_path / "state" / "scoring"),
                   "MINI_BANG_RESULT_STORE": "filesystem", "MINI_BANG_RESULT_DIR": str(tmp_path / "results")}
    process = subprocess.Popen([sys.executable, "-m", "mini_bang.client.cli.official_runner",
                                "--host", "127.0.0.1", "--port", str(port)],
                               env=environment, cwd=tmp_path, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if _http(base, "GET", "/health")[0] == 200:
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("official runner did not start: " + process.stdout.read())
                time.sleep(0.1)
        yield base
    finally:
        process.terminate()
        process.communicate(timeout=30)


def test_official_http_runner_observes_and_scores_the_new_task(official_runner) -> None:
    base = official_runner
    status, created = _http(base, "POST", "/v1/runs", {"task_id": TASK_ID}, ADMIN_TOKEN)
    assert status == 200, created
    session, token = created["session_id"], created["session_token"]
    status, run = _http(base, "GET", f"/v1/run?session_id={session}", token=token)
    assert run["metadata"]["test_saturations"] == RATES
    assert run["capabilities"]["remaining_generate_calls"] == 6
    assert "0.3331" not in json.dumps(run)
    status, signature = _http(base, "GET", f"/v1/api?session_id={session}", token=token)
    assert signature == {"generate_samples": {"positional": ["saturation", "runs", "steps"]}}

    api = RemoteTaskAPI(runner_url=base, instructions=run["instructions"], capabilities=run["capabilities"],
                        session_id=session, session_token=token)
    first = api.generate_samples(RATES[0], 4, 100)
    rest = api.generate_samples_batch([{"saturation": r, "runs": 4, "steps": 100} for r in RATES[1:]])
    assert len(first["samples"]) == 4 and len(first["samples"][0]["trajectories"][0]["x"]) == 101
    assert api.remaining_calls() == 2
    with pytest.raises(AgentRuntimeError, match="saturation must be one of the listed rates"):
        api.generate_samples(saturation=0.1)
    with pytest.raises(TypeError, match="at most 3 positional"):
        api.generate_samples(1.5, 1, 10, 0.5)
    with pytest.raises(AgentRuntimeError, match="call limit exceeded"):
        api.generate_samples_batch([{"saturation": 1.5, "steps": 5}] * 3)
    assert api.remaining_calls() == 2

    status, receipts = _http(base, "GET", f"/v1/observations?session_id={session}", token=token)
    completed = [item for item in receipts["observations"] if item["status"] == "completed"]
    # Generic receipt timing: no RAF or Flow Lenia work estimate for another task's requests.
    assert [item["timing"]["work"] for item in completed] == [{"rollouts": 4}, {"rollouts": 1}]
    assert completed[1]["budget_cost"] == {"generate_samples": 3}

    answer = {"means": _means(zip(RATES, [first, *rest]))}
    status, accepted = _http(base, "POST", "/v1/submission", {"session_id": session, "answer": answer}, token)
    assert status in {200, 202} and accepted["ok"] is True
    deadline = time.monotonic() + 60
    while True:
        status, result = _http(base, "GET", f"/v1/result?session_id={session}", token=ADMIN_TOKEN)
        if result.get("status") not in {"pending", "scoring"} or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    assert result["status"] == "succeeded", result
    metrics = result["run_record"]["score_summary"]["metrics"]
    assert 0.9 <= metrics["score"] <= 1.0 and metrics["missing"] == []


def test_deployment_carries_simulator_settings_without_simulator_specific_code(toy, monkeypatch) -> None:
    from mini_bang.client.cli.scoring_environment import scoring_deployment_environment
    from mini_bang.client.deployment.compose import render_compose_yaml
    from mini_bang.client.deployment.k8s import render_k8s_manifests
    from mini_bang.client.cli.parser import build_parser

    assert registry.get_simulator_entry("toy_logistic").runtime_environment == ("MINI_BANG_TOY_DATA_LABEL",)
    monkeypatch.setenv("MINI_BANG_TOY_DATA_LABEL", "toy-pool-1")
    assert scoring_deployment_environment()["MINI_BANG_TOY_DATA_LABEL"] == "toy-pool-1"

    setting = ["--runtime-setting", "MINI_BANG_TOY_DATA_LABEL=toy-pool-1"]
    k8s = render_k8s_manifests(build_parser().parse_args(["k8s", "render", *setting]))
    assert '  MINI_BANG_TOY_DATA_LABEL: "toy-pool-1"' in k8s["configmap.yaml"].splitlines()
    compose = render_compose_yaml(build_parser().parse_args(["compose", "up", *setting]))
    assert '      MINI_BANG_TOY_DATA_LABEL: "toy-pool-1"' in compose.splitlines()
    for rejected in ("MINI_BANG_BACKEND=local", "MINI_BANG_TOY_TOKEN=secret", "TOY=1"):
        with pytest.raises(ValueError, match="runtime-setting"):
            render_k8s_manifests(build_parser().parse_args(["k8s", "render", "--runtime-setting", rejected]))


def test_simulator_manifest_adds_its_own_cli_command_and_deployment_option(toy, capsys) -> None:
    from mini_bang.cli import main
    from mini_bang.client.cli.parser import build_parser
    from mini_bang.client.deployment.k8s import render_k8s_manifests

    assert main(["toy-logistic"]) == 0
    assert capsys.readouterr().out.split() == [str(rate) for rate in RATES]
    k8s = render_k8s_manifests(build_parser().parse_args(["k8s", "render", "--toy-data-label", "toy-pool-2"]))
    assert '  MINI_BANG_TOY_DATA_LABEL: "toy-pool-2"' in k8s["configmap.yaml"].splitlines()
    with pytest.raises(ValueError, match="repeat an existing setting"):
        render_k8s_manifests(build_parser().parse_args(
            ["k8s", "render", "--toy-data-label", "a", "--runtime-setting", "MINI_BANG_TOY_DATA_LABEL=b"]))
