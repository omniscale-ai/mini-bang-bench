from __future__ import annotations

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mini_bang.execution.payloads import FileSystemPayloadStore, payload_to_transport_payload
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.framework.executable import ExecutionResult
from mini_bang.framework.task import TaskSubmission
from mini_bang.tasks.loader import TaskLoader
from mini_bang.tasks.flow_lenia_common.annotations import load_pool, gold_intervals
from mini_bang.tasks.flow_lenia_common.description import description_metrics, RubricJudgeClient
from mini_bang.tasks.flow_lenia_common.representation import scalar_metrics
from mini_bang.worker.stage_inputs import stage_file
from tests.synthetic_gold import use_synthetic_gold

REPRESENTATION, MINIBANG_METRIC = "flow_lenia/representation", "flow_lenia/minibang-metric"


@pytest.fixture(autouse=True)
def _synthetic_gold(monkeypatch):
    use_synthetic_gold(monkeypatch)


def task_parts(task_id, instance=None):
    spec = TaskLoader().get_spec(task_id)
    store = BenchmarkArtifactStore()
    visible = store.load_instance(spec, instance)
    return spec, visible, store.load_gold(visible)


class FakeServices:
    """Synthetic execution boundary test, not a real Kubernetes acceptance test."""
    def __init__(self, root, cases, *, fail=False):
        self.root, self.cases, self.fail = root, {c["case_id"]: c for c in cases}, fail
        self.store = FileSystemPayloadStore(root / "payloads")
        self.requests = []
        self.pool = load_pool()
        self.executed = False

    def run_batch(self, requests):
        self.requests = requests
        results = []
        for request in requests:
            case = self.cases[request.macro_params["case_id"]]
            # The deterministic synthetic A value makes the unit-test oracle
            # explicit. It is never packaged as benchmark data.
            label = int(bool(gold_intervals(self.pool, case)))
            sample = request.sample_params
            steps = np.arange(sample['start_step'], sample['end_step'] + 1, sample['step'])
            arrays = {key: np.full((len(steps), 2, 2), label, dtype=np.float16) for key in ("A", "P", "Food")}
            body = io.BytesIO()
            np.savez_compressed(body, steps=steps, hidden_label=[label], **arrays)
            ref = self.store.put(body.getvalue())
            payload = {"observation": payload_to_transport_payload(None, ref),
                       "shapes": {k: list(v.shape) for k, v in arrays.items()},
                       "dtypes": {k: str(v.dtype) for k, v in arrays.items()}}
            results.append(SimpleNamespace(materialize_payload=lambda p=payload: p))
        return results

    def run_executable(self, submission, config, limits, output_schema):
        assert limits.timeout_seconds == 600 and limits.memory_mb == 4096 and not limits.network_enabled
        assert set(config) - {"max_features"} == {"code", "files", "staged_artifacts", "entrypoint"}
        self.config = config
        assert "exec(" in submission["code"]  # trusted harness, not submitted source
        assert "split" not in json.dumps(config) and "trajectory" not in json.dumps(config)
        stage = self.root / "stage"
        stage.mkdir()
        for item in config["staged_artifacts"]:
            assert item["fields"] == ["A", "P", "Food"]
            stage_file(item, stage)
            with np.load(stage / item["name"], allow_pickle=False) as data:
                assert set(data.files) == {"A", "P", "Food"}
        if self.fail:
            return ExecutionResult(exit_code=137, timed_out=True)
        public = {k: v for k, v in config.items() if k != "staged_artifacts"}
        script, input_file, output_file = [self.root / n for n in ("harness.py", "input.json", "output.json")]
        # Test-only relocation; production always mounts the offline PVC path.
        location = 'checkpoint_root="/mini-bang/checkpoints"'
        assert submission["code"].count(location) == 1
        script.write_text(submission["code"].replace(location, f"checkpoint_root={str(stage)!r}"))
        input_file.write_text(json.dumps(public))
        completed = subprocess.run([sys.executable, str(script), str(input_file), str(output_file)],
                                   capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(output_file.read_text())
        output_schema(result)
        self.executed = True
        return ExecutionResult(exit_code=0, output=result)


@pytest.mark.parametrize("task_id,entrypoint", [(REPRESENTATION, "featurize"), (MINIBANG_METRIC, "score")],
                         ids=["representation-featurize", "minibang-metric-score"])
def test_hidden_metric_pipeline_runs_code_outside_scorer_and_strips_metadata(tmp_path, monkeypatch, task_id, entrypoint):
    spec, instance, gold = task_parts(task_id)
    services = FakeServices(tmp_path, gold.get("cases"))
    monkeypatch.setenv("MINI_BANG_FLOW_POOL_REF", "file:///synthetic-test-pool.json")
    monkeypatch.setenv("MINI_BANG_FLOW_OBSERVATION_MODE", "checkpoints")
    value = "[float(next(iter(window))['A'].mean())]" if entrypoint == "featurize" else "float(next(iter(window))['A'].mean())"
    code = f"def {entrypoint}(window):\n    return {value}\n"
    result = spec.create_scorer().score(task_spec=spec, instance_spec=instance, submission=TaskSubmission(answer={"code": code}),
                                      run_record=None, gold_store=gold, scoring_services=services)
    assert result.success and result.metrics["score"] == 1 and services.executed
    # Only the leave-pair-out representation default narrows the harness limit (1..8).
    assert services.config.get("max_features") == (8 if task_id == REPRESENTATION else None)
    assert {r.macro_params['case_id'] for r in services.requests} == {c['case_id'] for c in gold.get('cases')}
    assert all(r.sample_params['end_step'] - r.sample_params['start_step'] <= 180000 for r in services.requests)
    states = sum((r.sample_params['end_step'] - r.sample_params['start_step']) // r.sample_params['step'] + 1
                 for r in services.requests)
    assert result.metrics["compression_ratio"] == pytest.approx(3 * states / len(gold.get('cases')))
    assert all(r.sample_params["fields"] == ["A", "P", "Food"] for r in services.requests)


def test_metric_timeout_is_failed_submission_not_fabricated_features(tmp_path, monkeypatch):
    spec, instance, gold = task_parts(REPRESENTATION)
    monkeypatch.setenv("MINI_BANG_FLOW_POOL_REF", "file:///synthetic-test-pool.json")
    monkeypatch.setenv("MINI_BANG_FLOW_OBSERVATION_MODE", "checkpoints")
    result = spec.create_scorer().score(task_spec=spec, instance_spec=instance,
        submission=TaskSubmission(answer={"code": "while True: pass"}), run_record=None, gold_store=gold,
        scoring_services=FakeServices(tmp_path, gold.get("cases"), fail=True))
    assert not result.success and result.metrics["score"] == 0 and result.metrics["timed_out"]


def test_missing_hidden_observation_chunk_is_an_infrastructure_error(tmp_path, monkeypatch):
    spec, instance, gold = task_parts(REPRESENTATION)
    monkeypatch.setenv('MINI_BANG_FLOW_POOL_REF', 'file:///synthetic-test-pool.json')
    monkeypatch.setenv('MINI_BANG_FLOW_OBSERVATION_MODE', 'checkpoints')
    services = FakeServices(tmp_path, gold.get('cases'))
    original = services.run_batch
    services.run_batch = lambda requests: original(requests)[:-1]
    with pytest.raises(RuntimeError, match='lost a requested chunk'):
        spec.create_scorer().score(task_spec=spec, instance_spec=instance,
            submission=TaskSubmission(answer={'code': 'def featurize(window): return [1.0]'}),
            run_record=None, gold_store=gold, scoring_services=services)


def test_scalar_metric_does_not_flip_a_wrong_direction_with_a_trained_probe():
    _, _, gold = task_parts(MINIBANG_METRIC)
    pool, cases = load_pool(), gold.get("cases")
    wrong = [[-float(bool(gold_intervals(pool, c)))] for c in cases]
    assert scalar_metrics({str(k): wrong for k in (1, 2, 4)}, cases, pool)["score"] == 0
    constant = {str(k): [[3.0]] * len(cases) for k in (1, 2, 4)}
    assert scalar_metrics(constant, cases, pool)["score"] == 0.5


class FakeJudge:
    def __init__(self):
        self.calls = []

    def evaluate(self, kind, data):
        self.calls.append((kind, data))
        assert kind == "description"
        decision = lambda item: {"id": item["id"], "present": item["text"] in {"observed", "trap"},
                                 "quote": "candidate" if item["text"] in {"observed", "trap"} else "",
                                 "reason": "Synthetic encoding"}
        return {"coverage": [decision(i) for i in data["rubric"]],
                "traps": [decision(i) for i in data["traps"]],
                "claims": [{"claim": value, "quote": "candidate", "label": label,
                            "reference_quote": "reference" if label != "unverifiable" else "",
                            "reason": "Synthetic encoding"} for value, label in (
                                ("yes", "supported"), ("no", "contradicted"), ("speculation", "unverifiable"))]}


def test_description_scores_coverage_precision_uncertainty_and_traps_separately():
    event = {"rubric": [{"id": "a", "text": "observed", "weight": 3}, {"id": "b", "text": "missing", "weight": 1}],
             "reference": "reference", "traps": ["trap"], "uncertain_claims": ["speculation"]}
    judge = FakeJudge()
    result = description_metrics("candidate", event, judge, seed="case")
    assert result["rubric_recall"] == 0.75 and result["factual_precision"] == 0.5
    assert result["unverifiable_fraction"] == pytest.approx(1 / 3)
    assert result["score"] == pytest.approx(0.5) and result["trap_count"] == 1
    assert len(judge.calls) == 1
    assert result["traps"]["trap-01"]["quote"] == "candidate"
    assert result["coverage"]["a"]["present"] is True


def test_judge_requires_rubric_judge_service_and_explicit_same_model_authority():
    with pytest.raises(RuntimeError, match="rubric judge service"):
        RubricJudgeClient(SimpleNamespace(env={}))
    with pytest.raises(ValueError, match="Same-model judging lacks the exact explicit authorization"):
        RubricJudgeClient(SimpleNamespace(env={"MINI_BANG_JUDGE_URL": "http://judge", "MINI_BANG_JUDGE_TOKEN": "test",
            "MINI_BANG_JUDGE_MODEL": "same", "MINI_BANG_EVALUATED_MODEL": "same"}))
