from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mini_bang.tasks.raf_mechanism_prob.smoke_agent import E2ESmokeAgent
from mini_bang.backends.local import LocalBackend
from mini_bang.execution.engines import ProcessPoolEngine, SequentialEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.framework.runtime import RunManager
from tests.synthetic_gold import use_synthetic_gold


# The infrastructure smoke runs the default closure vs size instance (network families with
# hidden p) on synthetic gold. The smoke agent probes one family and submits a constant.
SMOKE_TASK = "raf/mechanism-prob"


def _assert_default_instance_scored(result) -> None:
    from mini_bang.tasks.loader import TaskLoader

    assert result.record.instance_id == TaskLoader().get_spec(SMOKE_TASK).default_instance_id
    metrics = result.validation.metrics
    assert metrics["primary_metric"] == "skill_vs_best_shared_answer"
    # A constant answer has no skill over the best shared answer.
    assert metrics["skill_vs_best_shared_answer"] == 0.0
    assert result.validation.success is False
    assert result.record.score_summary["success"] is False


def test_local_sequential_e2e(monkeypatch) -> None:
    use_synthetic_gold(monkeypatch)
    result = _run_local(SequentialEngine())

    _assert_default_instance_scored(result)


def test_local_process_pool_e2e(monkeypatch) -> None:
    use_synthetic_gold(monkeypatch)
    engine = ProcessPoolEngine(max_workers=2, reuse_strategy="none")
    try:
        try:
            result = _run_local(engine)
        except PermissionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip(str(exc))
            raise
    finally:
        engine.close()

    _assert_default_instance_scored(result)


def test_smoke_agent_uses_the_public_raf_regime_and_calls_the_task_api():
    calls = []
    task = SimpleNamespace(metadata={'saturations': [3, 6], 'macro_params': {'M0': 2, 'p': .15}},
        api=SimpleNamespace(generate_samples=lambda **kwargs: calls.append(kwargs)))
    answer = E2ESmokeAgent().solve(task).answer['probabilities']
    assert answer['3'] == pytest.approx(.15)
    assert answer['6'] == pytest.approx((1 - .85**4)**4)
    assert len(calls) == 1 and calls[0]['extras'] == ['is_raf']


def test_smoke_agent_probes_one_family_and_answers_every_family():
    calls = []
    task = SimpleNamespace(metadata={'saturations': [20, 30], 'network_families': ['family-1', 'family-2'],
                                     'experimental_domain': {'saturation': {'min': 3, 'max': 12}}},
        api=SimpleNamespace(generate_samples=lambda **kwargs: calls.append(kwargs)))
    answer = E2ESmokeAgent().solve(task).answer['probabilities']
    assert answer == {'family-1': {'20': 0.5, '30': 0.5}, 'family-2': {'20': 0.5, '30': 0.5}}
    assert len(calls) == 1 and calls[0]['saturation'] == 12
    assert calls[0]['macro_params'] == {'family': 'family-1', 'seed': 12345}


def test_smoke_agent_addresses_a_recorded_family_network_by_index():
    calls = []
    task = SimpleNamespace(metadata={'saturations': [20], 'network_families': ['family-1'],
                                     'observation_pool': 'probability-test-families'},
        api=SimpleNamespace(generate_samples=lambda **kwargs: calls.append(kwargs)))
    E2ESmokeAgent().solve(task)
    assert calls[0]['macro_params'] == {'family': 'family-1', 'network_index': 0}


def test_smoke_agent_rejects_missing_public_parameters_before_sampling():
    task = SimpleNamespace(metadata={'saturations': [6]},
        api=SimpleNamespace(generate_samples=lambda **kwargs: pytest.fail('No public RAF regime')))
    with pytest.raises(ValueError, match='public M0, p'):
        E2ESmokeAgent().solve(task)


def test_real_k8s_backend_simulation_e2e() -> None:
    status = _real_k8s_status()

    assert status["simulation"]["status"] == "CORRECT"
    assert status["simulation"]["payload_ref"] is True


def test_real_k8s_run_batch_parallelism_e2e() -> None:
    status = _real_k8s_status()

    assert status["batch_parallelism"]["status"] == "CORRECT"
    assert status["batch_parallelism"]["job_count"] > 1
    assert status["batch_parallelism"]["ordered_results"] is True


def test_real_k8s_submit_batch_parallelism_e2e() -> None:
    status = _real_k8s_status()

    batch = status["batch_parallelism"]
    assert batch["status"] == "CORRECT"
    assert batch["submit_batch_created_multiple_jobs"] is True
    assert batch["submit_batch_job_count"] > 1
    assert batch["submit_batch_max_parallel_jobs_enforced"] is True
    assert batch["submit_batch_ordered_fetch"] is True


def test_real_k8s_scoring_e2e() -> None:
    status = _real_k8s_status()

    assert status["scoring"]["status"] == "CORRECT"
    assert status["scoring"]["trace_separate"] is True


def test_real_k8s_executable_runner_e2e() -> None:
    status = _real_k8s_status()

    assert status["executable"]["status"] == "CORRECT"
    assert status["executable"]["credentials_hidden_from_code"] is True


def _run_local(engine):
    backend = LocalBackend(executor=SimulationExecutor(engine=engine))
    try:
        return RunManager(backend=backend).run(E2ESmokeAgent(), SMOKE_TASK)
    finally:
        backend.close()


_REAL_K8S_STATUS: dict | None = None


def _real_k8s_status() -> dict:
    global _REAL_K8S_STATUS
    if os.getenv("MINI_BANG_RUN_K8S_E2E") != "1":
        pytest.skip("set MINI_BANG_RUN_K8S_E2E=1 to run real Kubernetes E2E")
    if _REAL_K8S_STATUS is None:
        proc = subprocess.run(
            [sys.executable, "-m", "mini_bang.testing.kubernetes_benchmark"],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": f"{Path.cwd() / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"},
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stderr or proc.stdout)
        _REAL_K8S_STATUS = json.loads(proc.stdout)
    return _REAL_K8S_STATUS
