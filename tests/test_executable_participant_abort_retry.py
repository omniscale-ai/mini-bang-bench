"""`executable-forced-hold`, decision (a): participant self-abort is retried once, then scores 0."""
import json
from types import SimpleNamespace

import pytest

from mini_bang.execution.payloads import PAYLOAD_REF_SENTINEL
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.services import ExecutionServices
from mini_bang.framework.executable import ExecutionResult
from mini_bang.framework.execution_watchdog import ExecutionAborted, VERSION, policy
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner
from mini_bang.framework.scoring import ScoringServices, score_submission
from mini_bang.framework.task import TaskSubmission

PARTICIPANT = {'outcome': 'execution_aborted', 'abort_reason': 'worker_terminated_by_signal',
               'attribution_evidence': 'participant_process', 'exit_code': -9}
TRUSTED = {'outcome': 'execution_aborted', 'abort_reason': 'supervisor_or_child_bootstrap_failure',
           'attribution_evidence': 'trusted_monitor', 'exit_code': None}
FEATURES = {'features': {factor: [[1.0]] for factor in ('1', '2', '4')}}


class ScriptedClient:
    """Fake Kubernetes API: the n-th created executable Job ends with the n-th scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes, self.jobs, self.deleted = list(outcomes), [], []

    def create_config_map(self, *args):
        pass

    def create_job(self, namespace, value):
        self.jobs.append(value)

    def read_job_status(self, *args):
        return {'failed': 1}

    def read_job_logs(self, namespace, job_id, **kwargs):
        names = [job['metadata']['name'] for job in self.jobs]
        outcome = self.outcomes[names.index(job_id)]
        if outcome == 'success':
            result = {'exit_code': 0, 'output': FEATURES,
                      'metadata': {'policy_version': VERSION, 'outcome': 'completed'}}
        else:
            result = {'exit_code': 124, 'metadata': {'policy_version': VERSION, **outcome}}
        return '__MINI_BANG_EXECUTION_RESULT__' + json.dumps({'status': 'failed', 'execution_result': result})

    def delete_job(self, *args):
        self.deleted.append('job')

    def delete_config_map(self, *args):
        self.deleted.append('config')


def score(tmp_path, monkeypatch, client):
    """Run the unchanged representation scorer over the real K8s runner and a scripted cluster."""
    from mini_bang.tasks.flow_lenia_common import representation
    runner = K8sExecutableRunner(config=K8sExecutableConfig(
        namespace='test', image='test-image', execution_mode='legacy_jobs', staging_image='test-stager',
        result_store_config={'kind': 'filesystem', 'root': str(tmp_path / 'results')},
        execution_policy=policy({'version': VERSION})), k8s_client=client)
    services = ScoringServices(ExecutionServices(), RuntimeContext(), executable_runner=runner)
    payload = {'observation': {PAYLOAD_REF_SENTINEL: True, 'ref': {'codec': 'bytes', 'size_bytes': 1, 'sha256': 'x'}},
               'shapes': {'A': [2, 1], 'P': [2, 1], 'Food': [2, 1]},
               'dtypes': {'A': 'float16', 'P': 'float16', 'Food': 'float32'}}
    monkeypatch.setattr(services, 'run_batch',
                        lambda requests: [SimpleNamespace(materialize_payload=lambda: payload) for _ in requests])
    monkeypatch.setattr(representation, 'load_pool', lambda _: {})
    monkeypatch.setattr(representation, 'observation_settings', lambda: ('fixture', 'checkpoints'))
    monkeypatch.setattr(representation, 'observation_request', lambda *a, **k: SimpleNamespace(
        sample_params={'start_step': 0, 'end_step': 100, 'step': 100}))
    monkeypatch.setattr(representation, 'checkpoint_ranges', lambda *a, **k: [(0, 100, 100)])
    monkeypatch.setattr('mini_bang.framework.k8s_staging.K8sInputStage.prepare', lambda self: None)
    scorer = representation.RepresentationScorer()
    # The real metric needs a golden pool; a fixed stand-in shows the retry's score is used as is.
    monkeypatch.setattr(scorer, 'feature_protocol', lambda *a: (lambda *b: {'score': 0.7, 'feature_dimension': 1}, 128))
    case = {'case_id': 'test-window', 't1': 0, 't2': 100, 'observation_window': [0, 100]}
    validation = score_submission(
        scorer, task_spec=SimpleNamespace(config={'api': {'max_states_per_request': 64,
                                                         'max_request_raw_steps': 180000}}),
        instance_spec=SimpleNamespace(instance_id='test-instance'),
        submission=TaskSubmission(answer={'code': 'def featurize(w): return [0]'}), run_record=None,
        gold_store=SimpleNamespace(get=lambda key: [case] if key == 'cases' else None), scoring_services=services)
    return validation, services


def test_participant_abort_twice_scores_zero_as_failed_submission(tmp_path, monkeypatch):
    client = ScriptedClient([PARTICIPANT, PARTICIPANT])
    validation, services = score(tmp_path, monkeypatch, client)
    assert validation.success is False
    assert validation.metrics['score'] == 0.0 and validation.metrics['valid_submission'] is False
    names = [job['metadata']['name'] for job in client.jobs]
    assert len(names) == 2 and len(set(names)) == 2  # A fresh one-shot Pod for the retry.
    resources = [job['spec']['template']['spec']['containers'][0]['resources'] for job in client.jobs]
    assert resources[0] == resources[1]  # Same limits.
    failure = [e for e in services.trace if e['event'] == 'scoring_executable_participant_abort_failure']
    record = failure[0]['participant_abort_retry']
    assert record['resolution'] == 'scored as failed submission' and record['attempts'] == 2
    assert [a['job_id'] for a in record['participant_aborts']] == names
    assert [e['attempt'] for e in services.trace if e['event'] == 'scoring_executable_finished'] == [1, 2]
    assert services.resource_usage['executable_runs'] == 2


def test_participant_abort_then_success_keeps_the_real_score(tmp_path, monkeypatch):
    client = ScriptedClient([PARTICIPANT, 'success'])
    validation, services = score(tmp_path, monkeypatch, client)
    assert validation.success is True
    assert validation.metrics['score'] == 0.7 and validation.metrics['valid_submission'] is True
    assert len(client.jobs) == 2
    finished = [e for e in services.trace if e['event'] == 'scoring_executable_finished']
    record = finished[-1]['metadata']['participant_abort_retry']
    assert record['resolution'] == 'retry result used as is' and record['attempts'] == 2
    assert record['participant_aborts'][0]['reason'] == 'worker_terminated_by_signal'


def test_trusted_monitor_abort_is_still_held_without_retry(tmp_path, monkeypatch):
    client = ScriptedClient([TRUSTED, 'success'])
    with pytest.raises(ExecutionAborted, match='supervisor_or_child_bootstrap_failure') as caught:
        score(tmp_path, monkeypatch, client)
    assert len(client.jobs) == 1 and not client.deleted  # Held and preserved for recovery.
    assert 'participant_abort_retry' not in caught.value.metadata


class RaisingRunner:
    """Runner stub: each call raises or returns the next scripted item."""

    def __init__(self, items):
        self.items, self.calls = list(items), 0

    def run(self, *args):
        item = self.items[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def aborted(evidence, reason):
    metadata = {'job_id': f'job-{reason}'}
    if evidence:
        metadata['attribution_evidence'] = evidence
    return ExecutionAborted(reason, metadata=metadata)


@pytest.mark.parametrize('evidence', ['trusted_monitor', None])
def test_retry_that_ends_without_participant_evidence_is_held_with_both_attempts(evidence):
    runner = RaisingRunner([aborted('participant_process', 'submitted_code_cpu_safety_fuse'),
                            aborted(evidence, 'scheduling_or_startup_stalled')])
    services = ScoringServices(ExecutionServices(), RuntimeContext(), executable_runner=runner)
    with pytest.raises(ExecutionAborted) as caught:
        services.run_executable(None, {}, None)
    record = caught.value.metadata['participant_abort_retry']
    assert record['attempts'] == 2 and record['resolution'].startswith('held')
    assert record['participant_aborts'][0]['reason'] == 'submitted_code_cpu_safety_fuse'


def test_unattributed_abort_is_held_on_first_attempt():
    runner = RaisingRunner([aborted(None, 'pool_transport_failure'), ExecutionResult(exit_code=0)])
    services = ScoringServices(ExecutionServices(), RuntimeContext(), executable_runner=runner)
    with pytest.raises(ExecutionAborted, match='pool_transport_failure'):
        services.run_executable(None, {}, None)
    assert runner.calls == 1


HELD_REASONS = {'trusted_precheck_failed', 'worker_bootstrap_failure'}
REASONS = ['trusted_precheck_failed', 'worker_bootstrap_failure', 'worker_version_mismatch', 'trusted_loader_failure',
           'cgroup_oom', 'submitted_code_cpu_safety_fuse', 'whole_worker_cpu_safety_fuse',
           'worker_terminated_by_signal', 'nonzero_without_confirmed_participant_contract']
# Participant code that would leave this marker if it ever ran.
MARKER = 'open(__import__("os").environ["MINI_BANG_FEATURE_STATUS_DIR"] + "/../participant-ran", "w").close()\n'
FORGE_VERSION = MARKER + """import json, os
path = os.path.join(os.environ['MINI_BANG_FEATURE_STATUS_DIR'], 'admission.json')
value = json.load(open(path))
value['worker_version'] = 'forged-version'
open(path + '.forged', 'w').write(json.dumps(value))
os.replace(path + '.forged', path)
def featurize(window):
    return [1.0]
"""
FORGE_LOADER = MARKER + """import numpy
def broken(*args, **kwargs):
    raise OSError('participant-made loader complaint')
numpy.load = broken
def featurize(window):
    return [float(len(list(window)))]
"""


def run_harness(tmp_path, code, *, files=None, expected=None, **settings):
    """The real harness under the real supervisor, with the trusted precheck configured."""
    import sys
    from mini_bang.framework.execution_watchdog import supervise
    from mini_bang.tasks.flow_lenia_common import representation_worker as worker
    from tests.test_execution_safety_fuses import quick_policy, runtime
    from tests.test_representation_worker_performance import chunks
    if files is None:
        files = [[str(tmp_path / name) for name in names] for names in chunks(tmp_path, count=1)]
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'code': code, 'files': files}))
    precheck = {'harness': worker.__file__, 'input': str(source)}
    supervise([sys.executable, worker.__file__, str(source), str(tmp_path / 'features.json')],
              quick_policy(idle_seconds=5, warning_seconds=2, expected_worker_version=expected or worker.WORKER_VERSION,
                           trusted_precheck=precheck, **settings), tmp_path / 'monitor', 'execution')
    return runtime(tmp_path / 'monitor')


def watchdog_record(tmp_path, monkeypatch, reason):
    """Produce `reason` with the real supervisor (and the real harness where it matters)."""
    import sys
    import mini_bang.framework.execution_watchdog as watchdog
    from mini_bang.framework.execution_watchdog import supervise
    from mini_bang.tasks.flow_lenia_common import representation_worker as worker
    from tests.test_execution_safety_fuses import quick_policy, runtime
    if reason == 'trusted_precheck_failed':  # A genuinely unreadable staged checkpoint.
        (tmp_path / 'broken.npz').write_bytes(b'not a zip archive')
        return run_harness(tmp_path, MARKER + 'def featurize(window):\n    return [1.0]\n',
                           files=[[str(tmp_path / 'broken.npz')]])
    if reason == 'worker_version_mismatch':  # Forged by submitted code after the precheck passed.
        return run_harness(tmp_path, FORGE_VERSION)
    if reason == 'trusted_loader_failure':  # Made by submitted code after the precheck passed.
        return run_harness(tmp_path, FORGE_LOADER)
    if reason == 'submitted_code_cpu_safety_fuse':
        return run_harness(tmp_path, 'def featurize(window):\n    while True: pass\n', submitted_cpu_seconds=.05)
    settings, command = {}, [sys.executable, '-c', 'pass']
    if reason == 'worker_bootstrap_failure':  # The harness never admitted itself.
        settings = {'expected_worker_version': worker.WORKER_VERSION}
    elif reason == 'worker_terminated_by_signal':
        command = [sys.executable, '-c', 'import os,signal;os.kill(os.getpid(),signal.SIGKILL)']
    elif reason == 'nonzero_without_confirmed_participant_contract':
        command = [sys.executable, '-c', 'raise SystemExit(3)']
    else:
        key, values = (('memory.events', ['oom 0\noom_kill 0', 'oom 1\noom_kill 1']) if reason == 'cgroup_oom'
                       else ('cpu.stat', ['usage_usec 1000000', 'usage_usec 1900000']))
        samples, last = iter({key: value} for value in values), {}

        def measured():
            nonlocal last
            last = next(samples, last)
            return last
        monkeypatch.setattr(watchdog, 'resources', measured)
        if reason == 'whole_worker_cpu_safety_fuse':
            command, settings = [sys.executable, '-c', 'import time;time.sleep(30)'], {'whole_worker_cpu_seconds': .5}
    supervise(command, quick_policy(idle_seconds=5, warning_seconds=2, **settings), tmp_path / 'monitor', 'execution')
    return runtime(tmp_path / 'monitor')


@pytest.mark.parametrize('reason', REASONS)
def test_each_watchdog_reason_is_held_or_retried_then_scored_zero(tmp_path, monkeypatch, reason):
    record = watchdog_record(tmp_path, monkeypatch, reason)
    held = reason in HELD_REASONS
    assert record['abort_reason'] == reason and record['outcome'] == 'execution_aborted'
    assert record['attribution_evidence'] == ('harness_reported_infrastructure' if held else 'participant_process')
    if reason in {'trusted_precheck_failed', 'worker_version_mismatch', 'trusted_loader_failure'}:
        assert record['trusted_precheck']['passed'] is (reason != 'trusted_precheck_failed')
        assert (tmp_path / 'monitor' / 'participant-ran').exists() is (reason != 'trusted_precheck_failed')
    runner = RaisingRunner([ExecutionAborted(reason, metadata=dict(record)) for _ in range(2)])
    services = ScoringServices(ExecutionServices(), RuntimeContext(), executable_runner=runner)
    if held:  # Infrastructure fault: held, never retried or charged to the participant.
        with pytest.raises(ExecutionAborted, match=reason):
            services.run_executable(None, {}, None)
        assert runner.calls == 1
    else:  # The limits are the rules, and forged complaints are the participant's: retry, then 0.
        result = services.run_executable(None, {}, None)
        assert runner.calls == 2 and result.exit_code != 0
        assert result.metadata['participant_abort_retry']['resolution'] == 'scored as failed submission'


def test_precheck_rejects_a_wrong_shipped_harness_before_submitted_code_runs(tmp_path):
    record = run_harness(tmp_path, MARKER + 'def featurize(window):\n    return [1.0]\n',
                         expected='expected-harness-version')
    assert record['abort_reason'] == 'trusted_precheck_failed' and record['worker_pid'] is None
    assert record['attribution_evidence'] == 'harness_reported_infrastructure'
    assert 'harness version' in record['trusted_precheck']['error']
    assert not (tmp_path / 'monitor' / 'participant-ran').exists()


def test_harness_admits_itself_before_submitted_code_is_loaded(tmp_path, monkeypatch, capfd):
    from mini_bang.tasks.flow_lenia_common import representation_worker as worker
    from tests.test_representation_worker_performance import chunks
    monkeypatch.setenv('MINI_BANG_FEATURE_STATUS_DIR', str(tmp_path / 'state'))
    code = ('import os\nADMITTED = os.path.exists(os.path.join(os.environ["MINI_BANG_FEATURE_STATUS_DIR"], '
            '"admission.json"))\ndef featurize(window):\n    return [float(ADMITTED)]\n')
    result = worker.extract({'files': chunks(tmp_path, count=1), 'code': code}, workers=1, checkpoint_root=tmp_path)
    assert all(row == [1.0] for matrix in result['features'].values() for row in matrix)
    order = [json.loads(line[len(worker.EVENT_PREFIX):]) for line in capfd.readouterr().err.splitlines()
             if line.startswith(worker.EVENT_PREFIX)]
    phases = [item.get('phase') if item['event'] == 'progress' else item['event'] for item in order]
    assert phases.index('admission') < phases.index('submitted_initialization')


def test_submitted_code_cannot_turn_an_admitted_run_into_a_bootstrap_hold(tmp_path):
    import sys
    from mini_bang.framework.execution_watchdog import supervise
    from tests.test_execution_safety_fuses import quick_policy, runtime
    code = """import json, os, socket
state = os.environ['MINI_BANG_FEATURE_STATUS_DIR']
def send(value):
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as stream:
        stream.sendto(json.dumps(value).encode(), os.environ['MINI_BANG_FEATURE_PROGRESS_SOCKET'])
os.makedirs(state, exist_ok=True)
admission = {'pid': os.getpid(), 'worker_version': 'v', 'allocated_workers': 1}
open(os.path.join(state, 'admission.json'), 'w').write(json.dumps(admission))
send({'event': 'admission', 'worker_version': 'v', 'allocated_workers': 1})  # The harness.
send({'event': 'admission', 'worker_version': None, 'allocated_workers': None})  # Submitted code.
os.remove(os.path.join(state, 'admission.json'))
"""
    supervise([sys.executable, '-c', code], quick_policy(expected_worker_version='v'), tmp_path, 'execution')
    record = runtime(tmp_path)
    assert record['abort_reason'] == 'missing_trusted_harness_terminal'
    assert record['attribution_evidence'] == 'participant_process'


def test_k8s_code_pod_supervisor_is_configured_with_the_trusted_precheck(tmp_path):
    runner = K8sExecutableRunner(config=K8sExecutableConfig(
        namespace='test', image='test-image', execution_mode='legacy_jobs',
        result_store_config={'kind': 'filesystem', 'root': str(tmp_path)},
        execution_policy=policy({'version': VERSION, 'expected_worker_version': 'v'})), k8s_client=ScriptedClient([]))
    configured = json.loads(runner._config_map_manifest('job', 'code', {})['data']['execution-policy.json'])
    assert configured['trusted_precheck'] == {'harness': '/mini-bang/input/submission.py',
                                              'input': '/mini-bang/input/input.json'}
