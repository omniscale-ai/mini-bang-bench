from copy import deepcopy
from types import SimpleNamespace

import pytest

from mini_bang.backends.base import JobStatus
from mini_bang.framework.scoring_recovery import run_durable


class FakeBackend:
    def __init__(self):
        self.config = SimpleNamespace(namespace='test',polling_interval_seconds=0,worker_mode='legacy_jobs')
        self.inputs, self.created, self.exists = [], [], False
        self.crash_at, self.crashed = 'poll', False
        self.result_store = SimpleNamespace(job_result_uri=lambda run,job:f'saved/{run}/{job}',
            put_job_input=self.put,stat=lambda _:False)
        self._client = SimpleNamespace(create_job=self.create)

    def put(self, run, job, manifest):
        self.inputs.append(deepcopy(manifest)); return f'inputs/{run}/{job}'

    def _job_input_manifest(self, job, **identity):
        return {'run_id':job.run_id,'task_id':job.task_id,'instance_id':job.instance_id,
                'submission':job.submission, **identity}

    def _job_manifest(self, job, uri, context): return {'name':job,'input':uri}

    def create(self, namespace, manifest):
        self.created.append(manifest)
        if self.crash_at == 'dispatch' and not self.crashed:
            self.crashed = True; raise ConnectionError('runner lost before dispatch')
        if self.exists:
            error = RuntimeError('already exists'); error.status = 409; raise error
        self.exists = True

    def poll(self, handle):
        if self.crash_at == 'poll' and not self.crashed:
            self.crashed = True; raise ConnectionError('runner lost after dispatch')
        return JobStatus(state='succeeded')

    def fetch(self, handle): return {'validation':{'score':.7},'job_id':handle.job_id}


@pytest.mark.parametrize('crash_at',['dispatch','poll'])
def test_restart_preserves_same_prepared_or_running_job_without_new_model(tmp_path, crash_at):
    backend = FakeBackend(); backend.crash_at = crash_at
    job = SimpleNamespace(run_id='readable-run',task_id='example/task',instance_id='reviewed',
                          submission={'answer':{'scientific_result':42}},scoring_context=None)
    with pytest.raises(ConnectionError): run_durable(backend,job,tmp_path)
    original = deepcopy(backend.inputs[0])
    result = run_durable(backend,job,tmp_path)
    assert len(backend.inputs) == 1 and backend.inputs[0] == original
    assert backend.created[0] == backend.created[1]
    assert result['job_id'] == original['job_id']
    assert result['job_id'].startswith('score-example-task-')
    assert (tmp_path/'readable-run/scoring.json').exists()


def test_changed_answer_cannot_reuse_prior_scientific_score(tmp_path):
    backend = FakeBackend(); backend.crash_at = None
    job = SimpleNamespace(run_id='readable-run',task_id='example/task',instance_id='reviewed',
                          submission={'answer':1},scoring_context=None)
    run_durable(backend,job,tmp_path)
    job.submission = {'answer':2}
    with pytest.raises(OSError,match='scientific inputs differ: submission'):
        run_durable(backend,job,tmp_path)
    assert len(backend.created) == 1
