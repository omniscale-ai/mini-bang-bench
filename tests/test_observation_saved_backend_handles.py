"""HTTP recovery resumes the persisted operation, including its retry ordinals."""
from dataclasses import asdict, replace
import json
import threading
from types import SimpleNamespace

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, post_json
from mini_bang.backends.base import CompositeJobHandle, JobHandle
from mini_bang.client.cli.observation_backend_handles import restore_handle, resume_saved_backend
from mini_bang.client.cli.observation_recovery import recovered_execution
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.execution.progress import report_progress
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.transport import request_to_transport
from tests.observation_lifecycle_support import wait_until
from tests.test_official_boundaries import _server
from tests.test_parallel_observations import answer, setup


def job(scope, name, count=1):
    return JobHandle('k8s', name, run_id=scope.run_id, task_id=scope.task_id,
        instance_id=scope.instance_id, result_uri='s3://organizer/results/'+name,
        metadata={'input_uri':'s3://organizer/inputs/'+name, 'batch_size':count})


@pytest.mark.parametrize('batch', [False, True])
def test_http_recovery_uses_same_saved_operation_and_never_reenters_submission(tmp_path, monkeypatch, batch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path/'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    original = backend.run_batch
    entered, release = threading.Event(), threading.Event()
    submitted, resumed = [], []
    saved = {}
    count = 2 if batch else 1
    def interrupted(values, *, context=None, services=None):
        submitted.append([value.to_payload() for value in values])
        if len(submitted) != 1:
            pytest.fail('Recovery attempted a new backend submission')
        children = tuple(job(context, 'original-child-'+str(i)) for i in range(count))
        handle = (CompositeJobHandle('k8s', 'original-batch', run_id=context.run_id,
            task_id=context.task_id, instance_id=context.instance_id,
            metadata={'batch_size':count}, child_handles=children,
            child_indexes=((0,), (1,))) if batch else children[0])
        saved.update(requests=list(values), context=context, services=services, handle=handle)
        report_progress('backend_job', backend_kind='k8s_handle',
            handle={'kind':'composite' if batch else 'job', 'value':asdict(handle)},
            job_id=handle.job_id, run_id=handle.run_id, result_uri=handle.result_uri)
        raise ObservationInfrastructureFailure('Transport interruption after the original operation was saved')
    def resume(handle):
        resumed.append(handle)
        entered.set()
        assert release.wait(5)
        backend.release.set()
        return original(saved['requests'], context=saved['context'], services=saved['services'])
    backend.run_batch, backend._wait_and_fetch = interrupted, resume
    try:
        with _server(server) as endpoint:
            token = server.session_token(session.run_id)
            route = '/v1/observation-batches' if batch else '/v1/observations'
            body = {'requests':[{k:v for k,v in request.items() if k!='session_id'} for request in requests]} if batch else requests[0]
            receipt = post_json(endpoint+route, body, token=token)
            assert entered.wait(4)
            record = server.observations.get(str(session.run_id), receipt['observation_id'])
            assert record['status']=='pending'
            assert 'k8s_handle' not in json.dumps(record) and 's3://' not in json.dumps(record)
            with pytest.raises(AgentRuntimeError, match='409'):
                post_json(endpoint+'/v1/submission', {'answer':answer(session)}, token=token)
            duplicate = post_json(endpoint+route, body, token=token)
            assert duplicate['observation_id']==receipt['observation_id']
            release.set()
            completed = wait_until(lambda:next((r for r in server.observations.get(str(session.run_id))
                if r['status']=='completed'),None))
            assert completed['observation_id']==receipt['observation_id']
            assert len(submitted)==len(resumed)==1 and resumed[0]==saved['handle']
            assert session.budget_counters['generate_samples']==session.budget_counters['observed_states']==count
            assert backend.calls==1
            private = server.observations.journal.load(str(session.run_id))[receipt['observation_id']]
            assert private['private_jobs'][0]['handle']['value']['job_id']==saved['handle'].job_id
    finally:
        release.set(); backend.release.set(); server.observations.close()


def test_restored_composite_checks_scope_and_complete_nonoverlapping_request_coverage():
    scope = RuntimeContext(run_id='same-run', task_id='raf/signature', instance_id='example')
    session = SimpleNamespace(run_id=scope.run_id, runtime_context=scope)
    children = (job(scope, 'first'), job(scope, 'second'))
    composite = CompositeJobHandle('k8s', 'batch', run_id=scope.run_id,
        task_id=scope.task_id, instance_id=scope.instance_id, metadata={'batch_size':2},
        child_handles=children, child_indexes=((1,), (0,)))
    encoded = {'kind':'composite', 'value':asdict(composite)}
    assert restore_handle(encoded, session, 2)==composite
    for wrong in (replace(composite, run_id='foreign'),
                  replace(composite, child_handles=(replace(children[0], run_id='foreign'),children[1])),
                  replace(composite, child_indexes=((0,), (0,))),
                  replace(composite, child_indexes=((0,),)),
                  replace(composite, child_handles=(children[0],children[0])),
                  replace(composite, child_handles=(replace(children[0],metadata={'input_uri':'input','batch_size':2}),children[1])),
                  replace(composite, child_handles=())):
        with pytest.raises(ObservationInfrastructureFailure):
            restore_handle({'kind':'composite','value':asdict(wrong)}, session, 2)


def test_saved_job_without_handle_does_not_reset_retry_budget_by_submitting_new_backend_work():
    scope = RuntimeContext(run_id='same-run', task_id='raf/signature', instance_id='example')
    backend = SimpleNamespace(run_batch=lambda *a,**kw:pytest.fail('Untracked replacement submitted'))
    session = SimpleNamespace(run_id=scope.run_id, runtime_context=scope, backend=backend)
    # Context decoding is real; a saved job without its handle cannot be reconstructed.
    from mini_bang.execution.models import SimulationRequest
    request = SimulationRequest(instance_id=scope.instance_id, simulator_id='raf', saturation=5)
    record = {'private_contexts':[{'request':request_to_transport(request)}],
        'private_jobs':[{'job_id':'old','run_id':scope.run_id,'result_uri':'s3://private/result'}]}
    with pytest.raises(ObservationInfrastructureFailure, match='original-handle'):
        recovered_execution(None, session, record)()
