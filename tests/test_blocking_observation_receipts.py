"""Blocking HTTP calls and atomic batches retain durable retrieval after disconnect."""
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.official_protocol import remaining_budget
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.execution.progress import observe_progress
from mini_bang.framework.runtime import RunManager
from mini_bang.tasks.flow_lenia_common.api import observation_request
from tests.observation_lifecycle_support import wait_until
from tests.test_official_boundaries import _server
from tests.test_parallel_observations import answer, setup, wait_receipts


@pytest.mark.parametrize('path', ['/v1/generate_samples','/v1/simulations/generate_samples'])
def test_sigterm_blocking_http_client_keeps_receipt_and_exact_retry_costs_once(tmp_path,monkeypatch,path):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR',str(tmp_path/'durable'))
    server,session,backend,requests=setup(tmp_path,monkeypatch)
    client=None
    code='''import json,sys,urllib.request
request=urllib.request.Request(sys.argv[1],data=sys.argv[2].encode(),headers={
"Authorization":"Bearer "+sys.argv[3],"Content-Type":"application/json"})
print(urllib.request.urlopen(request).read().decode(),flush=True)
'''
    try:
        with _server(server) as endpoint:
            run=str(session.run_id);token=server.session_token(run)
            client=subprocess.Popen([sys.executable,'-c',code,endpoint+path,json.dumps(requests[0]),token],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            assert backend.wait_calls(1)
            client.terminate();client.communicate(timeout=5)
            assert client.returncode != 0
            records=get_json(endpoint+'/v1/observations?session_id='+run,token=token)['observations']
            assert len(records)==1 and records[0]['status']=='pending'
            with pytest.raises(AgentRuntimeError,match='409'):
                post_json(endpoint+'/v1/submission',{'answer':answer(session)},token=token)
            backend.release.set()
            record=wait_receipts(server,1)[0]
            # Explicit default and the other API route still recover the same
            # canonical request and original artifact, rather than debit again.
            retried=post_json(endpoint+'/v1/generate_samples',{**requests[0],'materialize':True},token=token)
            assert retried=={'payload':record['payload'],'remaining_budget':remaining_budget(session.capabilities())}
            async_retry=post_json(endpoint+'/v1/observations',requests[0],token=token)
            assert async_retry['observation_id']==record['observation_id']
            assert backend.calls==1 and session.budget_counters['generate_samples']==1
            assert session.budget_counters['observed_states']==1
    finally:
        if client is not None and client.poll() is None:
            client.terminate();client.communicate(timeout=5)
        backend.release.set();server.observations.close()


def test_atomic_batch_has_one_pending_ledger_entry_and_blocking_list_response(tmp_path,monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR',str(tmp_path/'durable'))
    server,session,backend,requests=setup(tmp_path,monkeypatch)
    batch=[{k:v for k,v in request.items() if k!='session_id'} for request in requests]
    try:
        with _server(server) as endpoint:
            run=str(session.run_id);token=server.session_token(run)
            receipt=post_json(endpoint+'/v1/observation-batches',{'requests':batch},token=token)
            assert backend.wait_calls(1)
            records=get_json(endpoint+'/v1/observations?session_id='+run,token=token)['observations']
            assert len(records)==1 and records[0]['observation_id']==receipt['observation_id']
            assert records[0]['status']=='pending' and len(records[0]['request']['requests'])==2
            with pytest.raises(AgentRuntimeError,match='409'):
                post_json(endpoint+'/v1/submission',{'answer':answer(session)},token=token)
            assert session.budget_counters['generate_samples']==2
            backend.release.set()
            completed=wait_receipts(server,1)[0]
            assert isinstance(completed['payload'],list) and len(completed['payload'])==2
            retry=post_json(endpoint+'/v1/generate_samples_batch',{'requests':batch},token=token)
            assert retry=={'payloads':completed['payload'],'remaining_budget':remaining_budget(session.capabilities())}
            assert backend.calls==1 and session.budget_counters['generate_samples']==2
    finally:
        backend.release.set();server.observations.close()


@pytest.mark.parametrize('admitted',[False,True])
def test_batch_restore_preserves_flow_npz_handle_list_shape_and_atomic_debits(tmp_path,monkeypatch,admitted):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR',str(tmp_path/'durable'))
    server,session,backend,requests=setup(tmp_path,monkeypatch)
    run=str(session.run_id);token=server.session_token(run)
    parameters={'requests':[{k:v for k,v in request.items() if k!='session_id'} for request in requests],
                'materialize':False}
    contexts=[]
    if admitted:
        originals=[observation_request(**request,instance_id=session.instance_id) for request in parameters['requests']]
        with observe_progress(lambda kind,values:contexts.extend(values['contexts']) if kind=='admitted' else None):
            session._validate_and_consume_requests(method_name='generate_samples',requests=originals)
    record={'observation_id':'observation-0001','status':'pending','request':parameters,
            'created_at':datetime.now(timezone.utc).isoformat(),'phase':'queued'}
    if admitted:record['private_contexts']=contexts
    server.observations.journal.save(run,record)
    server.observations.close()
    recreated=OfficialRunServer(run_manager=RunManager(backend=LocalBackend()),
        mode='multi_session_official',admin_token='operator-only')
    try:
        with _server(recreated) as endpoint:
            get_json(endpoint+'/v1/run?session_id='+run,token=token)
            completed=wait_until(lambda:next((r for r in recreated.observations.get(run)
                if r['status']=='completed'),None))
            assert len(completed['payload'])==2
            assert all('observation' in value and 'artifact_id' in value['observation'] for value in completed['payload'])
            for value,original in zip(completed['payload'],parameters['requests']):
                assert value['steps']==[original['start_step']]
                handle=value['observation']
                metadata=get_json(endpoint+f"/v1/artifacts/{handle['artifact_id']}/metadata?session_id={run}",token=token)
                assert metadata['size_bytes']>0
            restored=recreated._sessions[run]
            assert restored.budget_counters['generate_samples']==restored.budget_counters['observed_states']==2
            # Multi-session mode routes only by session_id; it is not part of the request identity.
            retry=post_json(endpoint+'/v1/generate_samples_batch',{**parameters,'session_id':run},token=token)
            assert retry=={'payloads':completed['payload'],'remaining_budget':remaining_budget(restored.capabilities())}
            assert restored.budget_counters['generate_samples']==2
    finally:
        recreated.observations.close()


def test_external_reference_batch_recovery_preserves_reference_only_results(tmp_path):
    from mini_bang.client.cli.observation_recovery import recovered_execution
    from mini_bang.execution.models import SimulationRequest,SimulationResult
    from mini_bang.execution.payloads import FileSystemPayloadStore
    from mini_bang.execution.runtime import RuntimeContext
    store=FileSystemPayloadStore(tmp_path/'payloads')
    originals=[SimulationRequest(simulator_id='raf',saturation=6,macro_params={'seed':seed}) for seed in (1,2)]
    refs=[store.put({'scientific_value':seed}) for seed in (1,2)]
    results=[SimulationResult(request=r,request_hash=r.request_hash,payload=None,payload_ref=ref)
             for r,ref in zip(originals,refs)]
    calls=[]
    def backend_run(requests,**kwargs):
        assert [r.to_payload() for r in requests]==[r.to_payload() for r in originals]
        return results
    def recorded(**kwargs):
        calls.append(kwargs['materialize'])
        return kwargs['result'].to_payload_handle()
    session=SimpleNamespace(run_id='same-run',trace=[],backend=SimpleNamespace(run_batch=backend_run),
        runtime_context=RuntimeContext(run_id='same-run'),runtime_services=None,
        _record_simulation_result=recorded,_lock=threading.RLock())
    server=SimpleNamespace(_session_locks={'same-run':threading.RLock()},_artifact_refs={'same-run':{}},
                           state=SimpleNamespace(save=lambda _:None))
    record={'request':{'requests':[{},{}],'materialize':False},'private_contexts':[
        {'request':r.to_payload(),'started_at':time.time(),'method_name':'generate_samples',
         'budget_before':{},'budget_after':{}} for r in originals]}
    public=recovered_execution(server,session,record)()
    assert len(public)==2 and calls==[False,False]
    assert all('artifact_id' in value and value['payload_ref']['uri'].startswith('artifact://') for value in public)
    assert 'scientific_value' not in json.dumps(public) and 'file://' not in json.dumps(public)
    assert list(server._artifact_refs['same-run'].values())==refs
