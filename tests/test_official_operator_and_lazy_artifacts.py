import time

import pytest

from mini_bang.agent_runtime import AgentRuntimeError,RemotePayloadHandle,get_json,post_json
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.execution.payloads import FileSystemPayloadStore
from tests.test_official_boundaries import DummySession,_server
from tests.test_parallel_observations import setup


def test_json_legacy_and_lazy_handle_load_share_safe_encoder(tmp_path):
    store=FileSystemPayloadStore(tmp_path/'payloads')
    ref=store.put({'observed':[1,2,3]},key='readable-observation')
    server=OfficialRunServer(DummySession());server._artifact_refs['run-1']['artifact-1']=ref
    token=server.session_token('run-1')
    try:
        with _server(server) as endpoint:
            for suffix in ('/json',''):
                result=get_json(endpoint+'/v1/artifacts/artifact-1'+suffix,token=token)
                assert result['payload']=={'observed':[1,2,3]}
            handle=RemotePayloadHandle(runner_url=endpoint,artifact_id='artifact-1',
                session_id='run-1',session_token=token,metadata={'codec':'json'})
            assert handle.load()=={'observed':[1,2,3]}
    finally:server.observations.close()


def test_admin_repairs_three_failed_retries_same_receipt_and_budget(tmp_path,monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR',str(tmp_path/'durable'))
    old,session,backend,requests=setup(tmp_path,monkeypatch);old.observations.close()
    server=OfficialRunServer(session,mode='multi_session_official',admin_token='operator-admin')
    run=str(session.run_id);token=server.session_token(run);backend.release.set()
    actual=backend.run_batch;attempts=[];broken=[True]
    def backend_work(values,**kwargs):
        attempts.append([request.to_payload() for request in values])
        if broken[0]:raise ValueError('Private decoder failed after admission')
        return actual(values,**kwargs)
    backend.run_batch=backend_work
    def wait(predicate,seconds=12):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            value=predicate()
            if value:return value
            time.sleep(.02)
        pytest.fail('Expected observation recovery phase did not arrive')
    try:
        with _server(server) as endpoint:
            receipt=post_json(endpoint+'/v1/observations',requests[0],token=token)
            stuck=wait(lambda:next((r for r in server.observations.get(run)
                if r.get('requires_operator')),None))
            assert stuck['status']=='pending' and len(attempts)==4
            with pytest.raises(AgentRuntimeError,match='401'):
                post_json(endpoint+'/v1/observations/recover',{'session_id':run},token=token)
            broken[0]=False
            post_json(endpoint+'/v1/observations/recover',{'session_id':run},token='operator-admin')
            complete=wait(lambda:next((r for r in server.observations.get(run)
                if r['status']=='completed'),None))
            assert complete['observation_id']==receipt['observation_id']
            assert len(attempts)==5 and all(value==attempts[0] for value in attempts)
            assert session.budget_counters['generate_samples']==1
            assert session.budget_counters['observed_states']==1
    finally:server.observations.close()
