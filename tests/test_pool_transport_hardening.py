"""Original identities survive long-poll outages, backpressure and archive reads."""
import base64
import json
import time

import pytest

from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.pool.client import PoolClient
from mini_bang.pool.scheduler import PoolScheduler
from mini_bang.pool.journal import PoolJournal
from mini_bang.pool.protocol import MAX_ARCHIVE_CHUNK_BYTES, PoolBackpressure, PoolUnavailable
from tests.test_persistent_k8s_backend import serving, TOKEN, ForbiddenKubernetesClient
from tests.test_pool_scheduler_hardening import QUEUED_AT, Slot, scheduler, wait, work


def requests(count):
    return [SimulationRequest('raf',4,runs=1,macro_params={'seed':i}) for i in range(count)]


def finish(store,item):
    manifest=item['payload']['manifest']
    store.put_job_result(manifest['run_id'],item['work_id'],{
        'job_id':item['work_id'],'run_id':manifest['run_id'],'status':'succeeded',
        'results':[{'request':raw,'payload':{'index':raw['macro_params']['seed']}} for raw in manifest['requests']]})


def backend(tmp_path,**options):
    return K8sBackend(config=K8sBackendConfig(namespace='fixed',worker_image='fixed',
        polling_interval_seconds=.01,**options),result_store=FileSystemResultStore(tmp_path),
        k8s_client=ForbiddenKubernetesClient())


def test_long_poll_outage_is_caught_and_canonical_result_wins_without_new_identity(tmp_path):
    value=backend(tmp_path);sent=[]
    class Client:
        def submit(self,item):sent.append(item);return {'state':'pending'}
        def status(self,identity,wait_seconds=0):
            if wait_seconds:
                finish(value.result_store,sent[0])
                raise PoolUnavailable('lost long-poll reply after commit')
            return {'state':'pending'}
    value._pool._client=Client()
    results=value.run_batch(requests(1),context=RuntimeContext(run_id='same-run'))
    assert results[0].payload=={'index':0} and len(sent)==1


def test_lost_transport_keeps_original_handle_and_bounded_backoff(tmp_path):
    value=backend(tmp_path,observation_idle_seconds=.035);sent=[]
    class Client:
        def submit(self,item):sent.append(item);return {'state':'pending'}
        def status(self,*args,**kwargs):raise PoolUnavailable('transport outage')
    value._pool._client=Client()
    with pytest.raises(ObservationInfrastructureFailure) as caught:
        value.run_batch(requests(1),context=RuntimeContext(run_id='same-run'))
    assert len({item['work_id'] for item in sent})==1
    assert caught.value.original_handle.job_id==sent[0]['work_id']
    assert len(sent)<15


def test_paid_batch_larger_than_queue_capacity_admits_every_original_child(tmp_path,monkeypatch):
    store=FileSystemResultStore(tmp_path/'results');slot=Slot();seen=[]
    def execute(item):
        time.sleep(.025);seen.append(item['work_id']);finish(store,item)
        return {'work_id':item['work_id'],'state':'succeeded'}
    slot.execute=execute
    service=scheduler(tmp_path/'scheduler',[slot],max_pending=1)
    monkeypatch.setenv('MINI_BANG_POOL_TOKEN',TOKEN)
    with serving(service) as url:
        value=K8sBackend(config=K8sBackendConfig(namespace='fixed',worker_image='fixed',pool_url=url,
            batch_chunk_size=1,polling_interval_seconds=.01),result_store=store,k8s_client=ForbiddenKubernetesClient())
        actual=value.run_batch(requests(7),context=RuntimeContext(run_id='paid-batch'))
        assert [result.payload['index'] for result in actual]==list(range(7))
        assert len(seen)==len(set(seen))==7
        assert len(list((tmp_path/'results/job-inputs/paid-batch').glob('*.json')))==7


def test_queue_backpressure_is_distinct_http429_and_duplicate_id_still_rejoins(tmp_path):
    slot=Slot();slot.release.clear();service=scheduler(tmp_path,[slot],max_pending=1)
    with serving(service) as url:
        client=PoolClient(url,TOKEN);one=work('paid-first');client.submit(one)
        assert slot.started.wait(1)
        try:
            with pytest.raises(PoolBackpressure):client.submit(work('paid-second'))
            assert client.submit(one)['work_id']==one['work_id']
        finally:slot.release.set()


def test_archive_route_is_authenticated_terminal_and_bound_to_agent_slot(tmp_path):
    slot=Slot(role='agent');reads=[];payload=b'retained agent logs'
    def archive(identity,offset=0,limit=MAX_ARCHIVE_CHUNK_BYTES):
        reads.append((identity,offset,limit));data=payload[offset:offset+limit]
        return {'offset':offset,'total_size':len(payload),'eof':offset+len(data)>=len(payload),
                'data_base64':base64.b64encode(data).decode()}
    slot.read_archive=archive
    journal=PoolJournal(tmp_path)
    journal.put({'work':work('finished-agent',role='agent',padding='PRIVATE-env'),
        'state':'succeeded','slot':slot.name,'created_at':QUEUED_AT})
    journal.put({'work':work('finished-observation'),
        'state':'succeeded','slot':slot.name,'created_at':QUEUED_AT})
    service=scheduler(tmp_path,[slot])
    with serving(service) as url:
        with pytest.raises(PoolUnavailable):PoolClient(url,'wrong-token').archive_chunk('finished-agent')
        assert not reads
        value=PoolClient(url,TOKEN).archive_chunk('finished-agent',offset=2,limit=5)
        assert base64.b64decode(value['data_base64'])==payload[2:7]
        assert reads==[('finished-agent',2,5)] and 'PRIVATE' not in json.dumps(value)
        with pytest.raises(ValueError):PoolClient(url,TOKEN).archive_chunk('finished-observation')
        with pytest.raises(ValueError):PoolClient(url,TOKEN).archive_chunk('finished-agent',limit=MAX_ARCHIVE_CHUNK_BYTES+1)


def test_cli_passes_static_resource_json_and_invalid_json_never_starts_service(tmp_path,monkeypatch):
    from mini_bang.pool import __main__ as main
    import mini_bang.pool.worker as worker
    captured=[]
    monkeypatch.setattr(main,'token_from_env',lambda:'private-static-token')
    monkeypatch.setattr(worker,'WorkerService',lambda *args,**kwargs:captured.append(kwargs) or object())
    monkeypatch.setattr(main,'serve',lambda *args,**kwargs:None)
    monkeypatch.setenv('MINI_BANG_POOL_RESOURCE_LIMITS','{"cpu":"2","memory":"8Gi","gpu":0}')
    main.main(['observation','--state-root',str(tmp_path)])
    assert captured[0]['resource_limits']=={'cpu':'2','memory':'8Gi','gpu':0}
    monkeypatch.setenv('MINI_BANG_POOL_RESOURCE_LIMITS','not-json')
    with pytest.raises(ValueError):main.main(['observation'])
    assert len(captured)==1
