"""Observation recovery waits beyond kubelet's declared force-stop grace."""
from types import SimpleNamespace

import pytest

from mini_bang.backends import k8s_client
from mini_bang.backends.k8s_client import (
    KubernetesJobClient, OBSERVATION_TERMINATION_GRACE_SECONDS, OBSERVATION_STOP_WAIT_SECONDS,
)
from tests.test_observation_unavailable_logs import setup_backend


def stopping_backend(tmp_path, monkeypatch, *, never_stops=False):
    backend,handle,client=setup_backend(tmp_path)
    clock=[0.];events=[]
    monkeypatch.setattr(k8s_client.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(k8s_client.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    def delete(**values):
        assert values=={'namespace':'test','name':handle.job_id,'propagation_policy':'Foreground'}
        events.append(('delete',clock[0]))
    def pods(**values):
        assert values=={'namespace':'test','label_selector':'job-name='+handle.job_id}
        # Model a running PID1 which ignores TERM until kubelet forces its stop.
        phase='Running' if never_stops or clock[0]<OBSERVATION_TERMINATION_GRACE_SECONDS else 'Failed'
        events.append((phase,clock[0]))
        return SimpleNamespace(items=[SimpleNamespace(status=SimpleNamespace(phase=phase))])
    client._batch=SimpleNamespace(delete_namespaced_job=delete)
    client._core=SimpleNamespace(list_namespaced_pod=pods)
    client.stop_job=KubernetesJobClient.stop_job.__get__(client)
    original_create=client.create_job
    def create(namespace,body):
        events.append(('replacement',clock[0]))
        return original_create(namespace,body)
    client.create_job=create
    return backend,handle,client,clock,events


def test_observation_manifest_declares_grace_shorter_than_default_stop_wait(tmp_path):
    backend,handle,client=setup_backend(tmp_path)
    pod=client.jobs[handle.job_id]['spec']['template']['spec']
    assert pod['terminationGracePeriodSeconds']==5
    assert OBSERVATION_STOP_WAIT_SECONDS==30>pod['terminationGracePeriodSeconds']
    assert pod['restartPolicy']=='Never' and pod['automountServiceAccountToken'] is False
    assert pod['containers'][0]['resources']['limits']=={'cpu':'1','memory':'1Gi'}


def test_real_stop_method_waits_full_grace_before_recovery_starts_replacement(tmp_path,monkeypatch):
    backend,handle,client,clock,events=stopping_backend(tmp_path,monkeypatch)
    original=backend.result_store.get_job_input(handle.metadata['input_uri'])
    assert backend.poll(handle).state=='pending'
    assert len(client.jobs)==2
    assert ('Running',4.5) in events and ('Failed',5.) in events
    assert events[-1]==('replacement',5.)
    assert clock[0]<OBSERVATION_STOP_WAIT_SECONDS
    assert backend.result_store.get_job_input(handle.metadata['input_uri'])==original


def test_real_stop_timeout_fails_closed_without_overlapping_replacement(tmp_path,monkeypatch):
    backend,handle,client,clock,events=stopping_backend(tmp_path,monkeypatch,never_stops=True)
    with pytest.raises(OSError,match='replacement not started'):backend.poll(handle)
    assert clock[0]==OBSERVATION_STOP_WAIT_SECONDS and len(client.jobs)==1
    assert not any(kind=='replacement' for kind,_ in events)


@pytest.mark.parametrize('wait_seconds',[0,5,float('inf'),float('nan')])
def test_stop_wait_must_exceed_grace_before_deletion_is_requested(wait_seconds):
    client=object.__new__(KubernetesJobClient)
    client._batch=SimpleNamespace(delete_namespaced_job=lambda **k:pytest.fail('Invalid stop wait deleted a Job'))
    with pytest.raises(ValueError,match='exceed'):client.stop_job('test','same-job',wait_seconds=wait_seconds)
