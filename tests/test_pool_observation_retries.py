"""Only confirmed-dead observation children may retry one immutable paid work."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import pytest

from mini_bang.pool.journal import PoolJournal
from mini_bang.pool.process import ChildFailure
from mini_bang.pool.worker import WorkerService
from tests.test_persistent_pool_workers import work, terminal


def retry_processor(value):
    from mini_bang.pool.observation import prepare_work
    root=Path(value['payload']['records']);log=root/(value['work_id']+'-attempts.jsonl')
    with log.open('a') as handle:handle.write(json.dumps({'pid':os.getpid(),'work_id':value['work_id']})+'\n')
    first=root/(value['work_id']+'-first-attempt')
    if value['payload'].get('reject_then_die'):
        manifest,store=prepare_work(value)
        store.put_job_result(manifest['run_id'],manifest['job_id'],{'job_id':manifest['job_id'],
            'run_id':manifest['run_id'],'status':'failed','results':[],
            'error':{'category':'request_rejected','reason':'explicit canonical rejection'}})
        os._exit(23)
    if not first.exists():
        first.write_text('one original paid work')
        if value['payload'].get('crash_once'):os._exit(21)
        time.sleep(20)
    if value['payload'].get('always_crash'):os._exit(21)
    manifest,store=prepare_work(value)
    store.put_job_result(manifest['run_id'],manifest['job_id'],{'job_id':manifest['job_id'],
        'run_id':manifest['run_id'],'status':'succeeded','results':[]})
    return {'status':'succeeded','result_uri':manifest['result_uri']}


def service(tmp_path,role='observation'):
    return WorkerService(role,root=tmp_path/'journal',processor=retry_processor,
                         idle_seconds=.15,poll_seconds=.01)


@pytest.mark.parametrize('crash',[False,True])
def test_hang_or_crash_once_recovers_same_work_manifest_with_no_extra_debit(tmp_path,crash):
    value=work(tmp_path,'original-paid-observation',crash_once=crash)
    value['payload']['manifest']['watchdog_policy']['maximum_retries']=2
    value['payload']['manifest']['original_budget_snapshot']={'calls':1}
    original=deepcopy(value);worker=service(tmp_path)
    try:
        first_pid=worker.process.pid;worker.submit(value)
        result=terminal(worker,value['work_id'])
        assert result['state']=='succeeded' and result['attempts_started']==2 and result['infrastructure_retries']==1
        assert worker.journal.get(value['work_id'])['work']==original
        attempts=[json.loads(line) for line in (tmp_path/(value['work_id']+'-attempts.jsonl')).read_text().splitlines()]
        assert len(attempts)==2 and len({item['pid'] for item in attempts})==2
        assert {item['work_id'] for item in attempts}=={value['work_id']}
        assert not Path('/proc',str(first_pid)).exists()
        assert worker.submit(value)==result  # Joining again grants no new execution.
    finally:worker.close()


def test_recovery_is_bounded_to_two_retries_and_survives_restart_journal(tmp_path):
    value=work(tmp_path,'bounded-observation',crash_once=True,always_crash=True)
    value['payload']['manifest']['watchdog_policy']['maximum_retries']=2
    worker=service(tmp_path)
    try:
        worker.submit(value);result=terminal(worker,value['work_id'])
        assert result['state']=='failed' and result['attempts_started']==3 and result['infrastructure_retries']==2
        assert len((tmp_path/(value['work_id']+'-attempts.jsonl')).read_text().splitlines())==3
    finally:worker.close()
    restored=service(tmp_path)
    try:
        assert restored.submit(value)==result
        assert len((tmp_path/(value['work_id']+'-attempts.jsonl')).read_text().splitlines())==3
    finally:restored.close()


def test_committed_rejection_wins_over_child_death_and_never_retries(tmp_path):
    value=work(tmp_path,'canonical-rejected',reject_then_die=True)
    value['payload']['manifest']['watchdog_policy']['maximum_retries']=2
    worker=service(tmp_path)
    try:
        worker.submit(value);result=terminal(worker,value['work_id'])
        assert result['state']=='failed' and result['attempts_started']==1 and result['infrastructure_retries']==0
        assert result['result']['retained'] is True
        assert len((tmp_path/(value['work_id']+'-attempts.jsonl')).read_text().splitlines())==1
    finally:worker.close()


def test_scoring_child_death_never_repeats_a_potential_judge_or_solve(tmp_path):
    value=work(tmp_path,'private-scoring',role='scoring',crash_once=True)
    value['payload']['manifest']['watchdog_policy']['maximum_retries']=2
    worker=service(tmp_path,role='scoring')
    try:
        worker.submit(value);result=terminal(worker,value['work_id'])
        assert result['state']=='failed' and result['attempts_started']==1 and result['infrastructure_retries']==0
        assert len((tmp_path/(value['work_id']+'-attempts.jsonl')).read_text().splitlines())==1
    finally:worker.close()


def test_retry_requires_positive_previous_process_death_proof(tmp_path,monkeypatch):
    value=work(tmp_path,'unconfirmed-old-process')
    value['payload']['manifest']['watchdog_policy']['maximum_retries']=2
    worker=service(tmp_path)
    monkeypatch.setattr(worker.process,'run',lambda *a,**k:(_ for _ in ()).throw(ChildFailure('transport_ambiguous')))
    try:
        original_pid=worker.process.pid;worker.submit(value);result=terminal(worker,value['work_id'])
        assert result['state']=='failed' and result['attempts_started']==1 and result['infrastructure_retries']==0
        assert worker.process.pid==original_pid
    finally:worker.close()


def test_unfinished_restart_keeps_attempt_budget_and_does_not_recompute(tmp_path):
    value=work(tmp_path,'interrupted-second-attempt')
    value['payload']['manifest']['watchdog_policy']['maximum_retries']=2
    PoolJournal(tmp_path/'journal').put({'work':value,'state':'running',
        'attempts_started':2,'infrastructure_retries':1})
    worker=service(tmp_path)
    try:
        result=worker.submit(value)
        assert result['state']=='failed' and result['attempts_started']==2 and result['infrastructure_retries']==1
        assert not (tmp_path/(value['work_id']+'-attempts.jsonl')).exists()
    finally:worker.close()
