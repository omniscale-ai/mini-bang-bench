"""The standalone harness reports actual work without changing feature results."""
import json
import time

import pytest

from mini_bang.tasks.flow_lenia_common import representation_worker as worker
from tests.test_representation_worker_performance import chunks, legacy_features


def events(capture):
    return [json.loads(line[len(worker.EVENT_PREFIX):]) for line in capture.splitlines()
            if line.startswith(worker.EVENT_PREFIX)]


def test_actual_pool_admission_exits_and_order_remain_visible(tmp_path, capfd):
    config = {'files': chunks(tmp_path, count=3), 'code': '''
import time
def featurize(window):
    values=list(window)
    # Different completion times must not reorder the scientific rows.
    time.sleep(max(0, .03-float(values[0]['A'][0,0,0])*.01))
    return [float(values[0]['A'].sum()),len(values)]
'''}
    expected = legacy_features(config, tmp_path)
    result = worker.extract(config, workers=2, checkpoint_root=tmp_path)
    assert result == expected
    recorded = events(capfd.readouterr().err)
    admission = next(item for item in recorded if item['event']=='admission')
    assert admission['allocated_workers'] == worker.allocated_workers(2)
    assert admission['windows'] == 3 and admission['thinning_factors']==[1,2,4]
    completed = [item for item in recorded if item['event']=='window_factor_complete']
    assert {(e['window_index'],e['thinning']) for e in completed} == {
        (index,factor) for index in range(3) for factor in (1,2,4)}
    assert all(e['loader_seconds']>=0 and e['submitted_seconds']>=.009 for e in completed)
    if admission['allocated_workers']==2:
        exits = next(item for item in recorded if item['event']=='pool_complete')['worker_exits']
        assert len(exits)==2 and all(item['exit_code']==0 for item in exits)


def test_loader_and_submitted_wall_time_are_separate(tmp_path, capfd):
    config = {'files': chunks(tmp_path,count=1), 'code': '''
import time
def featurize(window):
    result=sum(float(state['A'].sum()) for state in window)
    time.sleep(.025)
    return [result]
'''}
    before = time.monotonic()
    worker.extract(config,workers=1,checkpoint_root=tmp_path)
    elapsed = time.monotonic()-before
    complete = [item for item in events(capfd.readouterr().err) if item['event']=='window_factor_complete']
    assert len(complete)==3
    assert all(item['submitted_seconds']>=.025 for item in complete)
    assert all(item['loader_seconds']>0 and item['loader_cpu_seconds']>=0 for item in complete)
    assert sum(item['submitted_seconds']+item['loader_seconds'] for item in complete)<=elapsed


@pytest.mark.parametrize('loader_failure',[False,True])
def test_failure_evidence_distinguishes_loader_from_submitted_code(tmp_path,capfd,loader_failure):
    names = ['missing.npz'] if loader_failure else chunks(tmp_path,count=1)[0]
    code = 'def featurize(window):\n    return [sum(float(x["A"].sum()) for x in window)]\n' if loader_failure else (
        'def featurize(window):\n    raise ValueError("submitted failure")\n')
    with pytest.raises(worker.TrustedLoaderError if loader_failure else ValueError):
        worker.one_window((0,names,code,'featurize',str(tmp_path)))
    failure = next(item for item in events(capfd.readouterr().err) if item['event']=='failure')
    assert failure['attribution']==('infrastructure' if loader_failure else 'participant')
    assert failure['phase']==('trusted_loader' if loader_failure else 'submitted_code')
    assert failure['error_type']==('FileNotFoundError' if loader_failure else 'ValueError')
