from copy import deepcopy

import pytest

from mini_bang.backends.local import LocalBackend
from mini_bang.framework.runtime import RunManager
from mini_bang.tasks.raf_common.work import RAFWorkLimitExceeded

from tests.synthetic_gold import SMOKE_INSTANCE, use_synthetic_gold

# A closure vs size instance without an instance work budget and without network families.
NO_BUDGET = SMOKE_INSTANCE


def test_cumulative_runtime_budget_rejects_atomic_batch_before_any_new_debit(monkeypatch):
    use_synthetic_gold(monkeypatch)
    monkeypatch.setenv('MINI_BANG_RAF_MAX_WORK_UNITS','32000000')
    monkeypatch.setenv('MINI_BANG_RAF_MAX_SESSION_WORK_UNITS','18')
    class CountBackend(LocalBackend):
        calls=0
        def run_batch(self,*args,**kwargs):
            self.calls+=1
            return super().run_batch(*args,**kwargs)
    backend=CountBackend();session=RunManager(backend=backend).prepare('raf/mechanism-prob',instance_id=NO_BUDGET)
    api=session.environment.api
    api.generate_samples(saturation=6,runs=1,snapshot_times=[1/50000])
    assert session.budget_counters['raf_work_units']==6
    before=deepcopy(session.budget_counters)
    with pytest.raises(RAFWorkLimitExceeded,match='session cumulative'):
        api.generate_samples_batch([{'saturation':6,'runs':1,'snapshot_times':[1/50000]},
                                    {'saturation':8,'runs':1,'snapshot_times':[1/50000]}])
    assert session.budget_counters==before and backend.calls==1
    capability=session.capabilities()['runtime_work_bound']
    assert capability['remaining_session_work_units']==12


def test_signature_budget_comes_only_from_the_instance_and_covers_every_public_case(monkeypatch):
    monkeypatch.setenv('MINI_BANG_RAF_SIGNATURE_MIN_CALLS','1024')    # removed knob: ignored
    session=RunManager(backend=LocalBackend()).prepare('raf/signature')
    cases=sum(len(values) for values in session.instance_spec.visible_data['cases_by_saturation'].values())
    limit=session.instance_spec.metadata['api_limits']['max_generate_calls']
    assert cases==96 and session.policy.max_calls(session)==limit==160>=cases+16


def test_session_work_budget_is_counted_by_default_and_offline_switch_keeps_legacy_counters(monkeypatch):
    use_synthetic_gold(monkeypatch)
    monkeypatch.delenv('MINI_BANG_RAF_MAX_SESSION_WORK_UNITS',raising=False)
    session=RunManager(backend=LocalBackend()).prepare('raf/mechanism-prob',instance_id=NO_BUDGET)
    assert session.budget_counters['raf_work_units']==0
    assert session.capabilities()['runtime_work_bound']['max_reaction_event_work_units_per_session']==160000000
    monkeypatch.setenv('MINI_BANG_RAF_MAX_SESSION_WORK_UNITS','off')
    session=RunManager(backend=LocalBackend()).prepare('raf/mechanism-prob',instance_id=NO_BUDGET)
    assert 'raf_work_units' not in session.budget_counters
