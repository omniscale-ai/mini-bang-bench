"""Durable history, honest cached readiness and fixed slot admission."""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time

import pytest

from mini_bang.pool.scheduler import PoolScheduler
from mini_bang.pool.journal import PoolJournal
from mini_bang.pool.protocol import PoolBackpressure, PoolConflict, PoolUnavailable, VERSION


QUEUED_AT = 1  # Journal rows only need an ordering timestamp here.


def work(name, role="observation", *, padding="", cpu=1, memory="1Gi"):
    return {"version":VERSION,"work_id":name,"role":role,"payload":{"PRIVATE":padding,
        "manifest":{"resource_spec":{"cpu":cpu,"memory":memory,"gpu":0}}}}


def wait(predicate, timeout=3):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if predicate():return
        time.sleep(.005)
    raise AssertionError("Condition did not converge")


class Slot:
    def __init__(self,name="fixed-observation",role="observation", *, gate=None, limits=None):
        self.name,self.role,self.gate=name,role,gate
        self.limits=limits or {};self.ready=True;self.calls=[];self.health_calls=0
        self.started=threading.Event();self.release=threading.Event();self.release.set()
        self.config={};self.pod_uid="original-pod";self.stager_uid="original-stager"

    def health(self):
        self.health_calls+=1
        if self.gate:self.gate.wait(10)
        result={"ready":self.ready,"role":self.role,"resource_limits":self.limits}
        if self.role in {"agent","executable"}:
            result.update(pod_name=self.name,pod_uid=self.pod_uid,stager_pod_uid=self.stager_uid)
        return result

    def execute(self,value):
        self.calls.append(value["work_id"]);self.started.set();self.release.wait(3)
        return {"work_id":value["work_id"],"state":"succeeded","result":{"worker_pid":17}}


def scheduler(root,slots,**options):
    return PoolScheduler(root,slots,health_interval=.01,health_stale_seconds=.2,**options)


def test_journal_streams_history_and_terminal_cache_is_bounded_and_lazy(tmp_path):
    journal=PoolJournal(tmp_path)
    for number in range(70):
        item=work(f"historic-observation-{number}",padding="x"*2000)
        journal.put({"work":item,"state":"succeeded","created_at":number,"slot":"fixed-observation"})
    assert iter(journal.records()) is not None and not isinstance(journal.records(),list)
    slot=Slot();service=scheduler(tmp_path,[slot],terminal_cache_entries=3,terminal_cache_bytes=7000)
    try:
        assert len(service.records)==len(service.terminal)==0
        for number in range(70):assert service.status(f"historic-observation-{number}")["state"]=="succeeded"
        assert len(service.terminal)<=3 and service.terminal_bytes<=7000
        assert service.submit(work("historic-observation-0",padding="x"*2000))["state"]=="succeeded"
        with pytest.raises(PoolConflict):service.submit(work("historic-observation-0",padding="changed"))
        assert not slot.calls
    finally:service.close()


def test_readiness_waits_for_every_fixed_slot_and_never_blocks_http_probe(tmp_path):
    gate=threading.Event();slow=Slot("slow",gate=gate);healthy=Slot("healthy")
    service=scheduler(tmp_path,[healthy,slow])
    try:
        wait(lambda:slow.health_calls>0)
        before=time.monotonic()
        for _ in range(100):assert service.health()["ready"] is False
        assert time.monotonic()-before<.2 and slow.health_calls==1
        gate.set();wait(lambda:service.health()["ready"])
        slow.ready=False;wait(lambda:not service.health()["ready"])
    finally:gate.set();service.close()


def test_background_health_skips_an_active_offline_lease(tmp_path):
    slot=Slot(role="agent");slot.release.clear();service=scheduler(tmp_path,[slot])
    try:
        wait(lambda:service.health()["ready"])
        service.submit(work("agent-run",role="agent"));assert slot.started.wait(1)
        calls=slot.health_calls
        for _ in range(30):service.health();time.sleep(.005)
        assert slot.health_calls==calls
        slot.release.set();wait(lambda:service.status("agent-run")["state"]=="succeeded")
    finally:slot.release.set();service.close()


def test_close_does_not_wait_for_blocked_health_or_publish_late_binding(tmp_path):
    gate=threading.Event();slot=Slot(role="agent",gate=gate)
    first=scheduler(tmp_path,[slot]);wait(lambda:slot.health_calls>0)
    before=time.monotonic();first.close()
    assert time.monotonic()-before<.5
    replacement=Slot(role="agent");replacement.pod_uid="replacement"
    second=scheduler(tmp_path,[replacement])
    try:
        wait(lambda:second.health()["ready"]);gate.set();time.sleep(.03)
        assert second.journal.slot_binding(slot.name)["pod_uid"]=="replacement"
    finally:gate.set();second.close()


def test_offline_uid_binding_survives_restart_and_refuses_replacement(tmp_path):
    first=scheduler(tmp_path,[Slot(role="agent")]);wait(lambda:first.health()["ready"]);first.close()
    changed=Slot(role="agent");changed.stager_uid="different-stager"
    second=scheduler(tmp_path,[changed])
    try:
        wait(lambda:second.slot_health[changed.name]["checked_at"] is not None)
        assert second.health()["ready"] is False
        assert changed.config["expected_pod_uid"]=="original-pod"
        assert changed.config["expected_stager_pod_uid"]=="original-stager"
        second.submit(work("do-not-rerun-model",role="agent"));time.sleep(.03)
        assert not changed.calls and second.status("do-not-rerun-model")["state"]=="pending"
    finally:second.close()


def test_pending_byte_backpressure_releases_after_completion(tmp_path):
    slot=Slot();slot.release.clear()
    service=scheduler(tmp_path,[slot],max_pending=20,max_pending_bytes=6000)
    try:
        wait(lambda:service.health()["ready"])
        one=work("first-large",padding="x"*3000);two=work("second-large",padding="x"*3000)
        service.submit(one);assert slot.started.wait(1)
        with pytest.raises(PoolBackpressure):service.submit(two)
        assert service.status(two["work_id"]) is None and service.pending_bytes<=6000
        # Accounting already reserves the started/assigned/status envelope.
        assert len(json.dumps(service.records[one['work_id']]).encode())<=service.pending_bytes
        slot.release.set();wait(lambda:service.status(one["work_id"])["state"]=="succeeded")
        service.submit(two);wait(lambda:service.status(two["work_id"])["state"]=="succeeded")
        assert service.pending_bytes==0 and slot.calls==[one["work_id"],two["work_id"]]
    finally:slot.release.set();service.close()


def test_lost_offline_response_rejoins_original_slot_without_a_second_solve(tmp_path):
    class RecoverableSlot(Slot):
        def __init__(self):
            super().__init__(role='agent');self.solves=0;self.inspected=False
        def health(self):
            value=super().health()
            if self.calls:self.inspected=True  # Offline health inspects the original child.
            return value
        def execute(self,value):
            self.calls.append(value['work_id'])
            if len(self.calls)==1:
                self.solves+=1
                raise PoolUnavailable('Response lost after starting the original child')
            assert self.inspected
            return {'work_id':value['work_id'],'state':'succeeded','result':{'retained':True}}
    slot=RecoverableSlot();service=scheduler(tmp_path,[slot],retry_seconds=.01)
    try:
        service.submit(work('retained-agent',role='agent'))
        wait(lambda:service.status('retained-agent')['state']=='succeeded')
        assert slot.calls==['retained-agent','retained-agent'] and slot.solves==1
        assert service.status('retained-agent')['slot']==slot.name
    finally:service.close()


def test_restart_recovery_only_slot_rejoins_assigned_work_and_never_starts_new_work(tmp_path):
    class QuarantinedSlot(Slot):
        def health(self):
            return {**super().health(),'recovery_only_work_id':'original-agent'}
    slot=QuarantinedSlot(role='agent');slot.release.clear()
    PoolJournal(tmp_path).put({'work':work('original-agent',role='agent'),'created_at':QUEUED_AT,
        'state':'recovery_required','slot':slot.name})
    service=scheduler(tmp_path,[slot])
    try:
        assert slot.started.wait(1)
        assert service.health()['ready'] is False
        service.submit(work('new-agent',role='agent'));slot.release.set()
        wait(lambda:service.status('original-agent')['state']=='succeeded')
        time.sleep(.04)
        assert slot.calls==['original-agent'] and service.status('new-agent')['state']=='pending'
        assert service.health()['slots'][0]['recovery_only_work_id']=='original-agent'
    finally:slot.release.set();service.close()


def test_fixed_roles_and_capacity_route_only_to_compatible_slot(tmp_path):
    small=Slot("small",limits={"cpu":1.,"memory":1024**3,"gpu":0.})
    large=Slot("large",limits={"cpu":2.,"memory":8*1024**3,"gpu":0.})
    scoring=Slot("scoring",role="scoring",limits={"cpu":1.,"memory":1024**3,"gpu":0.})
    service=scheduler(tmp_path,[small,large,scoring])
    try:
        wait(lambda:service.health()["ready"])
        item=work("large-observation",cpu=2,memory="8Gi")
        service.submit(item);wait(lambda:service.status(item["work_id"])["state"]=="succeeded")
        assert not small.calls and not scoring.calls and large.calls==[item["work_id"]]
        bad=service.submit(work("too-large",cpu=3,memory="8Gi"))
        assert bad["state"]=="failed" and bad["error"]["category"]=="infrastructure_failed"
        unsupported=service.submit(work("missing-role",role="agent"))
        assert unsupported["state"]=="failed" and not small.calls
    finally:service.close()


def test_unknown_slot_exception_is_infrastructure_and_does_not_expose_work_payload(tmp_path):
    slot=Slot()
    def failed(value):raise RuntimeError("PRIVATE context accidentally in worker exception")
    slot.execute=failed;service=scheduler(tmp_path,[slot])
    try:
        service.submit(work("unknown-failure",padding="PRIVATE"))
        wait(lambda:service.status("unknown-failure")["state"]=="failed")
        value=service.status("unknown-failure")
        assert value["error"]["category"]=="infrastructure_failed" and "PRIVATE" not in json.dumps(value)
    finally:service.close()


def test_journal_tightens_existing_auth_files_and_rejects_symlinks(tmp_path):
    root=tmp_path/"private";root.mkdir(mode=0o777);root.chmod(0o777)
    item=work("agent-session",role="agent",padding="sensitive-model-env")
    path=root/"agent-session.json";path.write_text(json.dumps({"work":item,"state":"pending"}));path.chmod(0o644)
    journal=PoolJournal(root)
    assert root.stat().st_mode&0o777==0o700 and path.stat().st_mode&0o777==0o600
    assert journal.get("agent-session")["work"]==item
    outside=tmp_path/"outside.json";outside.write_text('{}');outside.chmod(0o644)
    (root/"symlink.json").symlink_to(outside)
    with pytest.raises(OSError):journal.get("symlink")
    assert outside.stat().st_mode&0o777==0o644
    with pytest.raises(ValueError):PoolJournal(root)
