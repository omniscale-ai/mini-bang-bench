from contextlib import nullcontext
import socket
import time

from mini_bang.framework.network_boundary import network_boundary_source


def test_guard_waits_for_every_service_and_restarts_confirmation_after_a_reachable_sample(monkeypatch):
    attempts=[]
    sleeps=[]
    results=iter([False,True, False,False, True,False, False,False, False,False])
    def connect(address,**kwargs):
        attempts.append(address)
        if next(results):return nullcontext()
        raise OSError('Policy denies egress')
    monkeypatch.setattr(socket,'create_connection',connect)
    monkeypatch.setattr(time,'sleep',sleeps.append)
    code=network_boundary_source([('10.0.0.1',9000),('10.0.0.2',443)],confirmations=2)
    exec(compile(code,'trusted-network-startup-check','exec'),{})
    assert len(attempts)==10 and sleeps==[5,5,5,5]
