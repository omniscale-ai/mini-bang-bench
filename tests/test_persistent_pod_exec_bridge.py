"""Warm exec reuse, bounded ambiguity, and a real multi-RPC Unix socket bridge."""
from collections import deque
import json
import os
from pathlib import Path
import selectors
import socket
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from mini_bang.pool.pod_exec import BOOTSTRAP, PREFIX, PodExecTransport
from mini_bang.pool.protocol import MAX_MESSAGE_BYTES, PoolConflict, PoolUnavailable


class Connection:
    def __init__(self, mode='normal', gate=None):
        self.mode,self.gate=mode,gate
        self.open=True;self.pending=None;self.writes=[];self.stdout='';self.stderr=''
        self.captured='';self.drains=0;self.timeout_values=[]
        self.sock=SimpleNamespace(settimeout=self.timeout_values.append)

    def is_open(self):return self.open
    def close(self):self.open=False
    def write_stdin(self,body):
        assert self.pending is None, 'Interleaved RPCs on one websocket'
        self.writes.append(json.loads(body));self.pending=self.writes[-1]
        if self.mode=='write-error':raise OSError('PRIVATE credentials from SDK')
    def update(self,timeout=0):
        if self.gate and not self.gate.is_set():
            time.sleep(min(timeout,.01));return
        if self.pending is None:return
        if self.mode=='idle':time.sleep(min(timeout,.01));return
        if self.mode=='eof':self.open=False;return
        if self.mode=='oversized':self.stdout='x'*(MAX_MESSAGE_BYTES+100)
        elif self.mode=='stderr':self.stderr='PRIVATE'*10000
        elif self.mode=='malformed':self.stdout=PREFIX+'PRIVATE-not-json\n'
        elif self.mode=='extra':self.stdout=PREFIX+'{"ok":true,"value":1}\nEXTRA'
        elif self.mode=='conflict':self.stdout=PREFIX+'{"ok":false,"error":{"type":"WorkConflict"}}\n'
        elif self.mode=='private-error':self.stdout=PREFIX+'{"ok":false,"error":{"type":"PRIVATE-auth"}}\n'
        elif self.mode=='bad-error-shape':self.stdout=PREFIX+'{"ok":false,"error":"PRIVATE-auth"}\n'
        else:self.stdout=PREFIX+json.dumps({'ok':True,'value':{'action':self.pending['action']}})+'\n'
        self.captured+=self.stdout+self.stderr;self.pending=None
    def read_stdout(self,timeout=0):
        assert timeout==0
        value,self.stdout=self.stdout,'';return value
    def read_stderr(self,timeout=0):
        assert timeout==0
        value,self.stderr=self.stderr,'';return value
    def read_all(self):
        self.drains+=1;value,self.captured=self.captured,'';return value


@pytest.fixture
def sdk(monkeypatch):
    calls=[];connections=[];modes=deque();opening_gate={};api_clients=[]
    class ApiClient:
        def __init__(self,configuration=None):
            self.configuration=configuration;self.default_headers={};self.closed=False;api_clients.append(self)
        def close(self):self.closed=True
    class Core:
        def __init__(self,api_client):self.api_client=api_client
        def connect_get_namespaced_pod_exec(self,**kwargs):raise AssertionError('stream wrapper must be used')
        def read_namespaced_pod(self,**kwargs):return {'metadata':{'uid':'stable'},'spec':{}}
    original=ApiClient(configuration=object());original.default_headers={'Authorization':'PRIVATE-authority'}
    core=Core(original)
    def stream(method,**options):
        assert method.__self__.api_client is not original
        calls.append((method.__self__.api_client,options))
        gate=opening_gate.get(options['name'])
        if gate:gate.wait(2)
        connection=Connection(modes.popleft() if modes else 'normal')
        connections.append(connection);return connection
    package=ModuleType('kubernetes');package.__path__=[]
    client=ModuleType('kubernetes.client');client.ApiClient=ApiClient;client.CoreV1Api=Core
    streaming=ModuleType('kubernetes.stream');streaming.stream=stream
    monkeypatch.setitem(sys.modules,'kubernetes',package)
    monkeypatch.setitem(sys.modules,'kubernetes.client',client)
    monkeypatch.setitem(sys.modules,'kubernetes.stream',streaming)
    transport=PodExecTransport(SimpleNamespace(_core=core))
    yield SimpleNamespace(transport=transport,calls=calls,connections=connections,modes=modes,
                         opening_gate=opening_gate,api_clients=api_clients,original=original)
    for gate in opening_gate.values():gate.set()
    transport.close()


def rpc(sdk,action='health',pod='fixed',timeout=1):
    return sdk.transport.json('benchmark',pod,{'action':action,'PRIVATE':'request-secret'},container='agent',timeout=timeout)


def test_one_bridge_per_fixed_container_and_capture_memory_does_not_grow(sdk):
    for _ in range(100):assert rpc(sdk)=={'action':'health'}
    assert len(sdk.calls)==1 and len(sdk.connections[0].writes)==100
    assert sdk.connections[0].captured=='' and sdk.connections[0].drains==100
    assert sdk.calls[0][1]['command']==['python','-I','-S','-c',BOOTSTRAP]
    assert 'request-secret' not in str(sdk.calls[0][1]['command'])
    assert sdk.connections[0].open
    rpc(sdk,pod='stager')
    assert len(sdk.calls)==2 and sdk.calls[0][0] is not sdk.calls[1][0]
    assert sdk.calls[0][0].configuration is sdk.original.configuration
    sdk.transport.close()
    assert all(not connection.open for connection in sdk.connections)
    assert all(client.closed for client in sdk.api_clients[1:]) and not sdk.original.closed
    with pytest.raises(PoolUnavailable):rpc(sdk)


@pytest.mark.parametrize('mode',['eof','idle','oversized','stderr','malformed','extra','write-error','private-error','bad-error-shape'])
def test_uncertain_request_is_sent_once_and_only_later_inspect_reconnects(sdk,mode):
    sdk.modes.append(mode)
    with pytest.raises(PoolUnavailable) as failure:rpc(sdk,'start',timeout=.03)
    assert 'PRIVATE' not in str(failure.value) and 'request-secret' not in str(failure.value)
    assert len(sdk.calls)==1 and not sdk.connections[0].open
    assert rpc(sdk,'inspect')=={'action':'inspect'}
    assert len(sdk.calls)==2
    assert [request['action'] for connection in sdk.connections for request in connection.writes]==['start','inspect']


def test_confirmed_conflict_remains_typed_and_channel_is_closed(sdk):
    sdk.modes.append('conflict')
    with pytest.raises(PoolConflict):rpc(sdk,'start')
    assert not sdk.connections[0].open


def test_calls_on_one_channel_are_serialized_and_close_interrupts_wait(sdk):
    rpc(sdk);connection=sdk.connections[0];gate=connection.gate=threading.Event();results=[]
    first=threading.Thread(target=lambda:results.append(rpc(sdk,'inspect')),daemon=True)
    second=threading.Thread(target=lambda:results.append(rpc(sdk,'health')),daemon=True)
    first.start()
    until=time.monotonic()+1
    while len(connection.writes)<2 and time.monotonic()<until:time.sleep(.005)
    second.start();time.sleep(.025)
    assert len(connection.writes)==2
    gate.set();first.join(1);second.join(1)
    assert results==[{'action':'inspect'},{'action':'health'}] and len(sdk.calls)==1
    errors=[];connection.mode='idle'
    def waiting():
        try:rpc(sdk,'inspect',timeout=None)
        except PoolUnavailable as error:errors.append(str(error))
    thread=threading.Thread(target=waiting,daemon=True);thread.start();time.sleep(.02)
    before=time.monotonic();sdk.transport.close();thread.join(1)
    assert time.monotonic()-before<.5 and errors and not connection.open


def test_stuck_handshake_is_bounded_and_does_not_block_other_pod_or_get(sdk):
    gate=threading.Event();sdk.opening_gate['stuck']=gate
    with pytest.raises(PoolUnavailable):rpc(sdk,pod='stuck',timeout=.03)
    with pytest.raises(PoolUnavailable):rpc(sdk,pod='stuck',timeout=.03)
    assert len(sdk.calls)==1  # No unbounded replacement opener threads.
    before=time.monotonic()
    assert sdk.transport.read_pod('benchmark','healthy')['metadata']['uid']=='stable'
    assert rpc(sdk,pod='healthy')=={'action':'health'}
    assert time.monotonic()-before<.2
    gate.set()
    until=time.monotonic()+1
    while len(sdk.connections)<2 and time.monotonic()<until:time.sleep(.005)
    assert not sdk.connections[-1].open  # Late abandoned opener is closed.
    assert rpc(sdk,'inspect',pod='stuck')=={'action':'inspect'}


def test_real_bridge_reuses_same_process_for_multiple_unix_socket_requests():
    def receive(stream,size):
        result=b''
        while len(result)<size:
            part=stream.recv(size-len(result));assert part;result+=part
        return result
    peers=[]
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            pid,_,_=struct.unpack('3i',self.request.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12));peers.append(pid)
            size=struct.unpack('!I',receive(self.request,4))[0]
            request=json.loads(receive(self.request,size))
            body=json.dumps({'ok':True,'value':{'peer_pid':pid,'action':request['action']}}).encode()
            self.request.sendall(struct.pack('!I',len(body))+body)
    with tempfile.TemporaryDirectory(prefix='mini-bang-warm-exec-',dir='/tmp') as directory:
        address=str(Path(directory)/'control.sock')
        with socketserver.UnixStreamServer(address,Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            child=subprocess.Popen([sys.executable,'-I','-S','-c',BOOTSTRAP],stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout,selectors.EVENT_READ)
                    for action in ('health','cleanup','inspect'):
                        child.stdin.write(json.dumps({'socket_path':address,'action':action})+'\n');child.stdin.flush()
                        assert selector.select(2), 'Bridge did not return bounded RPC response'
                        line=child.stdout.readline();assert line.startswith(PREFIX)
                        assert json.loads(line[len(PREFIX):])['value']=={'peer_pid':child.pid,'action':action}
                        assert child.poll() is None
                child.stdin.close();assert child.wait(timeout=2)==0
                assert peers==[child.pid]*3
            finally:
                if child.poll() is None:child.kill();child.wait(timeout=2)
                child.stdout.close();child.stderr.close();server.shutdown();thread.join(1)
