"""Kubernetes exec stdin/stdout to a prestarted root-only Unix socket.

Argv contains only a fixed stdlib bridge script, never work data or credentials.
"""
from __future__ import annotations

import json
import math
import threading

from .pod_exec_channel import CONNECT_SECONDS, ExecChannel
from .protocol import MAX_MESSAGE_BYTES, PoolConflict, PoolUnavailable

PREFIX = "__MINI_BANG_SLOT_RESULT__"


class SlotOperationFailed(RuntimeError):
    """A confirmed operation failed, distinct from an uncertain transport."""


BOOTSTRAP = r'''
import json,socket,struct,sys
def receive(stream,size):
    result=b''
    while len(result)<size:
        part=stream.recv(size-len(result))
        if not part: raise OSError('Incomplete slot response')
        result+=part
    return result
while True:
    line=sys.stdin.buffer.readline(8*1024*1024+1)
    if not line: break
    if len(line)>8*1024*1024 or not line.endswith(b'\n'):
        raise ValueError('Oversized or incomplete slot request')
    request=json.loads(line)
    address=request.pop('socket_path','/control/offline.sock')
    body=json.dumps(request,allow_nan=False).encode()
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
        stream.connect(address)
        stream.sendall(struct.pack('!I',len(body))+body)
        size=struct.unpack('!I',receive(stream,4))[0]
        if size>8*1024*1024: raise ValueError('Oversized slot response')
        result=json.loads(receive(stream,size))
    print('__MINI_BANG_SLOT_RESULT__'+json.dumps(result,allow_nan=False),flush=True)
'''


class PodExecTransport:
    def __init__(self, client=None):
        self.client = client
        self._api_lock = threading.RLock()
        self._channel_lock = threading.RLock()
        self._channels, self._closed = {}, False

    def _client(self):
        with self._api_lock:
            if self.client is None:
                from mini_bang.backends.k8s import KubernetesJobClient
                self.client = KubernetesJobClient()
        return self.client

    def read_pod(self, namespace, pod_name):
        client = self._client()
        if callable(getattr(client, 'read_pod', None)):
            result = client.read_pod(namespace, pod_name)
        else:
            if not self._api_lock.acquire(timeout=CONNECT_SECONDS):
                raise PoolUnavailable('Slot API connection is unavailable')
            try:
                result = client._core.read_namespaced_pod(name=pod_name, namespace=namespace,
                                                        _request_timeout=CONNECT_SECONDS)
            finally:
                self._api_lock.release()
        return result.to_dict() if hasattr(result, 'to_dict') else result

    def delete_pod(self, namespace, pod_name, *, uid, grace_seconds=5):
        """Destroy one used one-shot Pod; the UID precondition never hits its replacement."""
        self.forget(namespace, pod_name)
        client = self._client()
        if callable(getattr(client, 'delete_pod', None)):
            return client.delete_pod(namespace, pod_name, uid=uid, grace_seconds=grace_seconds)
        from kubernetes.client import V1DeleteOptions, V1Preconditions
        if not self._api_lock.acquire(timeout=CONNECT_SECONDS):
            raise PoolUnavailable('Slot API connection is unavailable')
        try:
            client._core.delete_namespaced_pod(name=pod_name, namespace=namespace, _request_timeout=CONNECT_SECONDS,
                body=V1DeleteOptions(grace_period_seconds=grace_seconds, preconditions=V1Preconditions(uid=uid)))
        except Exception as error:
            if getattr(error, 'status', None) not in (404, 409):  # Already gone or already replaced.
                raise
        finally:
            self._api_lock.release()

    def forget(self, namespace, pod_name):
        with self._channel_lock:
            stale = [key for key in self._channels if key[:2] == (namespace, pod_name)]
            channels = [self._channels.pop(key) for key in stale]
        for channel in channels:
            channel.close()

    def json(self, namespace, pod_name, request, *, container=None, timeout=None):
        body = json.dumps(request, allow_nan=False) + '\n'
        if len(body.encode()) > MAX_MESSAGE_BYTES:
            raise ValueError('Slot request exceeds the control-message bound')
        if timeout is not None and (isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Slot timeout must be finite and positive, or explicitly None')
        if self._closed:
            raise PoolUnavailable('Slot exec transport is closed')
        command = ['python', '-I', '-S', '-c', BOOTSTRAP]
        client = self._client()
        if callable(getattr(client, 'exec_pod', None)):
            result = client.exec_pod(namespace, pod_name, command, stdin=body,
                                     container=container, timeout=timeout)
            stdout = result.get('stdout', '')
        else:
            key = namespace, pod_name, container
            with self._channel_lock:
                if self._closed:
                    raise PoolUnavailable('Slot exec transport is closed')
                channel = self._channels.get(key)
                if channel is None:
                    from kubernetes.client import ApiClient, CoreV1Api
                    # stream mutates ApiClient.request. Each channel has its own
                    # client so a stuck handshake cannot poison GET or another Pod.
                    original = client._core.api_client
                    private = ApiClient(configuration=original.configuration)
                    private.default_headers.update(original.default_headers)
                    core = CoreV1Api(api_client=private)
                    def connect():
                        from kubernetes.stream import stream
                        return stream(core.connect_get_namespaced_pod_exec,
                            name=pod_name, namespace=namespace, container=container, command=command,
                            stderr=True, stdin=True, stdout=True, tty=False, _preload_content=False,
                            _request_timeout=CONNECT_SECONDS)
                    channel = self._channels[key] = ExecChannel(connect, cleanup=private.close)
            return channel.call(body, prefix=PREFIX, timeout=timeout, decode=self._decode)
        if len(stdout.encode()) > MAX_MESSAGE_BYTES:
            raise PoolUnavailable('Oversized slot response')
        for line in reversed(stdout.splitlines()):
            if line.startswith(PREFIX):
                return self._decode(line[len(PREFIX):])
        raise PoolUnavailable('Slot exec ended without a result; work outcome is uncertain')

    @staticmethod
    def _decode(body):
        try:
            result = json.loads(body)
            if not isinstance(result, dict) or type(result.get('ok')) is not bool:
                raise ValueError('Invalid supervisor response')
            if not result['ok']:
                kind = result.get('error', {}).get('type', 'Error')
                if kind == 'WorkConflict':
                    raise PoolConflict('Work identity conflicts with retained immutable inputs')
                if kind == 'PreparationFailed':
                    raise SlotOperationFailed('Trusted input preparation failed')
                raise PoolUnavailable('Slot supervisor did not confirm this operation')
            return result['value']
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            if isinstance(error, PoolConflict):
                raise
            raise PoolUnavailable('Slot supervisor returned an invalid response') from None

    def close(self):
        with self._channel_lock:
            self._closed = True
            channels = list(self._channels.values())
        for channel in channels:
            channel.close()
