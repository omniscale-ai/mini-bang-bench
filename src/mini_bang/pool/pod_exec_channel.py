"""A serialized reusable exec websocket, with bounded transport-only failures."""
from __future__ import annotations

import threading
import time

from .protocol import MAX_MESSAGE_BYTES, PoolUnavailable

CONNECT_SECONDS = 30.
SOCKET_SECONDS = 30.
MAX_STDERR_BYTES = 64 * 1024


def _close(connection):
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


class ExecChannel:
    def __init__(self, connector, cleanup=None):
        self.connector = connector
        self.cleanup = cleanup
        self.lock = threading.Lock()
        self.connection = self.opening = None
        self.closed = False

    def invalidate(self):
        connection, self.connection = self.connection, None
        _close(connection)
        if self.opening is not None:
            self.opening["abandoned"] = True
            if self.opening["done"].is_set():
                _close(self.opening.get("connection"))

    def close(self):
        # Closing the transport must interrupt a blocked call, not wait on it.
        self.closed = True
        self.invalidate()
        if self.cleanup is not None:
            cleanup, self.cleanup = self.cleanup, None
            try:
                cleanup()
            except Exception:
                pass

    def _open(self, timeout):
        if self.closed:
            raise PoolUnavailable("Slot exec transport is closed")
        if self.connection is not None and self.connection.is_open():
            return self.connection
        _close(self.connection)
        self.connection = None
        if self.opening is not None:
            if not self.opening["done"].is_set():
                raise PoolUnavailable("Previous slot exec handshake has not finished")
            self.opening = None
        box = self.opening = {"done": threading.Event(), "abandoned": False}
        def connect():
            try:
                box["connection"] = self.connector()
            except Exception:
                box["failed"] = True  # Never retain API exceptions containing authority data.
            finally:
                box["done"].set()
                if box["abandoned"] or self.closed:
                    _close(box.get("connection"))
        threading.Thread(target=connect, name="fixed-slot-exec-connect", daemon=True).start()
        if not box["done"].wait(timeout):
            box["abandoned"] = True
            if box["done"].is_set():
                _close(box.get("connection"))
            raise PoolUnavailable("Slot exec handshake timed out; no RPC was resent")
        if self.closed or box.get("failed") or box.get("connection") is None:
            _close(box.get("connection"))
            raise PoolUnavailable("Slot exec handshake is unavailable")
        self.connection, self.opening = box["connection"], None
        return self.connection

    def call(self, body, *, prefix, timeout, decode):
        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = self.lock.acquire() if deadline is None else self.lock.acquire(timeout=timeout)
        if not acquired:
            raise PoolUnavailable("Slot exec channel is busy; this RPC was not sent")
        def remaining():
            value = None if deadline is None else deadline - time.monotonic()
            if value is not None and value <= 0:
                raise PoolUnavailable("Slot exec timed out; work outcome is uncertain")
            return value
        def socket_bound(connection):
            left = remaining()
            socket = getattr(connection, "sock", None)
            if callable(getattr(socket, "settimeout", None)):
                socket.settimeout(min(SOCKET_SECONDS, left) if left is not None else SOCKET_SECONDS)
        decoding = False
        try:
            left = remaining()
            connection = self._open(min(CONNECT_SECONDS, left) if left is not None else CONNECT_SECONDS)
            socket_bound(connection)
            connection.write_stdin(body)  # Exactly once, including after an ambiguous error.
            buffer, stderr_bytes = bytearray(), 0
            while True:
                if self.closed:
                    raise PoolUnavailable("Slot exec transport is closed")
                socket_bound(connection)
                left = remaining()
                connection.update(timeout=min(.25, left) if left is not None else .25)
                stdout = connection.read_stdout(timeout=0)
                stderr = connection.read_stderr(timeout=0)
                # Kubernetes WSClient otherwise duplicates every historical frame
                # indefinitely in its capture-all buffer, even after read_stdout.
                drain = getattr(connection, "read_all", None)
                if callable(drain):
                    drain()
                buffer.extend(stdout.encode())
                stderr_bytes += len(stderr.encode())
                if len(buffer) > MAX_MESSAGE_BYTES + len(prefix) + 1 or stderr_bytes > MAX_STDERR_BYTES:
                    raise PoolUnavailable("Slot exec exceeded its bounded response size")
                if "\n" in stdout:
                    line, suffix = bytes(buffer).split(b"\n", 1)
                    if suffix or not line.startswith(prefix.encode()):
                        raise PoolUnavailable("Slot exec returned an invalid response frame")
                    remaining()
                    decoding = True
                    return decode(line[len(prefix):].decode())
                if not connection.is_open():
                    raise PoolUnavailable("Slot exec ended without a result; work outcome is uncertain")
        except BaseException as error:
            self.invalidate()
            if decoding:
                raise
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(error, PoolUnavailable):
                raise
            raise PoolUnavailable("Slot exec transport failed; work outcome is uncertain") from None
        finally:
            self.lock.release()
