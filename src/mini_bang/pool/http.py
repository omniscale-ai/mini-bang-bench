"""Shared bounded JSON HTTP interface for the scheduler and trusted workers."""
from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, unquote, urlsplit

from .protocol import MAX_ARCHIVE_CHUNK_BYTES, MAX_MESSAGE_BYTES, PoolBackpressure, PoolConflict, PoolUnavailable


def handler(service, token):
    if not isinstance(token, str) or len(token) < 24:
        raise ValueError("Pool control requires an unpredictable token of at least 24 characters")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass  # Do not put private manifests, paths or credentials into logs.

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def _dispatch(self, method):
            if method == "GET" and urlsplit(self.path).path == "/ready":
                health = service.health()
                ready = bool(health.get("ready", health.get("ok", False)))
                return self._respond(200 if ready else 503, {"ready": ready})
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self._respond(401, {"error": "unauthorized"})
                return
            path = urlsplit(self.path).path
            try:
                if method == "GET" and path == "/health":
                    return self._respond(200, service.health())
                if method == "GET" and path.startswith("/v1/work/") and path.endswith("/archive"):
                    work_id = unquote(path[len("/v1/work/"):-len("/archive")])
                    query = parse_qs(urlsplit(self.path).query)
                    return self._respond(200, service.archive_chunk(work_id,
                        offset=int(query.get("offset", ["0"])[0]), limit=int(query.get("limit", [str(MAX_ARCHIVE_CHUNK_BYTES)])[0])))
                if method == "GET" and path.startswith("/v1/work/"):
                    work_id = unquote(path[len("/v1/work/"):])
                    wait = float(parse_qs(urlsplit(self.path).query).get("wait_seconds", ["0"])[0])
                    if not 0 <= wait <= 30:
                        raise ValueError("Invalid long-poll wait")
                    row = (service.wait_status(work_id, wait) if wait and hasattr(service, "wait_status")
                           else service.status(work_id))
                    return self._respond(200 if row else 404, row or {"error": "not_found"})
                if method == "POST" and path == "/v1/work":
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= MAX_MESSAGE_BYTES:
                        raise ValueError("Invalid JSON message size")
                    payload = json.loads(self.rfile.read(size))
                    return self._respond(202, service.submit(payload))
                self._respond(404, {"error": "not_found"})
            except PoolConflict:
                self._respond(409, {"error": "conflicting_work_identity"})
            except PoolBackpressure:
                self._respond(429, {"error": "pool_queue_full"})
            except PoolUnavailable:
                self._respond(503, {"error": "pool_unavailable"})
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                self._respond(400, {"error": "invalid_pool_message"})
            except Exception:
                self._respond(500, {"error": "pool_internal_failure"})

        def _respond(self, code, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(service, token, *, host="0.0.0.0", port=8770):
    server = ThreadingHTTPServer((host, port), handler(service, token))
    server.daemon_threads = True
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()
