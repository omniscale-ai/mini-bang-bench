"""Authenticated control client shared by all trusted K8s callers."""
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .protocol import MAX_ARCHIVE_CHUNK_BYTES, MAX_MESSAGE_BYTES, PoolBackpressure, PoolConflict, PoolUnavailable, TERMINAL


class PoolClient:
    def __init__(self, url: str, token: str | None = None, *, timeout=10, poll_seconds=.1):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("Pool URL must be an explicit HTTP(S) endpoint without credentials")
        self.url = url.rstrip("/")
        self.token = token if token is not None else os.environ.get("MINI_BANG_POOL_TOKEN")
        if not self.token:
            raise PoolUnavailable("MINI_BANG_POOL_TOKEN is required for trusted pool access")
        self.timeout, self.poll_seconds = timeout, poll_seconds
        # Cluster/internal traffic must never inherit an external HTTP proxy.
        self.opener = build_opener(ProxyHandler({}))

    def request(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        if body is not None and len(body) > MAX_MESSAGE_BYTES:
            raise ValueError("Pool message exceeds the control-plane size limit")
        request = Request(self.url + path, data=body, method=method, headers={
            "Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = response.read(MAX_MESSAGE_BYTES + 1)
        except HTTPError as error:
            if error.code == 429:
                raise PoolBackpressure("Persistent pool admission queue is full") from error
            if error.code == 409:
                raise PoolConflict("Pool work identity conflicts with existing inputs") from error
            if error.code == 400:
                raise ValueError("Pool work has invalid inputs or incompatible fixed resources") from error
            raise PoolUnavailable(f"Pool returned HTTP {error.code}") from error
        except (URLError, OSError, TimeoutError) as error:
            raise PoolUnavailable("Persistent pool transport is unavailable") from error
        if len(result) > MAX_MESSAGE_BYTES:
            raise PoolUnavailable("Oversized pool response")
        return json.loads(result)

    def health(self):
        return self.request("GET", "/health")

    def submit(self, work):
        return self.request("POST", "/v1/work", work)

    def status(self, work_id, *, wait_seconds=0):
        return self.request("GET", "/v1/work/" + quote(work_id, safe="")
                            + (f"?wait_seconds={wait_seconds}" if wait_seconds else ""))

    def archive_chunk(self, work_id, *, offset=0, limit=MAX_ARCHIVE_CHUNK_BYTES):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= MAX_ARCHIVE_CHUNK_BYTES:
            raise ValueError("Invalid bounded archive range")
        return self.request("GET", "/v1/work/" + quote(work_id, safe="")
                            + f"/archive?offset={offset}&limit={limit}")

    def wait(self, work_id, *, on_poll=None):
        while True:
            row = self.status(work_id, wait_seconds=min(2., self.timeout / 2))
            if on_poll:
                on_poll(row)
            if row["state"] in TERMINAL:
                return row

    def run(self, work, *, on_poll=None):
        # A lost HTTP response never grants a new identity or a second solve.
        try:
            self.submit(work)
        except PoolUnavailable:
            try:
                self.status(work["work_id"])
            except PoolUnavailable:
                self.submit(work)
        return self.wait(work["work_id"], on_poll=on_poll)
