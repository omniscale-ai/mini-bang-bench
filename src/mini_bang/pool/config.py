"""Startup-only pool settings; runtime work cannot grow or resize Pods."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .scheduler import RemoteSlot
from .protocol import PoolUnavailable


def token_from_env():
    path = os.getenv("MINI_BANG_POOL_TOKEN_FILE")
    token = Path(path).read_text().strip() if path else os.getenv("MINI_BANG_POOL_TOKEN", "")
    if len(token) < 24:
        raise PoolUnavailable("A private persistent-pool token is required")
    return token


def scheduler_slots(token):
    raw = json.loads(os.environ.get("MINI_BANG_POOL_SLOTS", "[]"))
    if not isinstance(raw, list) or not raw:
        raise PoolUnavailable("Declare fixed MINI_BANG_POOL_SLOTS before starting the benchmark")
    slots = []
    for item in raw:
        if item.get("kind", "http") == "http":
            slots.append(RemoteSlot(item["name"], item["role"], item["url"], token))
        else:
            from .offline import OfflineSlot
            slot = OfflineSlot(role=item["role"], pod_name=item["pod_name"],
                namespace=item.get("namespace", "mini-bang"), stager_pod=item.get("stager_pod"),
                config=item.get("config") or {})
            slot.name = item["name"]
            slots.append(slot)
    return slots
