"""Synthetic pinned-video digests, so pool tests run without the hidden media pins."""
from __future__ import annotations

import hashlib

from mini_bang.simulators.flow_lenia_pool.preparation import pool_manifest

MEDIA = {f"traj_{number:05d}": {"sha256": hashlib.sha256(f"video {number}".encode()).hexdigest(),
                                "size_bytes": 1000 + number} for number in range(3)}
TRAJECTORIES = tuple(sorted(MEDIA))


def use_synthetic_media(monkeypatch) -> dict:
    """Point the pool manifest at the synthetic pins for one test."""
    monkeypatch.setattr(pool_manifest, "reviewed_media", lambda: MEDIA)
    return MEDIA
