"""The recorded pool manifest: file references, safe paths and format checks."""
from __future__ import annotations

import functools
import hashlib
from importlib.resources import files
import json
from pathlib import Path, PurePosixPath

from mini_bang.simulators.flow_lenia_pool.observations import POOL_SCHEMA as SCHEMA

MEDIA_DIGESTS = ("mini_bang.tasks.flow_lenia_common", "gold/media_digests.json")


@functools.lru_cache(maxsize=1)
def reviewed_media() -> dict:
    """Pinned sha256 and size of every annotated video, read from hidden gold on first use."""
    package, name = MEDIA_DIGESTS
    return json.loads(files(package).joinpath(name).read_text())


def reviewed_trajectories() -> tuple[str, ...]:
    """Reviewed trajectories are exactly those whose annotated videos are pinned."""
    return tuple(sorted(reviewed_media()))


def file_ref(path: Path, *, uri=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"uri": uri or path.resolve().as_uri(), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or ":" in name or "\x00" in name or not path.parts):
        raise ValueError("Archive paths must be safe relative POSIX paths")
    return path


def walk_refs(value):
    if isinstance(value, dict):
        if "uri" in value:
            yield value
        else:
            for item in value.values():
                yield from walk_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_refs(item)


def validate_manifest(manifest, *, mode=None):
    trajectories = reviewed_trajectories()
    if manifest.get("schema") != SCHEMA or set(manifest.get("trajectories", {})) != set(trajectories):
        raise ValueError(f"Pool must contain exactly the {len(trajectories)} reviewed trajectories")
    for name, trajectory in manifest["trajectories"].items():
        if trajectory.get("last_step") != 500000:
            raise ValueError(f"{name}: expected original 500000-step recording")
        if mode in {"video", "checkpoints"}:
            video = trajectory.get("video", {})
            if (video.get("first_step") != 0 or video.get("step_stride") != 100
                    or video.get("frame_count") != 5001 or video.get("width") != 224
                    or video.get("height") != 224 or video.get("fps") != 30 or not video.get("ref")):
                raise ValueError(f"{name}: missing original 5001-frame video")
        if mode == "checkpoints":
            observations = trajectory.get("observations", [])
            steps = [s for chunk in observations for s in chunk.get("steps", [])]
            if (any(type(s) is not int for s in steps) or sorted(steps) != list(range(0, 500001, 100))
                    or any(not c.get("ref") for c in observations)):
                raise ValueError(f"{name}: observation chunks must cover each step 0..500000, stride 100 exactly once")
        if mode == "replay":
            if not trajectory.get("params_ref") or not trajectory.get("substrate_config") or not trajectory.get("restarts"):
                raise ValueError(f"{name}: replay needs params, exact config and full restart states")
            if trajectory.get("rng_protocol") != "minibang_batch_v1":
                raise ValueError(f"{name}: unknown RNG schedule")
            steps = [r["step"] for r in trajectory["restarts"]]
            if (any(type(s) is not int or s % 100 for s in steps)
                    or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != 500000):
                raise ValueError(f"{name}: ordered unique restarts must cover steps 0..500000")
            if max(b - a for a, b in zip(steps, steps[1:])) > 1000:
                raise ValueError(f"{name}: export restart spacing no greater than 1000 raw steps")
    for ref in walk_refs(manifest):
        if type(ref.get("size_bytes")) is not int or ref["size_bytes"] < 1:
            raise ValueError("Every source file requires a positive byte size")
        digest = ref.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Every source file requires its SHA256 digest")


def verify_reviewed_media(manifest):
    """Prevent accidentally attaching existing labels to a different rollout."""
    pins = reviewed_media()
    for name in reviewed_trajectories():
        ref = manifest["trajectories"][name].get("video", {}).get("ref", {})
        if any(ref.get(key) != pins[name][key] for key in ("sha256", "size_bytes")):
            raise ValueError(f"{name}: video differs from the human-reviewed original; do not transfer labels to another rollout")
