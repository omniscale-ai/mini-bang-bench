"""Build the pool manifest from a normalized export and pack it as one TAR."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np

from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import (
    SCHEMA, file_ref, reviewed_trajectories, safe_relative, validate_manifest, verify_reviewed_media, walk_refs)
from mini_bang.simulators.flow_lenia_pool.restart_schema import validate_restart
from mini_bang.tasks.flow_lenia_common.annotations import POOL_ID


def video_metadata(path, *, uri=None):
    completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
                               check=True, capture_output=True, text=True, timeout=120)
    stream = json.loads(completed.stdout)["streams"][0]
    if (stream["r_frame_rate"] != "30/1" or int(stream["nb_frames"]) != 5001
            or int(stream["width"]) != 224 or int(stream["height"]) != 224):
        raise ValueError("Original annotation requires 5001 frames at 30 fps")
    return {"ref": file_ref(path, uri=uri), "first_step": 0, "step_stride": 100,
            "frame_count": 5001, "width": int(stream["width"]), "height": int(stream["height"]), "fps": 30}


def discover_source(root: Path, mode):
    root = root.resolve()
    manifest = {"schema": SCHEMA, "pool_id": POOL_ID, "trajectories": {}}
    for name in reviewed_trajectories():
        directory = root / name
        trajectory = {"last_step": 500000, "replay_verified": False}
        candidates = [root / f"{name}.mp4", directory / "video.mp4"]
        for candidate in candidates:
            if candidate.is_file():
                trajectory["video"] = video_metadata(candidate, uri=candidate.relative_to(root).as_posix())
                break
        if mode in {"replay", "checkpoints"}:
            observations = []
            for path in sorted((directory / "observations").glob("*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    steps = data["steps"].tolist()
                    if any(type(s) is not int for s in steps) or steps != sorted(set(steps)):
                        raise ValueError("Observation steps must be ordered distinct integers")
                    if not {"A", "P", "Food"} <= set(data.files):
                        raise ValueError("Observation chunks require A, P, Food and steps")
                    for field in ("A", "P", "Food", "F"):
                        if field in data.files:
                            values = data[field]
                            if (values.dtype.kind != "f" or len(values) != len(steps)
                                    or not np.isfinite(values).all()):
                                raise ValueError("Observation arrays must be finite floating fields aligned with steps")
                observations.append({"steps": steps, "ref": file_ref(path, uri=path.relative_to(root).as_posix())})
            if observations:
                trajectory["observations"] = observations
        if mode == "replay":
            trajectory["substrate_config"] = json.loads((directory / "config.json").read_text())
            params = directory / "params.npz"
            with np.load(params, allow_pickle=False) as data:
                if data["params"].dtype != np.float32 or not np.isfinite(data["params"]).all():
                    raise ValueError("params.npz must contain finite float32 params")
            trajectory["params_ref"] = file_ref(params, uri=params.relative_to(root).as_posix())
            trajectory["rng_protocol"] = "minibang_batch_v1"
            restarts = []
            for path in sorted((directory / "restarts").glob("*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    required = {"state__A", "state__P", "state__Food", "state__fK", "state__m", "state__s",
                                "state__t", "state__mass_cycle_start", "rng", "step", "batch_size",
                                "batch_index", "snapshot_interval", "jit_microbatch"}
                    if required - set(data.files):
                        raise ValueError(f"{path.name}: not a full resume checkpoint")
                    step = validate_restart(data)
                restarts.append({"step": step, "ref": file_ref(path, uri=path.relative_to(root).as_posix())})
            trajectory["restarts"] = sorted(restarts, key=lambda item: item["step"])
        manifest["trajectories"][name] = trajectory
    validate_manifest(manifest, mode=mode)
    return manifest


def pack_source(root: Path, output: Path, *, mode):
    if output.exists():
        raise FileExistsError("Archive output already exists")
    manifest = discover_source(root, mode)
    verify_reviewed_media(manifest)
    body = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # MP4 and NPZ are already compressed. Uncompressed TAR avoids wasted hours
    # recompressing gigabytes while remaining streamable over remote storage.
    with tarfile.open(output, "x") as archive:
        header = tarfile.TarInfo("pool.json")
        header.size, header.mode = len(body), 0o644
        archive.addfile(header, io.BytesIO(body))
        seen = set()
        for ref in walk_refs(manifest):
            name = str(safe_relative(ref["uri"]))
            path = root.resolve() / name
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Source files must not escape the export directory")
            if name not in seen:
                archive.add(path, arcname=name, recursive=False)
                seen.add(name)
    return {"archive": str(output), "bytes": output.stat().st_size,
            "trajectories": len(manifest["trajectories"]), "mode": mode}


def link_video_pool(root: Path, output: Path):
    """Dev-only read-only media refs: no copying or changes to the source tree."""
    if output.exists():
        raise FileExistsError("Manifest already exists")
    manifest = discover_source(root, "video")
    for ref in walk_refs(manifest):
        ref["uri"] = (root.resolve() / safe_relative(ref["uri"])).as_uri()
    # One file: the returned ref is loaded directly as the pool manifest.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return file_ref(output)
