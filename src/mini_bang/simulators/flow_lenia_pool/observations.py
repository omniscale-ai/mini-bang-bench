from __future__ import annotations

import subprocess

import numpy as np

from mini_bang.artifacts import ArtifactRef
from mini_bang.simulators.flow_lenia_pool.indexed_frames import load_trajectory_index, read_frames, FORMAT

POOL_SCHEMA = "mini_bang.flow_pool.v1"


class PoolObservations:
    def __init__(self, manifest, trajectory, resolver, *, mode):
        if manifest.get("schema") != POOL_SCHEMA:
            raise ValueError("Unsupported Flow Lenia pool schema")
        self.manifest, self.resolver, self.mode = manifest, resolver, mode
        self.trajectory = load_trajectory_index(manifest["trajectories"][trajectory], resolver,
                                                trajectory=trajectory)

    def materialize(self, ref):
        path = self.resolver.materialize(ArtifactRef.from_dict(ref)).local_path
        if path is None:
            raise RuntimeError("Pool artifact did not materialize")
        return path

    def observe(self, steps, fields):
        if not fields or len(fields) != len(set(fields)) or set(fields) - {"A", "P", "F", "Food", "rgb"}:
            raise ValueError("Unsupported observation fields")
        if any(type(s) is not int or s < 0 or s > self.trajectory["last_step"] for s in steps):
            raise ValueError("Observation outside the simulation")
        if any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError("Observation steps must be strictly increasing")
        if self.mode in {'video', 'checkpoints'} and self.trajectory.get('format') == FORMAT:
            if self.mode == 'video' and list(fields) != ['rgb']:
                raise ValueError('Only RGB is available from videos')
            result = read_frames(self.trajectory, steps, fields, self.resolver)
        elif self.mode == "video":
            if list(fields) != ["rgb"]:
                raise ValueError("Only RGB is available from videos")
            result = {"rgb": self.video_frames(steps)}
        elif self.mode == "checkpoints":
            raw_fields = [field for field in fields if field != "rgb"]
            result = self.saved_fields(steps, raw_fields) if raw_fields else {}
            if "rgb" in fields:
                result["rgb"] = self.video_frames(steps)
        elif self.mode == "replay":
            from mini_bang.simulators.flow_lenia_pool.replay import replay_observations
            result = replay_observations(self, steps, fields)
        else:
            raise ValueError("Unknown observation mode")
        return {"steps": np.asarray(steps, dtype=np.int64), **result}

    def video_frames(self, steps):
        video = self.trajectory.get("video")
        if not video:
            raise ValueError("The pool has no video for this case")
        stride = video["step_stride"]
        first = video["first_step"]
        if any((s - first) % stride or s < first for s in steps):
            raise ValueError("Requested step is not an available video frame")
        indexes = [(s - first) // stride for s in steps]
        if max(indexes) >= video["frame_count"]:
            raise ValueError("Requested frame is beyond the video")
        expression = "+".join(f"eq(n\\,{index})" for index in indexes)
        command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(self.materialize(video["ref"])),
                   "-vf", f"select={expression}", "-fps_mode", "passthrough", "-frames:v", str(len(indexes)),
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        completed = subprocess.run(command, capture_output=True, timeout=180, check=False)
        if completed.returncode:
            raise RuntimeError("Video observation decoding failed")
        expected = len(steps) * video["height"] * video["width"] * 3
        if len(completed.stdout) != expected:
            raise ValueError("Video frame count does not match the pool manifest")
        return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(len(steps), video["height"], video["width"], 3)

    def saved_fields(self, steps, fields):
        result = {key: [] for key in fields}
        snapshots = self.trajectory.get("observations", [])
        assignments = {}
        for step in steps:
            candidates = [(i, item) for i, item in enumerate(snapshots) if step in item["steps"]]
            if len(candidates) != 1:
                raise ValueError("Requested checkpoint step is missing or duplicated")
            index, item = candidates[0]
            assignments.setdefault(index, []).append((step, item["steps"].index(step)))
        found = {}
        total_bytes = 0
        for index, selected in assignments.items():
            item = snapshots[index]
            with np.load(self.materialize(item["ref"]), allow_pickle=False) as data:
                if list(data["steps"]) != item["steps"]:
                    raise ValueError("Checkpoint times differ from manifest")
                for field in fields:
                    if field not in data.files:
                        raise ValueError("Requested field is not present in the pool")
                    values = data[field]
                    if values.shape[0] != len(item["steps"]) or values.dtype.hasobject:
                        raise ValueError("Malformed checkpoint field")
                    for step, row in selected:
                        value = np.array(values[row], copy=True)
                        total_bytes += value.nbytes
                        if total_bytes > 256 * 1024 * 1024:
                            raise ValueError("Observation exceeds 256 MiB; request fewer states")
                        found[(step, field)] = value
        for field in fields:
            result[field] = np.stack([found[(step, field)] for step in steps])
        return result
