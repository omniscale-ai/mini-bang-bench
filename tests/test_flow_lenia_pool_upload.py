"""An uploaded pool gets readable object names; sha256 values stay in the refs."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

from mini_bang.simulators.flow_lenia_pool.preparation.pool_archive import upload_pool
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import SCHEMA, file_ref
from tests.synthetic_pool_media import TRAJECTORIES, use_synthetic_media

UPLOAD = r"\d{4}-\d{2}-\d{2}/\d{4}(-\d{2}){5}-\d{6}--pool--\d{4}"


@pytest.fixture(autouse=True)
def _synthetic_media(monkeypatch):
    use_synthetic_media(monkeypatch)


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, source: str, bucket: str, key: str) -> None:
        self.objects[key] = Path(source).read_bytes()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[Key] = Body


def test_upload_names_objects_by_time_and_pool_path(tmp_path, monkeypatch) -> None:
    s3 = _FakeS3()
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *args, **kwargs: s3))
    trajectories = {}
    for name in TRAJECTORIES:
        path = tmp_path / "pool" / name / "params.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
        trajectories[name] = {"last_step": 500000, "params_ref": file_ref(path)}
    pool = tmp_path / "pool" / "pool.json"
    pool.write_text(json.dumps({"schema": SCHEMA, "trajectories": trajectories}))

    ref = upload_pool(pool, bucket="pilot", prefix="pool/test-pool")

    manifest_key = ref["uri"].removeprefix("s3://pilot/")
    upload = re.fullmatch(rf"pool/test-pool/manifests/({UPLOAD})\.json", manifest_key).group(1)
    assert ref["sha256"] == hashlib.sha256(s3.objects[manifest_key]).hexdigest()
    uploaded = json.loads(s3.objects[manifest_key])["trajectories"]
    for name in TRAJECTORIES:
        file = uploaded[name]["params_ref"]
        assert file["uri"] == f"s3://pilot/pool/test-pool/blobs/{upload}/{name}/params.npz"
        assert file["sha256"] == hashlib.sha256(name.encode()).hexdigest()
    assert not any(re.search(r"[0-9a-f]{64}", key) for key in s3.objects)


def test_upload_sends_indexed_shards_and_rewrites_their_index(tmp_path, monkeypatch) -> None:
    s3 = _FakeS3()
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *args, **kwargs: s3))
    root = tmp_path / "pool"
    trajectories = {}
    for number, name in enumerate(TRAJECTORIES, start=1):
        video = root / name / "video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video " + name.encode())
        shard = root / "indexed-frames" / f"simulation-{number:02d}" / "steps-000000000-000000100" / "A.frames.zip"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"shard " + name.encode())
        index = root / "indexed-frames" / f"simulation-{number:02d}" / "index.json"
        index.write_text(json.dumps({"trajectory": name, "blocks": [
            {"steps": [0, 100], "fields": {"A": file_ref(shard)}}]}))
        trajectories[name] = {"last_step": 500000, "video": {"ref": file_ref(video)}, "index_ref": file_ref(index)}
    pool = root / "indexed-pool.json"
    pool.write_text(json.dumps({"schema": SCHEMA, "trajectories": trajectories}))

    ref = upload_pool(pool, bucket="pilot", prefix="flow-lenia-pool")

    manifest = json.loads(s3.objects[ref["uri"].removeprefix("s3://pilot/")])
    first = manifest["trajectories"][TRAJECTORIES[0]]
    index_key = first["index_ref"]["uri"].removeprefix("s3://pilot/")
    assert index_key.endswith("/indexed-frames/simulation-01/index.json")
    assert first["index_ref"]["sha256"] == hashlib.sha256(s3.objects[index_key]).hexdigest()
    shard_ref = json.loads(s3.objects[index_key])["blocks"][0]["fields"]["A"]
    assert shard_ref["uri"].startswith("s3://pilot/flow-lenia-pool/blobs/")
    assert s3.objects[shard_ref["uri"].removeprefix("s3://pilot/")] == b"shard " + TRAJECTORIES[0].encode()
    assert first["video"]["ref"]["uri"].endswith(f"/{TRAJECTORIES[0]}/video.mp4")
