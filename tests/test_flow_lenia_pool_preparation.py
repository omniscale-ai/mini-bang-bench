from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from mini_bang.simulators.flow_lenia_pool.preparation.original_export import normalized_config, export_original, convert_restart
from mini_bang.simulators.flow_lenia_pool.preparation.pool_archive import extract_archive
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import (
    SCHEMA, file_ref, safe_relative, validate_manifest, verify_reviewed_media)
from mini_bang.simulators.flow_lenia_pool.preparation.pool_packing import discover_source
from mini_bang.client.cli.io import read_json
from mini_bang.client.readable_files import MAX_JSON_BYTES
from mini_bang.simulators.flow_lenia_pool.restart_schema import validate_restart
from tests.synthetic_pool_media import MEDIA, TRAJECTORIES, use_synthetic_media


@pytest.fixture(autouse=True)
def _synthetic_media(monkeypatch):
    use_synthetic_media(monkeypatch)


def video_ref():
    return {"first_step": 0, "step_stride": 100, "frame_count": 5001, "width": 224, "height": 224, "fps": 30,
            "ref": {"uri": "video.mp4", "sha256": "a" * 64, "size_bytes": 1}}


def manifest():
    trajectory = {"last_step": 500000, "video": video_ref(), "observations": [{
        "steps": list(range(0, 500001, 100)), "ref": {"uri": "states.npz", "sha256": "b" * 64, "size_bytes": 1}}]}
    return {"schema": SCHEMA, "trajectories": {name: copy.deepcopy(trajectory) for name in TRAJECTORIES}}


@pytest.mark.parametrize("corruption", ["missing_step", "duplicate_step", "bool_step", "missing_video", "wrong_size"])
def test_checkpoint_manifest_rejects_incomplete_original_coverage(corruption):
    pool = manifest()
    validate_manifest(pool, mode="checkpoints")
    trajectory = pool["trajectories"][TRAJECTORIES[0]]
    steps = trajectory["observations"][0]["steps"]
    if corruption == "missing_step":
        steps.pop()
    elif corruption == "duplicate_step":
        steps.append(100)
    elif corruption == "bool_step":
        steps[0] = False
    elif corruption == "missing_video":
        del trajectory["video"]
    else:
        trajectory["video"]["width"] = 128
    with pytest.raises(ValueError):
        validate_manifest(pool, mode="checkpoints")


def test_reviewed_media_identity_not_just_matching_names_and_fps():
    pool = manifest()
    for name, trajectory in pool["trajectories"].items():
        trajectory["video"]["ref"].update(MEDIA[name])
    verify_reviewed_media(pool)
    pool["trajectories"][TRAJECTORIES[0]]["video"]["ref"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="human-reviewed"):
        verify_reviewed_media(pool)


def original_source(tmp_path, *, food=False, missing_food=False):
    source = tmp_path / "original"
    config = {"substrate": {"substrate": "lenia_flow", "grid_size": 2, "C": 1, "food": food,
                            "M": "1", "mutations": True, "mutation_sz": 2, "mutation_p": 0.05,
                            "clip1": float("inf"), "flow_sigma": 0.65},
              "rollout": {"substrate": {"grid_size": 4}}}
    import yaml
    for name in TRAJECTORIES:
        root = source / name
        (root / "apf_logs").mkdir(parents=True)
        (root / "config.yaml").write_text(yaml.safe_dump(config))
        (root / "video.mp4").write_bytes(b"synthetic-video-fixture")
        np.save(root / "params.npy", np.zeros(8, dtype=np.float32))
        np.savetxt(root / "frame_times.csv", np.column_stack([np.arange(5001) * 100, np.arange(5001) / 30]),
                   delimiter=",", header="step,video_sec", comments="")
        arrays = {"A": np.ones((5001, 2, 2, 1), dtype=np.float16), "P": np.zeros((5001, 2, 2, 1), dtype=np.float16)}
        if not missing_food:
            arrays["Food"] = np.zeros((5001, 2, 2), dtype=np.float32)
        np.savez_compressed(root / "apf_logs/chunk.npz", steps=np.arange(5001) * 100, **arrays)
    return source


def test_original_export_preserves_fp16_and_only_reconstructs_disabled_food(tmp_path, monkeypatch):
    source = original_source(tmp_path, missing_food=True)
    output = tmp_path / "normalized"
    result = export_original(source, output)
    assert len(result["trajectories"]) == len(TRAJECTORIES)
    config = json.loads((output / TRAJECTORIES[0] / "config.json").read_text())
    assert config["grid_size"] == 4 and config["M"] == [[1]] and config["clip1"] == "inf"
    assert config["mutation"] is True and config["debug_return_F"] is True
    with np.load(output / TRAJECTORIES[0] / "observations/chunk_00000.npz", allow_pickle=False) as data:
        assert data["A"].dtype == np.float16 and data["P"].dtype == np.float16
        assert data["Food"].shape == (5001, 2, 2) and not data["Food"].any()
    monkeypatch.setattr("mini_bang.simulators.flow_lenia_pool.preparation.pool_packing.video_metadata", lambda path, uri=None: {
        **video_ref(), "ref": file_ref(path, uri=uri)})
    packed = discover_source(output, "checkpoints")
    assert len(packed["trajectories"]) == len(TRAJECTORIES)
    with pytest.raises(FileExistsError):
        export_original(source, output)


def test_enabled_food_cannot_be_fabricated(tmp_path):
    source = original_source(tmp_path, food=True, missing_food=True)
    with pytest.raises(ValueError, match="Food was enabled"):
        export_original(source, tmp_path / "normalized")


def test_unresolved_config_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("substrate:\n  substrate: lenia_flow\n  food: false\n  grid_size: '${grid}'\n")
    with pytest.raises(ValueError, match="interpolation"):
        normalized_config(path)


def test_fp16_is_not_relabelled_as_exact_restart():
    pytest.importorskip("jax")
    required = {"resume_batch_rng_key", "resume_batch_size", "resume_batch_index", "resume_snapshot_interval",
                "resume_jit_microbatch", "state_t", "state_mass_cycle_start"}
    fake = SimpleNamespace(files=list(required))
    with pytest.raises(ValueError, match="fp16"):
        convert_restart(fake, {"A": np.zeros(1, dtype=np.float16), "P": np.zeros(1, dtype=np.float16)},
                        0, 0, None, None)


@pytest.mark.parametrize("name", ["../escape", "/root/file", "a/../../escape", "a\\b", "file:///secret", "C:/data", ""])
def test_archive_path_validation(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_archive_rejects_links_and_does_not_overwrite(tmp_path):
    source = tmp_path / "bad.tar"
    with tarfile.open(source, "w") as archive:
        member = tarfile.TarInfo("payload")
        member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        archive.addfile(member)
    with pytest.raises(ValueError, match="links"):
        extract_archive(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("user-data")
    with pytest.raises(FileExistsError):
        extract_archive(source, target)
    assert (target / "keep").read_text() == "user-data"


def test_archive_roundtrip_digest_and_no_unlisted_files(tmp_path):
    body = b"original-npz-bytes"
    data = tmp_path / "data.npz"
    data.write_bytes(body)
    ref = file_ref(data, uri="data.npz")
    manifest = {"schema": SCHEMA, "trajectories": {name: {
        "last_step": 500000, "params_ref": ref, "replay_verified": True} for name in TRAJECTORIES}}
    source = tmp_path / "good.tar"
    with tarfile.open(source, "w") as archive:
        for name, content in (("pool.json", json.dumps(manifest).encode()), ("data.npz", body)):
            header = tarfile.TarInfo(name)
            header.size = len(content)
            archive.addfile(header, io.BytesIO(content))
    path = extract_archive(source, tmp_path / "imported")
    imported = read_json(path)
    assert all(t["replay_verified"] is False for t in imported["trajectories"].values())
    assert (path.parent / "data.npz").read_bytes() == body


def test_archive_rejects_digest_mismatch_and_directory_member_bombs(tmp_path):
    pool = {"schema": SCHEMA, "trajectories": {n: {"last_step": 500000, "params_ref": {
        "uri": "states.npz", "sha256": "0" * 64, "size_bytes": 1}} for n in TRAJECTORIES}}
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, data in (("pool.json", json.dumps(pool).encode()), ("states.npz", b"x")):
            header = tarfile.TarInfo(name)
            header.size = len(data)
            archive.addfile(header, io.BytesIO(data))
    with pytest.raises(ValueError, match="digest"):
        extract_archive(archive_path, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()
    directory_archive = tmp_path / "directories.tar"
    with tarfile.open(directory_archive, "w") as archive:
        for index in range(3):
            header = tarfile.TarInfo(f"directory-{index}")
            header.type = tarfile.DIRTYPE
            archive.addfile(header)
    with pytest.raises(ValueError, match="members"):
        extract_archive(directory_archive, tmp_path / "directories", max_files=2)


@pytest.mark.parametrize("archive_format", ["tar", "zip"])
@pytest.mark.parametrize("large_manifest", [False, True])
def test_archive_import_recognizes_outer_container_with_npz_member(tmp_path, archive_format, large_manifest):
    states = tmp_path / "states.npz"
    np.savez_compressed(states, A=np.ones((2, 2), dtype=np.float16))
    pool = {"schema": SCHEMA, "trajectories": {name: {"last_step": 500000,
        "params_ref": file_ref(states, uri="states.npz")} for name in TRAJECTORIES}}
    if large_manifest:
        pool["original_steps"] = list(range(5001))
    members = [("pool.json", json.dumps(pool).encode()), ("states.npz", states.read_bytes())]
    source = tmp_path / ("pool." + archive_format)
    if archive_format == "tar":
        with tarfile.open(source, "w") as archive:
            for name, data in members:
                header = tarfile.TarInfo(name)
                header.size = len(data)
                archive.addfile(header, io.BytesIO(data))
        # ZIP detection alone mistakes the final embedded NPZ for the archive.
        assert tarfile.is_tarfile(source) and zipfile.is_zipfile(source)
    else:
        with zipfile.ZipFile(source, "w") as archive:
            for name, data in members:
                archive.writestr(name, data)
    destination = tmp_path / "imported"
    manifest_path = extract_archive(source, destination)
    assert manifest_path == destination / "pool.json"
    assert (destination / "states.npz").read_bytes() == states.read_bytes()
    imported = read_json(manifest_path)
    assert imported["trajectories"][TRAJECTORIES[0]]["params_ref"]["uri"] == (destination / "states.npz").as_uri()
    assert imported.get("original_steps") == pool.get("original_steps")
    assert all(path.stat().st_size <= MAX_JSON_BYTES for path in destination.rglob("*.json"))


def test_real_flow_restart_initializes_static_rt_without_resampling_dynamic_state(tmp_path):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from mini_bang.simulators.flow_lenia.substrate_factory import create_flow_lenia_substrate
    from mini_bang.simulators.flow_lenia_pool.replay import ReplayCursor, load_cursor
    config = {"grid_size": 8, "C": 1, "M": [[1]], "kernel_components": 1, "seed_patch_size": 2,
              "seed_mode": "center", "render_mode": "A", "sobel_impl": "lax"}
    substrate = create_flow_lenia_substrate(**config)
    params = jnp.zeros(8, dtype=jnp.float32)
    initial = substrate.init_state(jax.random.PRNGKey(17), params)
    state = {"state__" + k: np.asarray(v) for k, v in initial.items()}
    path = tmp_path / "restart.npz"
    np.savez_compressed(path, **state, step=np.array(0), rng=np.array([0, 42], dtype=np.uint32),
                        batch_size=np.array(3), batch_index=np.array(1), snapshot_interval=np.array(2), jit_microbatch=np.array(2))
    with np.load(path, allow_pickle=False) as data:
        assert validate_restart(data) == 0
    reader = SimpleNamespace(materialize=lambda ref: path)
    fresh = create_flow_lenia_substrate(**config)
    assert not hasattr(fresh, "RT")
    restored = load_cursor(reader, {"step": 0, "ref": {}}, fresh, params)
    expected = ReplayCursor(substrate, params, initial, np.array([0, 42], dtype=np.uint32),
        start_step=0, batch_size=3, batch_index=1, snapshot_interval=2, jit_microbatch=2).at(4)
    actual = restored.at(4)
    for key in expected:
        np.testing.assert_array_equal(np.asarray(actual[key]), np.asarray(expected[key]))
    reused = load_cursor(reader, {"step": 0, "ref": {}}, fresh, params, compiled_advance=restored.advance)
    assert reused.advance is restored.advance
    np.testing.assert_array_equal(np.asarray(reused.at(4)["A"]), np.asarray(expected["A"]))
