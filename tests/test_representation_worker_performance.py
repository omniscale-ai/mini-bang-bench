"""Production harness equivalence, bounded concurrency and hard-failure checks."""
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

import numpy as np
import pytest

from mini_bang.tasks.flow_lenia_common import representation_worker as worker

SRC = Path(__file__).resolve().parents[1] / "src"


def executable_image_env():
    """Child environment of the executable images: mini_bang importable, one BLAS thread.

    Without the thread variables OpenBLAS starts one thread per CPU on import and
    reserves their address space before the test sets RLIMIT_AS.
    """
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    pythonpath = os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")]))
    return {**os.environ, "PYTHONPATH": pythonpath, **{name: "1" for name in threads}}


def chunks(root, *, count=4):
    files = []
    for window in range(count):
        names = []
        for part, length in enumerate((3, 4, 2)):
            name = f"window-{window}-{part}.npz"
            values = np.arange(length * 8, dtype=np.float32).reshape(length, 2, 2, 2)
            np.savez_compressed(root / name, A=values + window, P=values.astype(np.float16) / 3,
                                Food=np.asfortranarray(values.astype(np.float64) + part))
            names.append(name)
        files.append(names)
    return files


class LegacyWindow:
    """The former stream algorithm, retained as an independent behavior oracle."""

    def __init__(self, paths, thinning):
        self.paths, self.thinning = paths, thinning

    def __iter__(self):
        counts = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as archive:
                counts.append(len(archive["A"]))
        total, offset = sum(counts), 0
        for path, count in zip(self.paths, counts):
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in ("A", "P", "Food")}
                if any(len(values) != count for values in arrays.values()):
                    raise ValueError("Checkpoint chunk fields are not aligned")
                for index in range(count):
                    absolute = offset + index
                    if absolute % self.thinning == 0 or absolute == total - 1:
                        yield {name: values[index] for name, values in arrays.items()}
            offset += count


def legacy_features(config, root):
    features = {str(factor): [] for factor in (1, 2, 4)}
    for names in config["files"]:
        paths = [root / name for name in names]
        for factor in (1, 2, 4):
            namespace = {"__name__": "submitted_metric"}
            exec(compile(config["code"], "submitted_metric.py", "exec"), namespace)
            result = namespace[config.get("entrypoint", "featurize")](LegacyWindow(paths, factor))
            features[str(factor)].append(np.asarray(result, dtype=np.float64).reshape(-1).tolist())
    return {"features": features}


@pytest.mark.parametrize("entrypoint", ["featurize", "score"])
def test_production_parallel_matches_former_serial_values_and_order(tmp_path, entrypoint):
    code = """
import numpy as np
calls = 0
def featurize(window):
    global calls
    calls += 1
    # Re-iteration and mutation must not change the next stream or thinning.
    first = list(window)
    values = [sum(float(np.sum(c[name])) for c in first) for name in ('A', 'P', 'Food')]
    first[0]['A'][...] = -100000
    repeat = list(window)
    return values + [float(np.sum(repeat[0]['A'])), len(first), calls]
def score(window):
    return featurize(window)[2]
"""
    config = {"files": chunks(tmp_path), "code": textwrap.dedent(code), "entrypoint": entrypoint}
    expected = legacy_features(config, tmp_path)
    assert worker.extract(config, workers=1, checkpoint_root=tmp_path) == expected
    assert worker.extract(config, workers=2, checkpoint_root=tmp_path) == expected
    if entrypoint == "featurize":
        assert [row[-2] for row in expected["features"]["1"]] == [9] * 4
        assert [row[-2] for row in expected["features"]["2"]] == [5] * 4
        assert [row[-2] for row in expected["features"]["4"]] == [3] * 4
        assert all(row[-1] == 1 for rows in expected["features"].values() for row in rows)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_count_reads_header_only_without_array_decompression(tmp_path, monkeypatch, version):
    path = tmp_path / "compressed.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("A.npy", "w", force_zip64=True) as stream:
            np.lib.format.write_array(stream, np.zeros((100000, 3), dtype=np.float16), version=version)
    original, bytes_read = zipfile.ZipExtFile.read, []

    def read(self, n=-1):
        value = original(self, n)
        bytes_read.append(len(value))
        return value

    monkeypatch.setattr(zipfile.ZipExtFile, "read", read)
    assert worker.checkpoint_count(path) == 100000
    assert sum(bytes_read) < 1024


def test_iteration_inflates_each_field_once_and_keeps_plain_numpy_views(tmp_path, monkeypatch):
    files = chunks(tmp_path, count=1)[0]
    calls, original = [], np.lib.npyio.NpzFile.__getitem__

    def getitem(self, name):
        calls.append(name)
        return original(self, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    states = list(worker.Window([tmp_path / name for name in files], 4))
    assert calls == ["A", "P", "Food"] * 3
    assert len(states) == 3
    assert all(type(state) is dict and set(state) == {"A", "P", "Food"} for state in states)
    assert all(isinstance(value, np.ndarray) for state in states for value in state.values())
    assert states[0]["A"].dtype == np.float32
    assert states[0]["P"].dtype == np.float16
    assert states[0]["Food"].dtype == np.float64


def test_stream_does_not_preload_later_chunk_arrays(tmp_path, monkeypatch):
    files = chunks(tmp_path, count=1)[0]
    calls, original = [], np.lib.npyio.NpzFile.__getitem__

    def getitem(self, name):
        calls.append(name)
        return original(self, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    stream = iter(worker.Window([tmp_path / name for name in files], 1))
    next(stream)
    assert calls == ["A", "P", "Food"]
    stream.close()


@pytest.mark.parametrize("contents", [{"P": np.zeros(2)}, {"A": np.array(3)}])
def test_invalid_state_header_is_not_silently_empty(tmp_path, contents):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **contents)
    with pytest.raises((KeyError, ValueError)):
        worker.checkpoint_count(path)


def test_misaligned_food_fails_extraction(tmp_path):
    np.savez(tmp_path / "bad.npz", A=np.zeros(3), P=np.zeros(3), Food=np.zeros(2))
    with pytest.raises(worker.TrustedLoaderError, match="not aligned"):
        worker.extract({"code": "def featurize(w): return [sum(1 for _ in w)]",
                        "files": [["bad.npz"]]}, checkpoint_root=tmp_path)


@pytest.mark.parametrize("code,message", [
    ("def featurize(w): return []", "1..128"),
    ("def featurize(w): return [float('nan')]", "1..128"),
    ("def featurize(w): return [[1]]", "1..128"),
    ("def featurize(w): return [1] * (1 if len(list(w)) == 9 else 2)", "constant"),
    ("def featurize(w): raise RuntimeError('submitted failure')", "submitted failure"),
])
def test_submitted_errors_are_propagated_without_result(tmp_path, code, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        worker.extract({"code": code, "files": chunks(tmp_path, count=1)}, checkpoint_root=tmp_path)


def test_dimension_must_match_across_windows(tmp_path):
    np.savez(tmp_path / "a.npz", A=np.zeros(2), P=np.zeros(2), Food=np.zeros(2))
    np.savez(tmp_path / "b.npz", A=np.ones(2), P=np.ones(2), Food=np.ones(2))
    with pytest.raises(ValueError, match="constant"):
        worker.extract({"code": "def featurize(w): return [1] * (1 + int(next(iter(w))['A']))",
                        "files": [["a.npz"], ["b.npz"]]}, checkpoint_root=tmp_path)


@pytest.mark.parametrize("quota,expected", [("200000 100000", 2), ("100000 100000", 1),
                                           ("150000 100000", 1), ("max 100000", 2), ("bad", 1)])
def test_worker_count_obeys_affinity_and_cgroup2_quota(tmp_path, monkeypatch, quota, expected):
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _: {12, 13})
    (tmp_path / "cpu.max").write_text(quota)
    assert worker.allocated_workers(2, cgroup_root=tmp_path) == expected
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _: {12})
    assert worker.allocated_workers(2, cgroup_root=tmp_path) == 1


def test_cgroup1_cpu_quota_and_unexpected_worker_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _: {0, 1, 2})
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu/cpu.cfs_quota_us").write_text("100000")
    (tmp_path / "cpu/cpu.cfs_period_us").write_text("100000")
    assert worker.allocated_workers(2, cgroup_root=tmp_path) == 1
    with pytest.raises(ValueError, match="one or two"):
        worker.allocated_workers(3, cgroup_root=tmp_path)


def test_two_workers_overlap_and_never_create_a_third(tmp_path):
    if worker.allocated_workers(2) < 2:
        pytest.skip("This environment allocates only one CPU")
    code = f"""
import os, time
from pathlib import Path
def featurize(window):
    marker = Path({str(tmp_path)!r}) / ('started-' + str(os.getpid()))
    marker.write_text('active')
    deadline = time.monotonic() + 5
    while len(list(marker.parent.glob('started-*'))) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError('Two workers did not overlap')
        time.sleep(.01)
    return [os.getpid()]
"""
    existing_children = {child.pid for child in multiprocessing.active_children()}
    output = worker.extract({"code": textwrap.dedent(code), "files": chunks(tmp_path, count=6)},
                            checkpoint_root=tmp_path)
    pids = {row[0] for row in output["features"]["1"]}
    assert len(pids) == len(list(tmp_path.glob("started-*"))) == 2
    remaining = {child.pid for child in multiprocessing.active_children()}
    assert pids.isdisjoint(existing_children | remaining)
    assert remaining <= existing_children


@pytest.mark.parametrize("failure", ["raise RuntimeError('explicit child failure')", "os._exit(31)"])
def test_failed_worker_cleans_stuck_sibling_without_waiting_outer_deadline(tmp_path, failure):
    # The subprocess timeout protects this regression test, not scientific jobs.
    # One submitted call waits forever while the other fails or exits hard.
    np.savez(tmp_path / "blocked.npz", A=np.zeros(2), P=np.zeros(2), Food=np.zeros(2))
    np.savez(tmp_path / "failed.npz", A=np.ones(2), P=np.ones(2), Food=np.ones(2))
    code = f"""
import os, time
from pathlib import Path
def featurize(window):
    if int(next(iter(window))['A']) == 0:
        Path({str(tmp_path / 'blocked-started')!r}).write_text('started')
        while True: time.sleep(1)
    deadline = time.monotonic() + 5
    while not Path({str(tmp_path / 'blocked-started')!r}).exists():
        if time.monotonic() > deadline: raise RuntimeError('Sibling was not started')
        time.sleep(.01)
    {failure}
"""
    config = {"code": textwrap.dedent(code), "files": [["blocked.npz"], ["failed.npz"]]}
    script = "\n".join([
        "import json,multiprocessing,sys",
        "from mini_bang.tasks.flow_lenia_common.representation_worker import extract,allocated_workers",
        "assert allocated_workers(2) == 2",
        "try:",
        f"    extract(json.loads({json.dumps(json.dumps(config))}), checkpoint_root={str(tmp_path)!r})",
        "except Exception as exc:",
        "    print(type(exc).__name__)",
        "else:",
        "    raise AssertionError('Failed extraction returned a result')",
        "assert not multiprocessing.active_children()",
    ])
    if worker.allocated_workers(2) < 2:
        pytest.skip("This environment allocates only one CPU")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=15, check=True, env=executable_image_env())
    assert ("BrokenProcessPool" if "_exit" in failure else "ParticipantContractError") in result.stdout


def test_fork_workers_keep_process_memory_limit_and_cannot_allocate_past_it(tmp_path):
    config = {"files": chunks(tmp_path, count=2), "code": """
import resource
def featurize(window):
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    try:
        oversized = bytearray(limit + 1)
    except MemoryError:
        return [limit, 1]
    raise RuntimeError('Worker escaped inherited memory limit')
"""}
    # A small real OS limit in an isolated test subprocess, not a modification
    # of the official Job's unchanged aggregate 4-GiB cgroup memory limit.
    script = "\n".join([
        "import json,resource",
        "from mini_bang.tasks.flow_lenia_common.representation_worker import extract",
        "limit = 512 * 1024**2",
        "resource.setrlimit(resource.RLIMIT_AS, (limit, limit))",
        f"result=extract(json.loads({json.dumps(json.dumps(config))}), checkpoint_root={str(tmp_path)!r})",
        "assert all(row == [limit, 1] for rows in result['features'].values() for row in rows)",
        "print('inherited_memory_limit_enforced')",
    ])
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=15, check=True, env=executable_image_env())
    assert "inherited_memory_limit_enforced" in result.stdout


def test_actual_standalone_harness_cli_preserves_output_schema(tmp_path):
    files = [[str(tmp_path / name) for name in names] for names in chunks(tmp_path)]
    config = {"files": files, "code": "def featurize(w): return [sum(float(c['Food'].sum()) for c in w)]"}
    manifest, output = tmp_path / "input.json", tmp_path / "output.json"
    manifest.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, worker.__file__, str(manifest), str(output)],
                            capture_output=True, text=True, timeout=15, check=True)
    assert result.stdout == ""
    assert json.loads(output.read_text()) == legacy_features(config, Path("/unused"))
