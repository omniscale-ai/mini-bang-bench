"""Benign local warm-worker fixtures; no Kubernetes or untrusted payloads."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import pytest

from mini_bang.pool import offline_staging as staging
from mini_bang.pool import offline_service as service
from mini_bang.framework.execution_watchdog import policy


def worker_fixture(tmp_path):
    source = tmp_path / 'trusted-fixture.py'
    source.write_text(
        'import os,runpy,socket,sys,time\n'
        f'api=runpy.run_path({str(Path(staging.__file__).resolve())!r},run_name="fixture")\n'
        'stream=socket.socket(fileno=int(sys.argv[-1]))\n'
        'started=time.process_time()\n'
        'while time.process_time()-started<.15: pass\n'
        'api["send"](stream,{"status":"ready","pid":os.getpid()})\n'
        'while True:\n'
        '    try: request=api["receive"](stream)\n'
        '    except OSError: break\n'
        '    started=time.process_time()\n'
        '    while time.process_time()-started<.03: pass\n'
        '    open(request["stdout_path"],"w").write("trusted preparation")\n'
        '    open(request["stderr_path"],"w").close()\n'
        '    api["send"](stream,{"pid":os.getpid(),"exit_code":int(request.get("fail",False))})\n')
    return staging.WarmStager(tmp_path / 'control', worker_path=source)


def test_two_requests_reuse_imported_worker_pid_and_reset_cpu_baseline(tmp_path):
    warm = worker_fixture(tmp_path)
    warm.control = tmp_path.parent / 'warm-cpu-baseline'  # Linux AF_UNIX paths are limited to 108 bytes.
    watchdog = Path(__file__).parents[1] / 'src/mini_bang/framework/execution_watchdog.py'
    # Force the existing /proc fallback: startup/previous task CPU must not
    # trip this request's whole-worker fuse on a persistent PID.
    source = watchdog.read_text() + '\nresources=lambda:{}\n'
    guarded = policy({'idle_seconds': 2, 'warning_seconds': .5, 'poll_seconds': .02,
                      'whole_worker_cpu_seconds': .08})
    original_popen = staging.subprocess.Popen
    try:
        warm.ensure_started()
        first = warm.execute({'artifacts': [], 'destination': str(tmp_path)},
                             policy=guarded, watchdog_source=source)
        second = warm.execute({'artifacts': [], 'destination': str(tmp_path)},
                              policy=guarded, watchdog_source=source)
        assert first['warm_worker_pid'] == second['warm_worker_pid']
        assert first['warm_worker_start'] == second['warm_worker_start']
        assert first['warm_worker_starts'] == second['warm_worker_starts'] == 1
        assert first['worker_cpu_before_seconds'] >= .1
        assert second['worker_cpu_before_seconds'] >= first['worker_cpu_before_seconds']
        assert .01 <= second['worker_cpu_delta_seconds'] < .08
        metadata = json.loads((warm.control / 'stage/execution-runtime.json').read_text())
        assert metadata['outcome'] == 'completed' and metadata['phase'] == 'preparation'
        assert metadata['whole_worker_cpu_observed_seconds'] < .08
        assert staging.subprocess.Popen is original_popen
    finally:
        warm.close()


def test_failed_preparation_is_retained_and_does_not_redownload_on_replay(tmp_path):
    warm = worker_fixture(tmp_path)
    request = {'work_id': 'readable-failed-preparation', 'fingerprint': 'same-immutable-inputs',
               'artifacts': [], 'destination': str(tmp_path), 'fail': True}
    try:
        with pytest.raises(staging.PreparationFailed):
            warm.execute(request, policy=None, watchdog_source=None)
        assert warm.starts == 1 and warm.process is None
        with pytest.raises(staging.PreparationFailed, match='Retained'):
            warm.execute(request, policy=None, watchdog_source=None)
        assert warm.starts == 1 and warm.process is None
    finally:
        warm.close()


def test_cleanup_preserves_only_verified_warm_pid_identity(monkeypatch, tmp_path):
    warm = worker_fixture(tmp_path)
    try:
        warm.ensure_started()
        actual = warm.process.pid
        assert warm.preserved_pids() == (actual,)
        original = staging.identity
        with monkeypatch.context() as patch:
            patch.setattr(staging, 'identity', lambda pid: {**original(pid), 'start': 'reused-pid'})
            assert warm.preserved_pids() == ()
        slot = service.Service.__new__(service.Service)
        slot.config = {'role': 'stager', 'staging_root': str(tmp_path / 'inputs')}
        slot.warm = warm
        calls = []
        monkeypatch.setattr(service, 'stop_processes', lambda uid, preserve, **kwargs: calls.append((preserve, kwargs)))
        monkeypatch.setattr(service, 'clean_path', lambda path: None)
        slot.cleanup(peer_pid=1234)
        assert calls == [((1234, actual), {'stager': True})]
        selected = service.cleanup_targets({1: (0,), actual: (0,), 1234: (0,), 9999: (0,), 7777: (10001,)},
                                           10001, (1234, actual), stager=True)
        assert selected == [9999, 7777]
    finally:
        warm.close()
