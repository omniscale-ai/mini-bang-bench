"""The trusted input stager always runs under the execution watchdog."""
import json

import pytest

from mini_bang.framework.k8s_executable import K8sExecutableConfig
from mini_bang.framework.k8s_staging import K8sInputStage


def staged_manifest(monkeypatch, uncompressed):
    monkeypatch.setenv('MINI_BANG_STAGE_UNCOMPRESSED', str(int(uncompressed)))
    config = K8sExecutableConfig(namespace='proof', image='offline-code',
        staging_image='pinned-trusted-image', execution_timeout_seconds=0,
        execution_policy={'idle_seconds': 600})
    stage = K8sInputStage(client=None, config=config, job_id='composition-proof',
        artifacts=[{'name': 'window.npz', 'ref': {'size_bytes': 1, 'codec': 'bytes',
            'sha256': 'composition-fixture'}}])
    return stage, stage.job_manifest()


@pytest.mark.parametrize('uncompressed', [False, True])
def test_stager_command_is_wrapped_by_the_execution_watchdog(monkeypatch, uncompressed):
    stage, manifest = staged_manifest(monkeypatch, uncompressed)
    original = manifest['spec']['template']['spec']['containers'][0]['command']
    assert original[:3] == ['python', '-m', 'mini_bang.framework.execution_watchdog']
    assert original[3:6] == ['/stage/config/execution-policy.json',
        '/tmp/preparation-runtime', 'preparation']
    child = original[6:]
    assert child[:2] == ['python', '-c' if uncompressed else '-m']
    if not uncompressed:
        assert child == ['python', '-m', 'mini_bang.worker.stage_inputs']
    policy = json.loads(stage.config_map()['data']['execution-policy.json'])
    assert policy['expected_worker_version'] is None
    assert policy['idle_seconds'] == 600
