from types import SimpleNamespace

import pytest

from mini_bang.framework.k8s_staging import K8sInputStage, MAX_STAGED_INPUT_FILES


def inputs(count, size=1024):
    return [{'name': f'window_{i:03d}.npz',
             'ref': {'codec': 'bytes', 'size_bytes': size, 'sha256': 'verified'},
             'fields': ['A', 'P', 'Food']} for i in range(count)]


@pytest.mark.parametrize('count', [129, 140, 300, MAX_STAGED_INPUT_FILES])
def test_cumulative_train_test_inputs_fit_without_changing_payloads(count):
    artifacts = inputs(count)
    stage = K8sInputStage(client=None, config=SimpleNamespace(), job_id='evaluation', artifacts=artifacts)
    assert stage.artifacts is artifacts
    assert len(stage.artifacts) == count


@pytest.mark.parametrize('count', [0, MAX_STAGED_INPUT_FILES + 1])
def test_staging_count_is_still_bounded(count):
    with pytest.raises(ValueError, match='input files'):
        K8sInputStage(client=None, config=None, job_id='evaluation', artifacts=inputs(count))


def test_cumulative_capacity_retains_byte_and_field_boundaries():
    with pytest.raises(ValueError, match='16 GiB'):
        K8sInputStage(client=None, config=None, job_id='evaluation',
                      artifacts=inputs(300, 64 * 1024 ** 2))
    artifacts = inputs(140)
    artifacts[-1]['fields'] = ['labels']
    with pytest.raises(ValueError, match='approved state fields'):
        K8sInputStage(client=None, config=None, job_id='evaluation', artifacts=artifacts)
