from types import SimpleNamespace

import pytest

from mini_bang.tasks.raf_common.policy import RAFTaskPolicy


def session(metadata):
    return SimpleNamespace(task_id='raf/mechanism-seq', instance_spec=SimpleNamespace(metadata=metadata),
        task_spec=SimpleNamespace(config={'api': {'max_generate_calls': 50}}))


def test_task_default_budget_applies_without_instance_limit_and_override_is_explicit():
    policy = RAFTaskPolicy()
    assert policy.max_calls(session({})) == 50
    assert policy.max_calls(session({'api_limits': {'max_generate_calls': 160}})) == 160


@pytest.mark.parametrize('limit', [True, False, 0, -1, 1.5, '160', None])
def test_invalid_instance_budgets_fail_closed(limit):
    with pytest.raises(ValueError, match='positive integer'):
        RAFTaskPolicy().max_calls(session({'api_limits': {'max_generate_calls': limit}}))
