"""Separation of labelled training and unlabelled test observations."""
from mini_bang.execution.planner import SimulationPlanner
from mini_bang.tasks.raf_common.hidden_networks import NETWORK_KEY, PUBLIC_FIELD

RULE = 'Test networks allow observations but not is_raf labels; use training seeds for labelled queries'


def validate_test_label_access(visible, request):
    if visible.get('test_label_access') != 'observations_only' or 'is_raf' not in request.extras:
        return
    if PUBLIC_FIELD in visible:
        # Named test networks: raw hidden seeds are rejected in raf_common/hidden_networks.py.
        if NETWORK_KEY in request.macro_params:
            raise ValueError(f'{RULE}; no budget was spent')
        return
    protected = {int(seed) for seeds in visible.get('cases_by_saturation', {}).values() for seed in seeds}
    # Resolve the same macro seed as execution, including sample-base-derived
    # requests; checking only an explicit seed would leave an alias bypass.
    effective = SimulationPlanner().plan(request)[0].macro_params['seed']
    if effective in protected:
        raise ValueError(RULE)
