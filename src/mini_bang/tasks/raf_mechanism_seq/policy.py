from __future__ import annotations

from mini_bang.execution.planner import SimulationPlanner
from mini_bang.tasks.raf_common import hidden_networks
from mini_bang.tasks.raf_common.hidden_networks import NETWORK_KEY
from mini_bang.tasks.raf_common.policy import RAFMonotonicSequencePolicy

# Instances with visible.test_chains name their test chains; their macro seeds stay on the
# server (raf_common/hidden_networks.py). Instances with visible.test_seeds list raw seeds.
NAMED_TEST_ROLE, SEED_TEST_ROLE = 'test_chains', 'test_seeds'
NAMED_RULE = ('Test chains are addressed by name (macro_params.network); their macro seeds are hidden and '
              'a raw macro seed of a test chain is rejected. macro_params.seed may only be one of the listed '
              'train_seeds.')


def _test_role(visible):
    return NAMED_TEST_ROLE if NAMED_TEST_ROLE in visible else SEED_TEST_ROLE


def per_chain_access(visible):
    """{train seed or test chain: (set id, largest size allowed or None)} for per-chain
    instances, else None.

    Train chains may be observed at any size; test chains only up to their set's
    first size T1, because their later levels are the prediction targets.
    """
    role = _test_role(visible)
    if role not in visible:
        return None
    first_size = {item['id']: int(item['saturations'][0]) for item in visible['sequences']}
    access = {}
    for group, limit in (('train_seeds', None), (role, first_size)):
        for set_id, members in visible[group].items():
            for member in members:
                ident = member if group == NAMED_TEST_ROLE else int(member)
                if ident in access:
                    raise ValueError(f'Chain {member} is listed twice in the instance')
                access[ident] = (set_id, None if limit is None else limit[set_id])
    return access


class RAFMechanismSequencePolicy(RAFMonotonicSequencePolicy):
    """Request policy for the RAF mechanism sequence task.

    Per-chain instances (``visible.test_seeds`` or ``visible.test_chains``) accept only
    the listed train seeds, the listed test chains and the fixed public macro parameters;
    other instances use the sequence rules of the base policy.
    """

    def capabilities(self, session):
        base = super().capabilities(session)
        visible = session.instance_spec.visible_data
        if per_chain_access(visible) is not None:
            role = _test_role(visible)
            named = role == NAMED_TEST_ROLE
            base['allowed_macro_params'] = [NETWORK_KEY, 'seed'] if named else ['seed']
            base['chain_seed_access'] = {
                item['id']: {'train_seeds': visible['train_seeds'][item['id']],
                             role: visible[role][item['id']],
                             'test_max_size': item['saturations'][0]}
                for item in visible['sequences']}
            if named:
                base['test_networks']['rule'] = NAMED_RULE
        return base

    def _validate_seed_access(self, task_spec, instance_spec, request) -> None:
        access = per_chain_access(instance_spec.visible_data)
        if access is None:
            return super()._validate_seed_access(task_spec, instance_spec, request)
        named = _test_role(instance_spec.visible_data) == NAMED_TEST_ROLE
        fixed = instance_spec.visible_data.get('macro_params') or {}
        extra = set(request.macro_params) - set(fixed) - {'seed'} - ({NETWORK_KEY} if named else set())
        if extra or any(request.macro_params.get(k) != v for k, v in fixed.items()):
            raise ValueError('macro_params are fixed by this instance; only macro_params.seed (a listed train '
                             'seed) or a listed test chain may be chosen; no budget was spent')
        if named:
            # Unknown names, a name with a seed, raw hidden seeds and reserved sample seeds.
            hidden_networks.validate_access(instance_spec, request)
        if named and NETWORK_KEY in request.macro_params:
            chains = {request.macro_params[NETWORK_KEY]}
        else:
            chains = {int(job.macro_params['seed']) for job in SimulationPlanner().plan(request)}
        sizes = [request.saturation] if type(request.saturation) is int else list(request.saturation)
        for chain in chains:
            if chain not in access:
                # Report a guessed hidden seed as such, then reject any other unlisted seed.
                super()._validate_seed_access(task_spec, instance_spec, request)
                listed = 'train_seeds or test_chains' if named else 'train_seeds or test_seeds'
                raise ValueError(f'macro_params.seed={chain} is not a train or test chain of this '
                                 f'instance; use visible {listed}; no budget was spent')
            set_id, limit = access[chain]
            if limit is not None and max(sizes) > limit:
                raise ValueError(f'Test chain {chain} of {set_id} may be observed only at sizes <= '
                                 f'{limit}; its later levels are prediction targets; no budget was spent')
