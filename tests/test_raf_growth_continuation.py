"""RAF growth continuation: seeded chains stay bit-identical; continuation is exact.

``legacy_networks`` is a verbatim copy of the growth loop of ``MasterModel._build_chain``
before continuation support. It is the reference for the refactor.
"""
import random
from collections import defaultdict
from math import comb

import numpy as np
import pytest

from mini_bang.simulators.raf.alpha import coerce_alpha
from mini_bang.simulators.raf.factory import create_master_model

TASK_PARAMS = {'M0': 2, 'K': 3, 'p': 0.15, 'complexity': 0.5,
               'alpha': {'kind': 'linear_cutoff', 'scale': 0.05, 'cutoff': 3, 'floor': 0.0},
               'k_lig': 1.0, 'k_unlig': 0.05, 'step_limit': 50000}
DENSE_PARAMS = {'M0': 3, 'K': 4, 'p': 0.3, 'alpha': {'values': {'1': 0.2, '2': 0.05, '3': 0.01, '4': 0.004}}}
CHAINS = [[6, 8, 10], [10, 12, 14], [3], [12], [9, 5, 30], [7, 7, 11]]


def legacy_networks(params, seed, Ms):
    M0, K, p, alpha = params['M0'], params['K'], params['p'], coerce_alpha(params['alpha'])
    np_rng, rng = np.random.default_rng(seed), random.Random(seed)
    Ms_sorted = sorted(Ms)
    Mmax = Ms_sorted[-1]
    reactions, catalysis, produced_by, snapshots = [], defaultdict(list), {}, {}
    Mt, next_idx = M0, 0
    while Mt < Mmax:
        Mt_prev = Mt
        for i in range(1, min(K + 1, Mt_prev + 1)):
            r_i = np_rng.poisson(alpha(i) * comb(Mt_prev, i))
            for _ in range(r_i):
                Mt += 1
                new_species = str(Mt)
                reactant_ids = np_rng.choice(Mt_prev, size=i, replace=False) + 1
                rid = len(reactions)
                reactions.append((tuple(sorted(str(k) for k in reactant_ids)), new_species))
                produced_by[new_species] = rid
                for y_int in range(M0 + 1, Mt + 1):
                    y = str(y_int)
                    if y == new_species:
                        if rng.random() < p:
                            catalysis[rid].append(new_species)
                    else:
                        if y in produced_by and rng.random() < p:
                            catalysis[produced_by[y]].append(new_species)
                        if rng.random() < p:
                            catalysis[rid].append(y)
                while next_idx < len(Ms_sorted) and Mt >= Ms_sorted[next_idx]:
                    snapshots[Ms_sorted[next_idx]] = [
                        (sorted(reactions[r][0]), reactions[r][1], sorted(catalysis.get(r, [])))
                        for r in range(Mt - M0)]
                    next_idx += 1
                if Mt >= Mmax:
                    break
            if Mt >= Mmax:
                break
    return [snapshots[m] for m in Ms], np_rng.bit_generator.state, rng.getstate()


def fingerprint(simulator):
    crn = simulator.crn
    return [(sorted(crn.reaction_dict[r]['reactants']), next(iter(crn.reaction_dict[r]['products'])),
             sorted(crn.catalysis[r])) for r in sorted(crn.reaction_dict)]


def as_sets(network):
    return [(sorted(set(reactants)), product, sorted(set(cats))) for reactants, product, cats in network]


@pytest.mark.parametrize('params', [TASK_PARAMS, DENSE_PARAMS])
@pytest.mark.parametrize('sizes', CHAINS)
def test_seeded_chains_are_bit_identical_to_the_legacy_growth(params, sizes):
    sizes = [max(m, params['M0'] + 1) for m in sizes]
    for seed in list(range(1, 41)) + [10001, 96001, 106001, 2 ** 31 - 1]:
        model = create_master_model(**params, seed=seed)
        expected, np_state, py_state = legacy_networks(params, seed, sizes)
        simulators = model.get_micro_simulator(sizes)
        assert [fingerprint(s) for s in simulators] == [as_sets(n) for n in expected]
        assert model._np_rng.bit_generator.state == np_state and model._rng.getstate() == py_state


def test_scalar_and_sequence_requests_keep_identical_final_topology():
    for seed in range(1, 21):
        chain = create_master_model(**TASK_PARAMS, seed=seed).get_micro_simulator([10, 12, 14])
        scalar = create_master_model(**TASK_PARAMS, seed=seed).get_micro_simulator(14)
        assert fingerprint(chain[-1]) == fingerprint(scalar)


@pytest.mark.parametrize('first,later', [(6, [8, 10]), (10, [12, 14]), (3, [4, 9])])
def test_continuing_with_the_original_generators_reproduces_the_seeded_chain(first, later):
    for seed in range(1, 31):
        model = create_master_model(**TASK_PARAMS, seed=seed)
        state = model.growth_state_at(first)
        assert state.size == first and state.remaining >= 0
        continued = model.continue_growth(state, later, rng_seed=None)
        canonical = create_master_model(**TASK_PARAMS, seed=seed).get_micro_simulator([first] + later)
        assert [fingerprint(s) for s in continued] == [fingerprint(s) for s in canonical[1:]]


def test_fresh_continuations_keep_the_first_level_and_vary_afterwards():
    model = create_master_model(**TASK_PARAMS, seed=96001)
    state = model.growth_state_at(6)
    first = fingerprint(create_master_model(**TASK_PARAMS, seed=96001).get_micro_simulator(6))
    finals = set()
    for rng_seed in range(1, 21):
        level2, level3 = model.continue_growth(state, [8, 10], rng_seed=rng_seed)
        for network, size in ((level2, 8), (level3, 10)):
            reactions = fingerprint(network)
            assert len(reactions) == size - 2
            # Old reactions keep reactants; old catalysts stay, new species may join them.
            for (reactants, product, cats), (old_reactants, old_product, old_cats) in zip(reactions, first):
                assert (reactants, product) == (old_reactants, old_product)
                assert [c for c in cats if int(c) <= 6] == old_cats
        finals.add(repr(fingerprint(level3)))
    assert len(finals) > 10
    again = model.continue_growth(state, [8, 10], rng_seed=7)
    model.continue_growth(state, [8, 10], rng_seed=8)
    assert fingerprint(again[1]) == fingerprint(model.continue_growth(state, [8, 10], rng_seed=7)[1])


def test_continuation_rejects_sizes_not_above_the_state():
    model = create_master_model(**TASK_PARAMS, seed=1)
    state = model.growth_state_at(6)
    with pytest.raises(ValueError, match='exceed the state size'):
        model.continue_growth(state, [6, 8], rng_seed=1)
