"""Fixed-seed parity against the pre-optimization event loop."""
from __future__ import annotations

import random

import pytest

from mini_bang.simulators.raf.factory import create_master_model


def legacy_sample(simulator, seed, snapshot_times):
    rng = random.Random(seed)
    species = simulator.crn.species
    food = simulator.crn.food_set
    reactions = simulator.crn.reaction_dict
    catalysis = simulator.crn.catalysis
    rates = simulator.crn.rates
    counts = {s: (simulator.min_food_conc if s in food else 0) for s in species}
    traces = {s: [] for s in species}
    snapshot_times = sorted(snapshot_times)
    st = 0
    t = 0.0
    events = 0
    while events < simulator.step_limit:
        props, actions = [], []
        for rid in reactions:
            rx = reactions[rid]
            reactants = list(rx["reactants"])
            products = list(rx["products"])
            n_cat = sum(counts.get(c, 0) for c in catalysis[rid])
            effective_rate = rates[rid]["k_unlig"] + rates[rid]["k_lig"] * n_cat
            props.append(simulator._get_propensity(counts, reactants, effective_rate))
            actions.append(("lig", rid))
            props.append(simulator._get_propensity(counts, products, effective_rate))
            actions.append(("cle", rid))
        total_prop = sum(props)
        if total_prop <= 0:
            break
        t += rng.expovariate(total_prop)
        pick = rng.uniform(0, total_prop)
        cum = 0.0
        sel = None
        for idx, pr in enumerate(props):
            cum += pr
            if pick <= cum:
                sel = idx
                break
        if sel is None:
            break
        typ, rid = actions[sel]
        rx = reactions[rid]
        if typ == "lig":
            for r in rx["reactants"]:
                counts[r] = max(0, counts[r] - 1)
            for p in rx["products"]:
                counts[p] = counts.get(p, 0) + 1
        else:
            for p in rx["products"]:
                counts[p] = max(0, counts[p] - 1)
            for r in rx["reactants"]:
                counts[r] = counts.get(r, 0) + 1
        for f in food:
            if counts.get(f, 0) < simulator.min_food_conc:
                counts[f] = simulator.min_food_conc
        events += 1
        if (events + 1e-10) / simulator.step_limit >= snapshot_times[st]:
            for s in species:
                traces[s].append(counts.get(s, 0))
            st += 1
            if st >= len(snapshot_times):
                break
    return traces


@pytest.mark.parametrize("macro_seed", [111, 222, 333])
@pytest.mark.parametrize("sample_seed", [17, 2026103001])
@pytest.mark.parametrize("saturation", [6, 12])
def test_compiled_raf_matches_original_event_loop(macro_seed, sample_seed, saturation):
    simulator = create_master_model(seed=macro_seed, step_limit=1000).get_micro_simulator(saturation)
    snapshots = (0.001, 0.01, 0.1, 0.5, 1.0)
    expected = legacy_sample(simulator, sample_seed, snapshots)
    assert simulator.sample(seed=sample_seed, snapshot_times=snapshots) == expected
