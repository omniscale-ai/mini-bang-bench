from __future__ import annotations

import jax.numpy as jnp

from mini_bang.simulators.flow_lenia.substrate import FlowLenia

DEFAULT_SUBSTRATE_CONFIG = {
    "grid_size": 64,
    "C": 3,
    "dd": 5,
    "dt": 0.2,
    "sigma": 0.65,
    "border": "wall",
    "mix_rule": "stoch",
    "M": [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
    "k": 9,
}


def create_flow_lenia_substrate(**kwargs):
    params = dict(DEFAULT_SUBSTRATE_CONFIG)
    params.update(kwargs)
    if not hasattr(params["M"], "shape"):
        params["M"] = jnp.array(params["M"], dtype=int)
    substrate = FlowLenia(**params)
    substrate.name = "lenia_flow"
    substrate.rollout_steps = 512
    return substrate
