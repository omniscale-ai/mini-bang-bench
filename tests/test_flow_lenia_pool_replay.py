from __future__ import annotations

import numpy as np
import pytest

from mini_bang.simulators.flow_lenia_pool.replay import ReplayCursor


class RandomSubstrate:
    def step_state(self, key, state, params):
        import jax
        return {"A": state["A"] + jax.random.uniform(key, shape=(4,))}


def test_replay_keeps_original_batch_rng_schedule_across_observation_stops():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    substrate = RandomSubstrate()
    params, initial, rng = np.zeros(1), {"A": np.zeros(4, dtype=np.float32)}, jax.random.PRNGKey(19)
    kwargs = dict(start_step=0, batch_size=5, batch_index=3, snapshot_interval=7, jit_microbatch=3)
    direct = ReplayCursor(substrate, params, initial, rng, **kwargs)
    expected = np.asarray(direct.at(21)["A"])
    partitioned = ReplayCursor(substrate, params, initial, rng, **kwargs)
    for step in [1, 2, 4, 6, 7, 9, 13, 14, 20, 21]:
        actual = partitioned.at(step)
    np.testing.assert_array_equal(expected, np.asarray(actual["A"]))
    # Independent reference to the original runner: split by canonical snapshot
    # microbatch, not by the requested observation window or future endpoint.
    state, current = {"A": jnp.zeros(4)}, 0
    while current < 21:
        boundary = (current // 7 + 1) * 7
        count = min(3, boundary - current)
        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, count * 5).reshape(count, 5, 2)[:, 3]
        state = jax.lax.scan(lambda s, k: (substrate.step_state(k, s, params), None), state, keys)[0]
        current += count
    np.testing.assert_array_equal(expected, np.asarray(state["A"]))


def test_replay_uses_original_batch_column():
    jax = pytest.importorskip("jax")
    values = []
    for column in (0, 1):
        cursor = ReplayCursor(RandomSubstrate(), np.zeros(1), {"A": np.zeros(4, dtype=np.float32)},
            jax.random.PRNGKey(12), start_step=0, batch_size=2, batch_index=column, snapshot_interval=100, jit_microbatch=10)
        values.append(np.asarray(cursor.at(5)["A"]))
    assert not np.array_equal(*values)
    with pytest.raises(ValueError, match="backwards"):
        cursor.at(4)
