from __future__ import annotations

import numpy as np


class ReplayCursor:
    """Replay the original minibang runner's batch RNG schedule, independent of observations.

    Source protocol: flowlenia_minibang_resume.py splits the batch key once per
    canonical microbatch, then splits that subkey into (steps, original batch,
    key) and selects the trajectory's original batch column. Observation stops
    must slice those existing keys, never change the split count.
    """

    def __init__(self, substrate, params, state, rng, *, start_step, batch_size, batch_index,
                 snapshot_interval, jit_microbatch, compiled_advance=None):
        import jax
        import jax.numpy as jnp

        if not 1 <= batch_size <= 4096 or not 0 <= batch_index < batch_size:
            raise ValueError("Invalid original batch coordinates")
        if not 1 <= jit_microbatch <= 10000 or snapshot_interval <= 0:
            raise ValueError("Invalid original RNG microbatch schedule")
        if start_step < 0 or start_step % snapshot_interval:
            raise ValueError("Restart must be at an original snapshot boundary")
        self.jax, self.jnp = jax, jnp
        self.substrate = substrate
        self.params = jnp.asarray(params)
        self.state = {k: jnp.asarray(v) for k, v in state.items()}
        self.rng = jnp.asarray(rng, dtype=jnp.uint32)
        self.current = start_step
        self.batch_size, self.batch_index = batch_size, batch_index
        self.snapshot_interval, self.jit_microbatch = snapshot_interval, jit_microbatch
        self.pending = None

        def advance(state, keys):
            def body(carry, key):
                return substrate.step_state(key, carry, self.params), None
            return jax.lax.scan(body, state, keys)[0]

        self.advance = compiled_advance if compiled_advance is not None else jax.jit(advance)

    def at(self, target):
        if target < self.current:
            raise ValueError("Replay cursor cannot run backwards")
        while self.current < target:
            if self.pending is None or len(self.pending) == 0:
                boundary = (self.current // self.snapshot_interval + 1) * self.snapshot_interval
                n = min(self.jit_microbatch, boundary - self.current)
                self.rng, subkey = self.jax.random.split(self.rng)
                self.pending = self.jax.random.split(subkey, n * self.batch_size).reshape(n, self.batch_size, 2)[:, self.batch_index]
            count = min(target - self.current, len(self.pending))
            self.state = self.advance(self.state, self.pending[:count])
            self.pending = self.pending[count:]
            self.current += count
        return self.state


def replay_observations(reader, steps, fields):
    import jax
    from mini_bang.simulators.flow_lenia.substrate_factory import create_flow_lenia_substrate

    trajectory = reader.trajectory
    restarts = trajectory.get("restarts", [])
    if not any(r["step"] <= steps[0] for r in restarts):
        raise ValueError("No restart at or before the observation; import resume-capable states")
    if trajectory.get("rng_protocol") != "minibang_batch_v1":
        raise ValueError("Unknown original RNG protocol; refusing a different trajectory")
    if trajectory.get("replay_verified") is not True:
        raise ValueError("Pool replay has not passed anchor verification")
    substrate = create_flow_lenia_substrate(**trajectory["substrate_config"])
    with np.load(reader.materialize(trajectory["params_ref"]), allow_pickle=False) as params_data:
        params = np.asarray(params_data["params"], dtype=np.float32)
    result = {field: [] for field in fields}
    total_bytes = 0
    cursor, previous_restart = None, None
    for target in steps:
        restart = max((r for r in restarts if r["step"] <= target), key=lambda r: r["step"])
        if target - restart["step"] > 1000:
            raise ValueError("Exact pool observations require restart spacing at most 1000 steps")
        if restart["step"] != previous_restart:
            cursor = load_cursor(reader, restart, substrate, params,
                                 compiled_advance=cursor.advance if cursor is not None else None)
            previous_restart = restart["step"]
        state = cursor.at(target)
        for field in fields:
            if field == "rgb":
                frame = np.asarray(jax.device_get(substrate.render_state(state, cursor.params, img_size=224)))
                value = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                if field not in state:
                    raise ValueError("Requested field is unavailable in the restart protocol")
                value = np.asarray(jax.device_get(state[field]))
            total_bytes += value.nbytes
            if total_bytes > 256 * 1024 * 1024:
                raise ValueError("Observation exceeds 256 MiB; request fewer states")
            result[field].append(value)
    return {field: np.stack(values) for field, values in result.items()}


def load_cursor(reader, restart, substrate, params, *, compiled_advance=None):
    # init_state also constructs the substrate's static ReintegrationTracking
    # instance. Its sampled state is discarded; only the exact saved state and
    # original batch RNG below are used for replay.
    if not hasattr(substrate, "RT"):
        import jax
        substrate.init_state(jax.random.PRNGKey(0), params)
    with np.load(reader.materialize(restart["ref"]), allow_pickle=False) as data:
        from mini_bang.simulators.flow_lenia_pool.restart_schema import validate_restart
        validate_restart(data)
        state = {key[7:]: np.asarray(data[key]) for key in data.files if key.startswith("state__")}
        required = {"A", "P", "Food", "fK", "m", "s", "t", "mass_cycle_start"}
        if required - state.keys():
            raise ValueError("Incomplete restart state; parameters alone cannot reproduce annotated events")
        if int(data["step"].item()) != restart["step"]:
            raise ValueError("Restart step differs from manifest")
        return ReplayCursor(substrate, params, state, data["rng"], start_step=restart["step"],
                            batch_size=int(data["batch_size"].item()), batch_index=int(data["batch_index"].item()),
                            snapshot_interval=int(data["snapshot_interval"].item()),
                            jit_microbatch=int(data["jit_microbatch"].item()), compiled_advance=compiled_advance)
