import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.random import split
from .utils import conn_from_matrix, entropy_sink_patch, get_kernels_fft, sobel, sobel_lax, sobel_lax_legacy, growth
from .reintegration_tracking import ReintegrationTracking
from typing import NamedTuple

class Config(NamedTuple):
    X: int=128
    Y: int=128
    C: int=1
    c0: list[int]=[0]
    c1: list[list[int]]=[[0]]
    k: int=10
    dd: int=5
    dt: float=0.2
    sigma: float=.65
    border: str="wall"
    mix_rule: str="stoch"

def inv_sigmoid(x):
    return jnp.log(x) - jnp.log1p(-x)


class FlowLenia:
    def __init__(
        self,
        grid_size: int = 128,
        C: int = 1,
        k: int = 10,
        kernel_components: int = 3,
        M: jnp.ndarray = jnp.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]]),
        dd: int = 5,
        dt: float = 0.2,
        sigma: float = 0.65,
        border: str = "wall",
        mix_rule: str = "stoch",
        sobel_impl: str = "scipy",
        base_seed: int = 0,
        seed_patch_size: int = 20,
        seed_n_patches: int = 1,
        seed_mode: str = "notebook_centers",
        p_constant_per_patch: bool = True,
        render_mode: str = "Pcolor",
        clip1: float = float("inf"),
        clip2: float = float("inf"),
        mutation: bool = False,
        mutation_patch_size: int = 20,
        mutation_prob: float = 0.1,
        mutation_scale: float = 1.0,
        optimize_mutation_scale: bool = False,
        volcano: bool = False,
        volcano_patch_size: int = 30,
        volcano_prob: float = 0.01,
        volcano_delta_scale: float = 5.0,
        food_enabled: bool = False,
        food_spawn_interval: int = 128,
        food_n_patches: int = 3,
        food_patch_size: int = 16,
        food_amount: float = 1.0,
        food_consume_rate: float = 0.05,
        food_bonus: float = 1.0,
        mass_decay: float = 0.0,
        food_green_channel: int = 1,
        food_auto_size: bool = False,
        food_auto_scale: float = 1.0,
        food_conv_mode: str = "scalar",
        food_vis_scale: float = 1.0,
        food_vis_color=(0.6, 0.3, 0.0),
        food_diffusion_alpha: float = 0.0,
        mass_clip_eps: float = 0.0,
        mass_renorm: bool = False,
        debug_return_F: bool = False,
    ):
        self.grid_size = grid_size
        self.C = C
        self.k = k
        self.kc = int(kernel_components)
        self.M = M
        self.dd = dd
        self.dt = dt
        self.sigma = sigma
        self.border = border
        self.mix_rule = mix_rule
        self.sobel_impl = sobel_impl
        self.base_seed = int(base_seed)
        self.seed_patch_size = seed_patch_size
        self.seed_n_patches = seed_n_patches
        self.seed_mode = seed_mode
        self.p_constant_per_patch = bool(p_constant_per_patch)
        self.render_mode = render_mode
        self.palette_mode = (self.render_mode == "PcolorMix")
        self.clip1 = clip1
        self.clip2 = clip2
        self.mutation_scale = float(mutation_scale)
        self.optimize_mutation_scale = bool(optimize_mutation_scale)
        self.mutation_scale_log_bounds = (-5.0, 3.0)
        self.mutation_enabled = bool(mutation)
        self.mutation_sz = int(mutation_patch_size)
        self.mutation_p = float(mutation_prob)
        self.volcano_enabled = bool(volcano)
        self.volcano_sz = int(volcano_patch_size)
        self.volcano_p = float(volcano_prob)
        self.volcano_delta_scale = float(volcano_delta_scale)
        self.food_enabled = bool(food_enabled)
        self.food_spawn_interval = int(food_spawn_interval)
        self.food_n_patches = int(food_n_patches)
        self.food_patch_size = int(food_patch_size)
        self.food_amount = float(food_amount)
        self.food_consume_rate = float(food_consume_rate)
        self.food_bonus = float(food_bonus)
        self.mass_decay = float(mass_decay)
        self.food_green_channel = int(food_green_channel)
        self.food_auto_size = bool(food_auto_size)
        self.food_auto_scale = float(food_auto_scale)
        self.food_conv_mode = food_conv_mode
        self.food_vis_scale = float(food_vis_scale)
        self.food_vis_color = tuple(food_vis_color)
        self.food_diffusion_alpha = float(food_diffusion_alpha)
        self.mass_clip_eps = float(mass_clip_eps)
        self.mass_renorm = bool(mass_renorm)
        self.debug_return_F = bool(debug_return_F)
        c0, c1 = conn_from_matrix(M)
        self.k = len(c0)
        self.cfg = Config(
            X=grid_size,
            Y=grid_size,
            C=C,
            c0=c0,
            c1=c1,
            k=self.k,
            dd=dd,
            dt=dt,
            sigma=sigma,
            border=border,
            mix_rule=mix_rule,
        )
        self.bounds = {
            "R": (2.0, 25.0),
            "r": (0.2, 1.0),
            "m": (0.05, 0.5),
            "s": (0.001, 0.18),
            "a": (0.0, 1.0),
            "b": (0.001, 1.0),
            "w": (0.01, 0.5),
            "fcr": (0.0, 0.5),
        }
        kR, kr, km, ks, ka, kb, kw = jr.split(jr.key(self.base_seed), 7)
        base_R = jr.uniform(kR, (), minval=2.0, maxval=25.0)
        base_r = jr.uniform(kr, (self.k,), minval=0.2, maxval=1.0)
        base_m = jr.uniform(km, (self.k,), minval=0.05, maxval=0.5)
        base_s = jr.uniform(ks, (self.k,), minval=0.001, maxval=0.18)
        base_a = jr.uniform(ka, (self.k, self.kc), minval=0.0, maxval=1.0)
        base_b = jr.uniform(kb, (self.k, self.kc), minval=0.001, maxval=1.0)
        base_w = jr.uniform(kw, (self.k, self.kc), minval=0.01, maxval=0.5)

        def to_raw(x, name):
            lo, hi = self.bounds[name]
            norm = jnp.clip((x - lo) / (hi - lo), 1e-6, 1 - 1e-6)
            return inv_sigmoid(norm)

        raw_R = to_raw(base_R, "R").reshape((-1,))
        raw_r = to_raw(base_r, "r").reshape((-1,))
        raw_m = to_raw(base_m, "m").reshape((-1,))
        raw_s = to_raw(base_s, "s").reshape((-1,))
        raw_a = to_raw(base_a, "a").reshape((-1,))
        raw_b = to_raw(base_b, "b").reshape((-1,))
        raw_w = to_raw(base_w, "w").reshape((-1,))
        base_fcr = jnp.clip(jnp.array(self.food_consume_rate), self.bounds["fcr"][0], self.bounds["fcr"][1])
        raw_fcr = inv_sigmoid(jnp.clip((base_fcr - self.bounds["fcr"][0])/(self.bounds["fcr"][1]-self.bounds["fcr"][0]), 1e-6, 1-1e-6)).reshape((-1,))

        self.base_dyn_raw = jnp.concatenate([raw_R, raw_r, raw_m, raw_s, raw_a, raw_b, raw_w, raw_fcr], axis=0)
        self._dyn_lo = jnp.concatenate([
            jnp.full((1,), self.bounds["R"][0]),
            jnp.full((self.k,), self.bounds["r"][0]),
            jnp.full((self.k,), self.bounds["m"][0]),
            jnp.full((self.k,), self.bounds["s"][0]),
            jnp.full((self.k * self.kc,), self.bounds["a"][0]),
            jnp.full((self.k * self.kc,), self.bounds["b"][0]),
            jnp.full((self.k * self.kc,), self.bounds["w"][0]),
            jnp.full((1,), self.bounds["fcr"][0]),
        ], axis=0)
        self._dyn_hi = jnp.concatenate([
            jnp.full((1,), self.bounds["R"][1]),
            jnp.full((self.k,), self.bounds["r"][1]),
            jnp.full((self.k,), self.bounds["m"][1]),
            jnp.full((self.k,), self.bounds["s"][1]),
            jnp.full((self.k * self.kc,), self.bounds["a"][1]),
            jnp.full((self.k * self.kc,), self.bounds["b"][1]),
            jnp.full((self.k * self.kc,), self.bounds["w"][1]),
            jnp.full((1,), self.bounds["fcr"][1]),
        ], axis=0)

    def default_params(self, rng):
        n_dyn = self.base_dyn_raw.size
        rng, k1 = jr.split(rng)
        parts = [jr.normal(k1, (n_dyn,)) * 0.1]
        if self.palette_mode:
            rng, k2 = jr.split(rng)
            parts.append(jr.normal(k2, (self.k * 3,)) * 0.1)
        if self.optimize_mutation_scale:
            rng, k3 = jr.split(rng)
            parts.append(jr.normal(k3, (1,)) * 0.1)
        return jnp.concatenate(parts, axis=0)

    def seed_state(self, rng, params):
        n_dyn = self.base_dyn_raw.size
        dyn_delta = params[:n_dyn]
        raw_dyn = self.base_dyn_raw + jnp.clip(dyn_delta, -self.clip1, self.clip1)
        norm_dyn = jax.nn.sigmoid(raw_dyn)
        dyn_vals = self._dyn_lo + norm_dyn * (self._dyn_hi - self._dyn_lo)
        idx = 0
        R = dyn_vals[idx]; idx += 1
        r = dyn_vals[idx:idx + self.k]; idx += self.k
        m = dyn_vals[idx:idx + self.k]; idx += self.k
        s = dyn_vals[idx:idx + self.k]; idx += self.k
        a = dyn_vals[idx:idx + self.k * self.kc].reshape((self.k, self.kc)); idx += self.k * self.kc
        b = dyn_vals[idx:idx + self.k * self.kc].reshape((self.k, self.kc)); idx += self.k * self.kc
        w = dyn_vals[idx:idx + self.k * self.kc].reshape((self.k, self.kc)); idx += self.k * self.kc
        fcr = dyn_vals[idx]; idx += 1
        self.RT = ReintegrationTracking(
            SX=self.cfg.X, SY=self.cfg.Y, dt=self.cfg.dt, dd=self.cfg.dd,
            sigma=self.cfg.sigma, border=self.cfg.border, has_hidden=True,
            mix=self.cfg.mix_rule,
        )
        fK = get_kernels_fft(self.cfg.X, self.cfg.Y, self.cfg.k, R, r, a, w, b)
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        P = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.k))
        Food = jnp.zeros((self.cfg.X, self.cfg.Y))
        kA, kP = split(rng)
        sz = int(self.seed_patch_size)
        sz = max(1, min(sz, self.grid_size))
        mode = self.seed_mode
        if mode == "center":
            i0 = self.grid_size // 2 - sz // 2
            j0 = self.grid_size // 2 - sz // 2
            A_patch = jr.uniform(kA, (sz, sz, self.C))
            if self.p_constant_per_patch:
                P_vec = jr.uniform(kP, (1, 1, self.k))
                P_patch = jnp.ones((sz, sz, self.k)) * P_vec
            else:
                P_patch = jr.uniform(kP, (sz, sz, self.k))
            A = jax.lax.dynamic_update_slice(A, A_patch, (i0, j0, 0))
            P = jax.lax.dynamic_update_slice(P, P_patch, (i0, j0, 0))
        elif mode == "notebook_centers":
            g = self.grid_size
            max_i = g - sz
            max_j = g - sz
            r1, r2, rC = 0.27, 0.73, 0.5
            centers = [
                (int(round(g * r1)), int(round(g * r1))),
                (int(round(g * r1)), int(round(g * r2))),
                (int(round(g * r2)), int(round(g * r1))),
                (int(round(g * r2)), int(round(g * r2))),
                (int(round(g * rC)), int(round(g * rC))),
            ]
            i0s = jnp.array([max(0, min(ci - sz // 2, max_i)) for ci, _ in centers], dtype=jnp.int32)
            j0s = jnp.array([max(0, min(cj - sz // 2, max_j)) for _, cj in centers], dtype=jnp.int32)

            def body(t, carry):
                A_cur, P_cur, key_cur = carry
                key_next, kA_t, kP_t = jr.split(key_cur, 3)
                i0 = i0s[t]
                j0 = j0s[t]
                A_patch = jr.uniform(kA_t, (sz, sz, self.C))
                if self.p_constant_per_patch:
                    P_vec = jr.uniform(kP_t, (1, 1, self.k))
                    P_patch = jnp.ones((sz, sz, self.k)) * P_vec
                else:
                    P_patch = jr.uniform(kP_t, (sz, sz, self.k))
                A_cur = jax.lax.dynamic_update_slice(A_cur, A_patch, (i0, j0, 0))
                P_cur = jax.lax.dynamic_update_slice(P_cur, P_patch, (i0, j0, 0))
                return (A_cur, P_cur, key_next)

            A, P, _ = jax.lax.fori_loop(0, 5, body, (A, P, kA))
        else:
            n_target = max(1, int(self.seed_n_patches))
            max_i = self.grid_size - sz
            max_j = self.grid_size - sz
            key_loop = kA

            def body(t, carry):
                A_cur, P_cur, key_cur = carry
                key_next, kA_t, kP_t, ki, kj = jr.split(key_cur, 5)
                i0 = jr.randint(ki, (), 0, max_i + 1)
                j0 = jr.randint(kj, (), 0, max_j + 1)
                A_patch = jr.uniform(kA_t, (sz, sz, self.C))
                if self.p_constant_per_patch:
                    P_vec = jr.uniform(kP_t, (1, 1, self.k))
                    P_patch = jnp.ones((sz, sz, self.k)) * P_vec
                else:
                    P_patch = jr.uniform(kP_t, (sz, sz, self.k))
                A_cur = jax.lax.dynamic_update_slice(A_cur, A_patch, (i0, j0, 0))
                P_cur = jax.lax.dynamic_update_slice(P_cur, P_patch, (i0, j0, 0))
                return (A_cur, P_cur, key_next)

            A, P, _ = jax.lax.fori_loop(0, n_target, body, (A, P, key_loop))
        t = jnp.array(0, dtype=jnp.int32)
        state = {"A": A, "P": P, "fK": fK, "m": m, "s": s, "fcr": fcr, "Food": Food, "t": t}
        if self.debug_return_F:
            state["F"] = jnp.zeros((self.cfg.X, self.cfg.Y, 2, self.cfg.C), dtype=A.dtype)
        return state

    def init_state(self, rng, params):
        state = self.seed_state(rng, params)
        return self.step_state(rng, state, params)

    def step_state(self, rng, state, params):
        A, P, fK = state["A"], state["P"], state["fK"]
        m, s = state["m"], state["s"]
        Food = state.get("Food", jnp.zeros(A.shape[:2]))
        t = state.get("t", jnp.array(0, dtype=jnp.int32))
        fcr = state.get("fcr", jnp.array(self.food_consume_rate))
        fA = jnp.fft.fft2(A, axes=(0, 1))
        fAk = fA[:, :, self.cfg.c0]
        U = jnp.real(jnp.fft.ifft2(fK * fAk, axes=(0, 1)))
        U = growth(U, m, s) * P
        U = jnp.dstack([U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C)])
        if self.sobel_impl == "lax":
            sobel_fn = sobel_lax
        elif self.sobel_impl == "lax_legacy":
            sobel_fn = sobel_lax_legacy
        else:
            sobel_fn = sobel
        F = sobel_fn(U)
        C_grad = sobel_fn(A.sum(axis=-1, keepdims=True))
        alpha = jnp.clip((A[:, :, None, :] / 2) ** 2, 0.0, 1.0)
        mag = self.cfg.dd - self.cfg.sigma
        F = jnp.clip(F * (1 - alpha) - C_grad * alpha, -mag, mag)
        nA, nP = self.RT(A, P, F)
        if self.mass_renorm:
            mass_prev = jnp.sum(A, axis=(0, 1), keepdims=True)
            mass_next = jnp.sum(nA, axis=(0, 1), keepdims=True)
            scale = jnp.where(mass_next > 0.0, mass_prev / (mass_next + 1e-12), 1.0)
            nA = nA * scale
        if self.mass_clip_eps > 0:
            msum = jnp.sum(nA, axis=-1, keepdims=True)
            nA = jnp.where(msum < self.mass_clip_eps, 0.0, nA)
        if self.mutation_enabled:
            mut_scale = self.mutation_scale
            if self.optimize_mutation_scale and params is not None:
                n_dyn = self.base_dyn_raw.size
                extra = self.k * 3 if self.palette_mode else 0
                idx = n_dyn + extra
                if params.shape[0] > idx:
                    log_ms = jnp.clip(params[idx], self.mutation_scale_log_bounds[0], self.mutation_scale_log_bounds[1])
                    mut_scale = jnp.exp(log_ms)
            kmut, kpos, kprob = jr.split(rng, 3)
            sz = max(1, min(self.mutation_sz, nP.shape[0], nP.shape[1]))
            kdim = nP.shape[-1]
            mut = jnp.ones((sz, sz, kdim)) * jr.normal(kmut, (1, 1, kdim)) * mut_scale
            max_i = nP.shape[0] - sz
            max_j = nP.shape[1] - sz
            ki, kj = jr.split(kpos)
            i0 = jr.randint(ki, (), 0, max_i + 1)
            j0 = jr.randint(kj, (), 0, max_j + 1)
            dP = jax.lax.dynamic_update_slice(jnp.zeros_like(nP), mut, (i0, j0, 0))
            msk = (jr.uniform(kprob, ()) < self.mutation_p).astype(nP.dtype)
            nP = nP + dP * msk
        if self.volcano_enabled:
            kv_main, kv_key = jr.split(rng)
            def volcano_apply(carry):
                nA_cur, nP_cur = carry
                kpos, kmass, kgen = jr.split(kv_key, 3)
                sz = max(1, min(self.volcano_sz, nA_cur.shape[0], nA_cur.shape[1]))
                max_i = nA_cur.shape[0] - sz
                max_j = nA_cur.shape[1] - sz
                ki, kj = jr.split(kpos)
                i0 = jr.randint(ki, (), 0, max_i + 1)
                j0 = jr.randint(kj, (), 0, max_j + 1)
                A_patch = jax.lax.dynamic_slice(nA_cur, (i0, j0, 0), (sz, sz, nA_cur.shape[-1]))
                removed = jnp.sum(A_patch, axis=(0, 1))
                noise = jnp.abs(jr.normal(kmass, (sz, sz, nA_cur.shape[-1]))) + 1e-8
                norm = jnp.sum(noise, axis=(0, 1), keepdims=True)
                new_patch = noise / norm * removed
                nA_cur = jax.lax.dynamic_update_slice(nA_cur, new_patch, (i0, j0, 0))
                K_levels = jnp.maximum(2.0, jnp.ceil(self.volcano_delta_scale))
                dither = 0.05
                P_patch = jax.lax.dynamic_slice(nP_cur, (i0, j0, 0), (sz, sz, nP_cur.shape[-1]))
                P_quant = entropy_sink_patch(P_patch, K_levels, dither, kgen, per_channel=True)
                nP_cur = jax.lax.dynamic_update_slice(nP_cur, P_quant, (i0, j0, 0))
                return (nA_cur, nP_cur)
            do_volcano = (jr.uniform(kv_main, ()) < self.volcano_p)
            nA, nP = jax.lax.cond(do_volcano, volcano_apply, lambda carry: carry, (nA, nP))
        mass_cycle_start = state.get("mass_cycle_start", jnp.sum(nA))
        if self.food_enabled or (self.mass_decay > 0):
            if self.mass_decay > 0:
                nA = nA * (1.0 - self.mass_decay)
            if self.food_enabled:
                mass_cycle_start = state.get("mass_cycle_start", jnp.sum(nA))
                mass_cur = jnp.sum(nA)
                first_spawn = (t == 0)
                periodic_spawn = (self.food_spawn_interval > 0) & (jnp.mod(t, self.food_spawn_interval) == 0)
                do_spawn = first_spawn | periodic_spawn
                def spawn_food(Food_in, key, required_food):
                    max_sz = min(self.food_patch_size, Food_in.shape[0], Food_in.shape[1])
                    max_i = Food_in.shape[0] - max_sz
                    max_j = Food_in.shape[1] - max_sz
                    fp_sz = jnp.array(self.food_patch_size, dtype=jnp.int32)
                    fp_sz = jnp.clip(fp_sz, 0, max_sz)
                    n = int(self.food_n_patches)
                    cells_per_patch = jnp.maximum(1.0, jnp.float32(fp_sz) * jnp.float32(fp_sz))
                    patches = float(max(1, n))
                    auto_density = required_food / (cells_per_patch * patches + 1e-8)
                    food_amt_cell = jnp.where(
                        self.food_auto_size,
                        jnp.maximum(auto_density, jnp.array(self.food_amount, dtype=jnp.float32)),
                        jnp.array(self.food_amount, dtype=jnp.float32),
                    )
                    def body(i, carry):
                        Fcur, kcur = carry
                        kcur, ki, kj = jr.split(kcur, 3)
                        i0 = jr.randint(ki, (), 0, max_i + 1)
                        j0 = jr.randint(kj, (), 0, max_j + 1)
                        rows = jnp.arange(max_sz) < fp_sz
                        mask = (rows[:, None] & rows[None, :]).astype(jnp.float32)
                        patch = mask * food_amt_cell
                        Fcur = Fcur + jax.lax.dynamic_update_slice(jnp.zeros_like(Fcur), patch, (i0, j0))
                        return (Fcur, kcur)
                    Food_add, _ = jax.lax.fori_loop(0, n, body, (jnp.zeros_like(Food_in), rng))
                    return Food_in + Food_add
                observed_loss = jnp.maximum(0.0, mass_cycle_start - mass_cur)
                required_food = jnp.where(
                    self.food_auto_size,
                    self.food_auto_scale * observed_loss / (self.food_bonus + 1e-8),
                    0.0,
                )
                Food = jax.lax.select(do_spawn, spawn_food(Food, rng, required_food), Food)
                mass_cycle_start = mass_cycle_start
                gc = int(self.food_green_channel)
                gc = max(0, min(gc, int(nA.shape[-1]) - 1))
                A_g = nA[..., gc]
                if self.food_conv_mode == 'conv':
                    rate_field = fcr * Food
                else:
                    rate_field = fcr * A_g
                eat = jnp.minimum(Food, rate_field)
                Food = Food - eat
                nA = nA.at[..., gc].add(eat * self.food_bonus)
                new_mass_cycle_start = jnp.sum(nA)
                mass_cycle_start = jax.lax.select(do_spawn & self.food_auto_size, new_mass_cycle_start, mass_cycle_start)
                if self.food_diffusion_alpha > 0:
                    kernel = jnp.array([[1.0, 2.0, 1.0],
                                        [2.0, 4.0, 2.0],
                                        [1.0, 2.0, 1.0]], dtype=Food.dtype) / 16.0
                    if self.border == 'torus':
                        Food_padded = jnp.pad(Food, 1, mode='wrap')
                    else:
                        Food_padded = jnp.pad(Food, 1, mode='edge')
                    diffused = jsp.signal.convolve2d(Food_padded, kernel, mode='valid', boundary='fill')
                    a = self.food_diffusion_alpha
                    Food = (1.0 - a) * Food + a * diffused
            t = t + jnp.array(1, dtype=jnp.int32)
        out = {"A": nA, "P": nP, "fK": fK, "m": m, "s": s, "Food": Food, "t": t, "mass_cycle_start": mass_cycle_start}
        if self.debug_return_F:
            out["F"] = F
        return out

    def render_state(self, state, params, img_size=None):
        mode = getattr(self, 'render_mode', 'A')
        A = state["A"]
        if mode == 'PcolorMix':
            P = state["P"]
            inten = A.sum(axis=-1, keepdims=True)
            if params is None:
                w = jnp.ones((self.k, 3)) / float(self.k)
            else:
                n_dyn = self.base_dyn_raw.size
                w_raw = params[n_dyn:n_dyn + self.k * 3]
                w_raw = jnp.reshape(w_raw, (3, self.k))
                w_soft = jax.nn.softmax(w_raw, axis=1)
                w = jnp.transpose(w_soft, (1, 0))
            color = jnp.tensordot(P, w, axes=([2], [0]))
            img = jnp.clip(inten * color, 0.0, 1.0)
        elif mode == 'Pcolor':
            P = state["P"]
            inten = A.sum(axis=-1, keepdims=True)
            img = jnp.clip(inten * P[..., :3], 0.0, 1.0)
        else:
            C = A.shape[-1]
            if C == 1:
                img = jnp.dstack([A[..., 0], A[..., 0], A[..., 0]])
            elif C == 2:
                img = jnp.dstack([A[..., 0], A[..., 0], A[..., 1]])
            else:
                img = A[..., :3]
        Food = state.get('Food', None)
        if Food is not None:
            f = Food * self.food_vis_scale
            f = f / (1.0 + f)
            overlay_color = jnp.array(self.food_vis_color, dtype=img.dtype)
            overlay = f[..., None] * overlay_color[None, None, :]
            img = jnp.clip(img + overlay, 0.0, 1.0)

        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method="nearest")
        return img
