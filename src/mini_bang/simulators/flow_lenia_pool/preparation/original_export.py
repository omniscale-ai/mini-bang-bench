"""Convert original minibang APF logs without importing the original project.

No pickle checkpoints are loaded. For lossy original fp16 logs use checkpoint
observation mode; replay conversion requires the originally saved full-precision
arrays, RNG schedule and counters. It never fabricates a missing stochastic state.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np

from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import reviewed_trajectories

RENAMES = {
    "mutations": "mutation", "mutation_sz": "mutation_patch_size", "mutation_p": "mutation_prob",
    "volcano_sz": "volcano_patch_size", "volcano_p": "volcano_prob", "volcano_delta": "volcano_delta_scale",
    "food": "food_enabled", "food_interval": "food_spawn_interval", "food_n": "food_n_patches",
    "food_sz": "food_patch_size", "food_channel": "food_green_channel", "n_seeds": "seed_n_patches",
    "flow_sigma": "sigma"}
KEYS = {
    "grid_size", "C", "k", "kernel_components", "M", "dd", "dt", "sigma", "border", "mix_rule",
    "sobel_impl", "base_seed", "seed_patch_size", "seed_n_patches", "seed_mode", "p_constant_per_patch",
    "render_mode", "clip1", "clip2", "mutation", "mutation_patch_size", "mutation_prob", "mutation_scale",
    "optimize_mutation_scale", "volcano", "volcano_patch_size", "volcano_prob", "volcano_delta_scale",
    "food_enabled", "food_spawn_interval", "food_n_patches", "food_patch_size", "food_amount", "food_consume_rate",
    "food_bonus", "mass_decay", "food_green_channel", "food_auto_size", "food_auto_scale", "food_conv_mode",
    "food_vis_scale", "food_vis_color", "food_diffusion_alpha", "mass_clip_eps", "mass_renorm", "debug_return_F"}


def normalized_config(path: Path):
    import yaml
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Original config must be a resolved YAML mapping")
    raw = dict(value.get("substrate", {}))
    rollout = value.get("rollout", {})
    raw.update(rollout.get("substrate", {}))
    if raw.get("substrate") != "lenia_flow" or "food" not in raw and "food_enabled" not in raw:
        raise ValueError("Export requires a self-contained resolved Flow Lenia config, including food setting")
    result = {RENAMES.get(key, key): item for key, item in raw.items() if RENAMES.get(key, key) in KEYS and item is not None}
    if isinstance(result.get("M"), str):
        result["M"] = [[int(v) for v in row.split(",")] for row in result["M"].split(";")]
    if isinstance(result.get("food_vis_color"), str):
        result["food_vis_color"] = [float(v) for v in result["food_vis_color"].split(",")]
    result["debug_return_F"] = True  # original _make_substrate sets this explicitly
    for key in ("clip1", "clip2"):
        if isinstance(result.get(key), float) and math.isinf(result[key]):
            result[key] = "inf" if result[key] > 0 else "-inf"
    if "${" in json.dumps(result, allow_nan=False):
        raise ValueError("Resolve config interpolation in the source project before exporting")
    return result


def export_original(source: Path, destination: Path, *, mode="checkpoints", restart_stride=1000):
    if destination.exists():
        raise FileExistsError("Export destination must be new")
    if mode not in {"checkpoints", "replay"} or restart_stride not in {100, 200, 500, 1000}:
        raise ValueError("Unsupported export mode or restart stride")
    trajectories = reviewed_trajectories()
    # Validate all required paths BEFORE producing potentially large output.
    for name in trajectories:
        directory = source / name
        if not all((directory / file).is_file() for file in ("config.yaml", "params.npy", "video.mp4", "frame_times.csv")):
            raise ValueError(f"{name}: export needs config.yaml, params.npy, video.mp4 and frame_times.csv")
        if not list((directory / "apf_logs").glob("*.npz")):
            raise ValueError(f"{name}: apf_logs is empty; a parameter checkpoint is not a simulation state")
    destination.mkdir(parents=True)
    summary = []
    for name in trajectories:
        original, target = source / name, destination / name
        target.mkdir()
        config = normalized_config(original / "config.yaml")
        validate_frame_times(original / "frame_times.csv")
        (target / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False))
        params = np.load(original / "params.npy", allow_pickle=False)
        np.savez_compressed(target / "params.npz", params=np.asarray(params, dtype=np.float32))
        shutil.copyfile(original / "video.mp4", target / "video.mp4")
        output_dir = target / ("observations" if mode == "checkpoints" else "restarts")
        output_dir.mkdir()
        seen = set()
        substrate = None
        if mode == "replay":
            from mini_bang.simulators.flow_lenia.substrate_factory import create_flow_lenia_substrate
            substrate = create_flow_lenia_substrate(**config)
        for chunk_index, chunk in enumerate(sorted((original / "apf_logs").glob("*.npz"))):
            with np.load(chunk, allow_pickle=False) as archive:
                times = archive["steps"]
                if (times.ndim != 1 or times.dtype.kind not in "iu" or len(set(times.tolist())) != len(times)
                        or any(int(t) in seen for t in times)):
                    raise ValueError("Original chunks have invalid or duplicated step indices")
                seen.update(int(t) for t in times)
                if not {"A", "P"} <= set(archive.files):
                    raise ValueError("Original observations require A and P arrays")
                arrays = {field: archive[field] for field in ("A", "P", "Food", "F") if field in archive.files}
                if any(len(array) != len(times) for array in arrays.values()):
                    raise ValueError("Original chunk fields and steps are not aligned")
                if "Food" not in arrays:
                    if config["food_enabled"]:
                        raise ValueError("Food was enabled but its state was not saved; cannot reconstruct it")
                    arrays["Food"] = np.zeros(arrays["A"].shape[:3], dtype=np.float32)
                if mode == "checkpoints":
                    # Preserve original precision; do not relabel fp16 as an
                    # exact float32 restart merely by casting it.
                    np.savez_compressed(output_dir / f"chunk_{chunk_index:05d}.npz", steps=times, **arrays)
                else:
                    for index, step in enumerate(times):
                        if int(step) % restart_stride == 0:
                            restart = convert_restart(archive, arrays, index, int(step), substrate, params)
                            np.savez_compressed(output_dir / f"step_{int(step):07d}.npz", **restart)
        if sorted(seen) != list(range(0, 500001, 100)):
            raise ValueError(f"{name}: original observations must cover every 100 steps, 0..500000")
        summary.append({"trajectory": name, "original_states": len(seen), "mode": mode})
        print(f"Exported {name}: {len(seen)} original states, mode={mode}", flush=True)
    return {"normalized_source": str(destination), "trajectories": summary}


def convert_restart(archive, arrays, index, step, substrate, params):
    import jax
    import jax.numpy as jnp
    required = {"resume_batch_rng_key", "resume_batch_size", "resume_batch_index", "resume_snapshot_interval",
                "resume_jit_microbatch", "state_t", "state_mass_cycle_start"}
    if required - set(archive.files):
        raise ValueError("Original logs are not resume-capable; export checkpoint mode or provide full RNG/counters")
    if ("resume_stepper_mode" in archive.files
            and str(archive["resume_stepper_mode"][index]) != "legacy_scan_vmap"):
        raise ValueError("Unsupported original stepper protocol; use original checkpoint observations")
    if any(arrays[field].dtype != np.float32 for field in ("A", "P")):
        raise ValueError("Lossy fp16 observations cannot become exact replay states; use --mode checkpoints")
    state = dict(substrate.init_state(jax.random.PRNGKey(0), jnp.asarray(params, dtype=jnp.float32)))
    # fK, m, s are deterministic in the frozen parameters. Dynamic fields and
    # both counters come exclusively from the original simulation log.
    state.update({field: array[index] for field, array in arrays.items()})
    state["t"], state["mass_cycle_start"] = archive["state_t"][index], archive["state_mass_cycle_start"][index]
    result = {"state__" + field: np.asarray(value) for field, value in state.items()}
    result.update(step=np.array(step), rng=archive["resume_batch_rng_key"][index])
    for key in ("batch_size", "batch_index", "snapshot_interval", "jit_microbatch"):
        result[key] = archive["resume_" + key][index]
    return result


def validate_frame_times(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    if (rows.shape != (5001,) or not np.array_equal(rows["step"], np.arange(5001) * 100)
            or not np.allclose(rows["video_sec"], np.arange(5001) / 30, atol=1e-5)):
        raise ValueError("Video/raw-step mapping differs from the reviewed annotation protocol")
