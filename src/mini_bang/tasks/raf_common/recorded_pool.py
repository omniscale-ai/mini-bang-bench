"""Recorded observations of hidden RAF objects, shipped with the task package.

An instance with ``visible.observation_pool`` (a pool name) serves its hidden test
networks, chains or family members from recorded rollouts. Their macro seeds and
hidden parameters stay with the organizer; the pool holds only what an agent may
request through the task API:

- ``<task package>/observation_pools/<name>.json``: the manifest (readable);
- ``<task package>/observation_pools/<name>.npz``: the recorded counts.

Manifest::

    {"format": FORMAT, "name": ..., "snapshot_times": [grid],
     "address": {"group": "network"}              # macro_params.network names an object
              | {"group": "family", "member": "network_index"},
     "groups": {group: {"members": n, "rollouts_by_size": {T: R}, "is_raf": bool}}}

Arrays: ``counts__<group>__<T>`` with shape (members, R, len(grid), T): the count of
species 1..T (numeric order) of rollout r at every grid time, MISSING after a
rollout ended early; ``species__<group>__<T>`` with shape (members, T): 1 for the
species that belong to the network (a food species without a reaction does not);
``labels__<group>`` with shape (members, number of sizes): the RAF label per size in
increasing size order, present only when ``is_raf`` is true.

Rollout r of an object at a size is always the same recorded rollout, observed at any
subset of the grid; a request with ``runs=n`` returns rollouts 0..n-1. The policy
turns an admitted request into a ``recorded_network`` request for the RAF simulator
(simulators/raf/recorded.py), so the response has the form of a live simulation.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from mini_bang.execution.planner import SimulationPlanner
from mini_bang.framework.policy import PublicRequestError
from mini_bang.simulators.raf.recorded import RECORDED_KEY

POOL_FIELD = "observation_pool"
FORMAT = "mini_bang.raf_recorded_observations.v1"
POOL_DIR = "observation_pools"
MISSING = -1
RULE = ("Hidden objects are served from recorded rollouts. Rollout i of an object at a size is always "
        "the same recorded rollout, at every snapshot time: runs=n returns rollouts 0..n-1, and at most "
        "rollouts_by_size of them exist. sample_params.seed and macro_params.seed are not available for "
        "recorded objects; snapshot times are distinct values of the published grid.")
_CACHE: dict[str, tuple[tuple[int, int], "RecordedPool"]] = {}


@dataclass(frozen=True)
class RecordedPool:
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def grid(self) -> list[float]:
        return [float(value) for value in self.manifest["snapshot_times"]]

    @property
    def address(self) -> dict[str, str]:
        return self.manifest["address"]

    @property
    def groups(self) -> dict[str, dict[str, Any]]:
        return self.manifest["groups"]

    def rollouts(self, group: str) -> dict[int, int]:
        return {int(size): int(count) for size, count in self.groups[group]["rollouts_by_size"].items()}

    def counts(self, group: str, size: int) -> np.ndarray:
        return self.arrays[f"counts__{group}__{size}"]

    def species(self, group: str, member: int, size: int) -> list[int]:
        """Indices (0-based) of the species of one network along the last counts axis."""
        return [int(i) for i in np.flatnonzero(self.arrays[f"species__{group}__{size}"][member])]

    def label(self, group: str, member: int, size: int) -> bool | None:
        if not self.groups[group].get("is_raf"):
            return None
        sizes = sorted(self.rollouts(group))
        return bool(self.arrays[f"labels__{group}"][member, sizes.index(size)])


def pool_name(instance_spec) -> str | None:
    name = instance_spec.visible_data.get(POOL_FIELD)
    if name is not None and (not isinstance(name, str) or not name or "/" in name or name.startswith(".")):
        raise ValueError(f"visible.{POOL_FIELD} must be the name of a pool in {POOL_DIR}/")
    return name


def pool_directory(instance_spec) -> Path:
    if instance_spec.config_path is not None:
        return Path(instance_spec.config_path).resolve().parent.parent / POOL_DIR
    return Path(str(files(instance_spec.package))) / POOL_DIR


def load(instance_spec) -> RecordedPool | None:
    """The recorded pool of an instance (cached per file), or None for seed-served instances."""
    name = pool_name(instance_spec)
    if name is None:
        return None
    directory = pool_directory(instance_spec)
    return load_files(directory / f"{name}.json", directory / f"{name}.npz")


def load_files(manifest_path: Path, arrays_path: Path) -> RecordedPool:
    stat = arrays_path.stat()
    key = str(arrays_path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("name") != manifest_path.stem:
        raise ValueError(f"{manifest_path} is not a recorded RAF observation pool")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    pool = RecordedPool(manifest, arrays)
    for group, info in pool.groups.items():
        for size, count in pool.rollouts(group).items():
            shape = pool.counts(group, size).shape
            if shape != (info["members"], count, len(pool.grid), size):
                raise ValueError(f"Recorded counts of {group} at size {size} have shape {shape}")
    _CACHE[key] = (stamp, pool)
    return pool


def allowed_macro_keys(instance_spec) -> set[str]:
    pool = load(instance_spec)
    return set() if pool is None else set(pool.address.values())


def addressed(pool: RecordedPool, request) -> tuple[str, int] | None:
    """(group, member) of a request on a recorded object, None for a network of the agent's choice."""
    group_key, member_key = pool.address["group"], pool.address.get("member")
    macro = request.macro_params
    if group_key not in macro:
        return None
    group = macro[group_key]
    if not isinstance(group, str) or group not in pool.groups:
        raise PublicRequestError(f"Unknown recorded object; use one of {sorted(pool.groups)}; "
                                 "no budget was spent")
    members = pool.groups[group]["members"]
    if member_key is None:
        return group, 0
    member = macro.get(member_key)
    if type(member) is not int or not 0 <= member < members:
        raise PublicRequestError(f"macro_params.{member_key} selects one of the {members} recorded "
                                 f"networks of {group}: an integer 0..{members - 1}; no budget was spent")
    return group, member


def validate(instance_spec, request, fixed_macro: dict[str, Any]) -> None:
    """Reject a request on a recorded object that the recording cannot answer exactly."""
    pool = load(instance_spec)
    target = None if pool is None else addressed(pool, request)
    if target is None:
        return
    group, _ = target
    extra = set(request.macro_params) - set(fixed_macro) - set(pool.address.values())
    if extra or any(request.macro_params.get(key) != value for key, value in fixed_macro.items()):
        raise PublicRequestError("A recorded object takes only its name (and member index) besides the "
                                 "fixed macro parameters; macro_params.seed is not available; "
                                 "no budget was spent")
    if request.seed_mode != "planner_derived" or "seed" in request.sample_params:
        raise PublicRequestError("Recorded objects have fixed rollouts: sample_params.seed is not available; "
                                 "runs=n returns recorded rollouts 0..n-1; no budget was spent")
    available = pool.rollouts(group)
    sizes = [request.saturation] if type(request.saturation) is int else list(request.saturation)
    if type(request.saturation) is not int and not pool.manifest.get("sequences", False):
        raise PublicRequestError(f"{group} is observed one size at a time: saturation is one integer; "
                                 "no budget was spent")
    for size in sizes:
        if size not in available:
            raise PublicRequestError(f"{group} is recorded at sizes {sorted(available)} only; "
                                     "no budget was spent")
        if request.runs > available[size]:
            raise PublicRequestError(f"{group} has {available[size]} recorded rollouts at size {size}; "
                                     f"runs must be at most {available[size]}; no budget was spent")
    grid_indices(pool, request)
    if any(value is not False for value in request.micro_params.values()):
        raise PublicRequestError("Interventions are not recorded; no budget was spent")
    if "is_raf" in request.extras and not pool.groups[group].get("is_raf"):
        raise PublicRequestError("is_raf is not available for this recorded object; no budget was spent")


def grid_indices(pool: RecordedPool, request) -> list[int]:
    times = request.sample_params.get("snapshot_times", [1.0])
    grid = pool.grid
    if (not isinstance(times, list) or not times or any(isinstance(t, bool) or not isinstance(t, (int, float))
                                                        for t in times)
            or any(float(t) not in grid for t in times)
            or any(float(a) >= float(b) for a, b in zip(times, times[1:]))):
        raise PublicRequestError(f"Snapshot times of a recorded object are distinct increasing values of "
                                 f"the published grid {grid}; no budget was spent")
    return [grid.index(float(t)) for t in times]


def _series(values: np.ndarray) -> list[int]:
    out = []
    for value in values.tolist():
        if value == MISSING:
            break
        out.append(int(value))
    return out


def recorded_entries(pool: RecordedPool, group: str, member: int, sizes: list[int], runs: int,
                     indices: list[int]) -> list[dict[str, Any]]:
    entries = []
    for size in sizes:
        block = pool.counts(group, size)[member, :runs][:, indices, :]
        present = pool.species(group, member, size)
        rollouts = [[_series(block[run, :, column]) for column in present] for run in range(runs)]
        entries.append({"size": size, "species": [str(column + 1) for column in present],
                        "is_raf": pool.label(group, member, size), "rollouts": rollouts})
    return entries


def backend_request(instance_spec, request):
    """The request the backend runs: recorded rollouts in place of the hidden network."""
    pool = load(instance_spec)
    target = None if pool is None else addressed(pool, request)
    if target is None:
        return request
    group, member = target
    sizes = [request.saturation] if type(request.saturation) is int else list(request.saturation)
    entries = recorded_entries(pool, group, member, sizes, request.runs, grid_indices(pool, request))
    # The planner derives one micro seed per run from the sample base; the recording maps
    # each to its run. The first base whose seeds are distinct is used.
    for base in itertools.count():
        planned = dataclasses.replace(request, macro_params={},
                                      sample_params={**request.sample_params, "seed": base})
        seeds = [job.sample_params["seed"] for job in SimulationPlanner().plan(planned)]
        if len(set(seeds)) == len(seeds):
            break
    return dataclasses.replace(request, macro_params={RECORDED_KEY: {"run_seeds": seeds, "sizes": entries}},
                               sample_params=planned.sample_params)


def public_payload(request, payload):
    """The payload an agent sees: echoed parameters of the public request."""
    public = copy.deepcopy(payload)
    if isinstance(public, dict):
        for key in ("macro_params", "sample_params"):
            if key in public:
                public[key] = copy.deepcopy(getattr(request, key))
    return public


def capability(instance_spec) -> dict[str, Any]:
    pool = load(instance_spec)
    if pool is None:
        return {}
    objects = {group: {"members": info["members"], "rollouts_by_size": info["rollouts_by_size"],
                       "is_raf_available": bool(info.get("is_raf"))}
               for group, info in pool.groups.items()}
    return {"recorded_observations": {"pool": pool.name, "address_with": pool.address,
                                      "snapshot_times": pool.grid, "objects": objects,
                                      "sequences": bool(pool.manifest.get("sequences", False)),
                                      "rule": RULE}}
