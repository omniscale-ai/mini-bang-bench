"""Hidden RAF test networks addressed by public names (shared by RAF tasks).

A test network named by its macro seed can be grown with the public simulator in
the ``mini-bang`` package and simulated without budget. An instance therefore
lists public names in ``visible.test_networks``, and its gold file holds the
server-side registry::

    "hidden_networks": {"networks": {name: macro_seed, ...},
                        "reserved_sample_seeds": [gold rollout base seeds]}

Agents address a test network with ``macro_params.network``. The policy maps the
name to its seed only in the backend request (``framework/request_views.py``):
the agent-visible request, its hash, the session trace, budget labels and the
returned payload carry the name, never the seed. A raw macro seed equal to a
hidden one, and a sample seed reserved for hidden gold rollouts, are rejected
before any budget is spent. Networks of the agent's own choice (any other macro
seed) stay open and are paid from the same budget.

The policy reads only this registry from the gold file, never labels or counts.
Instances without ``visible.test_networks`` are unaffected.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

NETWORK_KEY = "network"            # macro_params key of a public network name
PUBLIC_FIELD = "test_networks"     # instance visible field: list of public names
REGISTRY_FIELD = "hidden_networks"  # gold field: the server-side registry
_CACHE: dict[tuple[str, str], "HiddenNetworks"] = {}


@dataclass(frozen=True)
class HiddenNetworks:
    seeds: dict[str, int]
    reserved_sample_seeds: frozenset[int]

    def secret_values(self) -> frozenset[int]:
        return frozenset(self.seeds.values()) | self.reserved_sample_seeds


def public_names(instance_spec) -> list[str] | None:
    """Public test network names of an instance, or None when it uses raw seeds."""
    names = instance_spec.visible_data.get(PUBLIC_FIELD)
    if names is None:
        return None
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError(f"visible.{PUBLIC_FIELD} must list unique network names")
    return list(names)


def registry(instance_spec) -> HiddenNetworks | None:
    """Server-side name -> seed registry, loaded from the instance's bound gold file."""
    names = public_names(instance_spec)
    if names is None:
        return None
    if instance_spec.gold_path is None:
        raise ValueError("Hidden test networks need a gold file with their registry")
    body = instance_spec.gold_path.read_bytes()
    key = (str(instance_spec.gold_path), hashlib.sha256(body).hexdigest())
    if key not in _CACHE:
        from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
        raw = BenchmarkArtifactStore().load_gold(instance_spec).get(REGISTRY_FIELD) or {}
        _CACHE[key] = _parse(raw, names)
    return _CACHE[key]


def _parse(raw: Any, names: list[str]) -> HiddenNetworks:
    networks = raw.get("networks") if isinstance(raw, dict) else None
    reserved = raw.get("reserved_sample_seeds", []) if isinstance(raw, dict) else None
    if (not isinstance(networks, dict) or sorted(networks) != sorted(names)
            or any(type(seed) is not int for seed in networks.values())
            or len(set(networks.values())) != len(networks)
            or not isinstance(reserved, list) or any(type(seed) is not int for seed in reserved)):
        raise ValueError("Gold hidden_networks must map every public name to a distinct integer seed")
    return HiddenNetworks(dict(networks), frozenset(reserved))


def allowed_macro_keys(instance_spec) -> set[str]:
    return {NETWORK_KEY} if public_names(instance_spec) is not None else set()


def capability(instance_spec) -> dict[str, Any]:
    from mini_bang.tasks.raf_common import recorded_pool
    names = public_names(instance_spec)
    if names is None:
        return {}
    rule = ("Test networks are addressed by name; their macro seeds are hidden and a raw "
            "macro seed of a test network is rejected. Any other macro_params.seed is a "
            "network of your own choice, paid from the same budget.")
    if recorded_pool.pool_name(instance_spec) is not None:
        rule = ("Test networks are addressed by name and served from recorded rollouts; their macro "
                "seeds are not part of the benchmark. Any macro_params.seed is a network of your own "
                "choice, paid from the same budget.")
    return {"test_networks": {"names": names, "address_with": f"macro_params.{NETWORK_KEY}", "rule": rule}}


def validate_access(instance_spec, request) -> None:
    """Reject unknown names, a name with a raw seed, hidden raw seeds and gold sample seeds.

    Instances with recorded observations (raf_common/recorded_pool.py) have no seed
    registry: only the names are checked here, the recording checks the rest.
    """
    from mini_bang.tasks.raf_common import recorded_pool
    macro = request.macro_params
    names = public_names(instance_spec)
    if NETWORK_KEY in macro:
        if names is None:
            raise ValueError(f"macro_params.{NETWORK_KEY} is not available for this task instance")
        if not isinstance(macro[NETWORK_KEY], str) or macro[NETWORK_KEY] not in names:
            raise ValueError(f"Unknown test network; use one of {sorted(names)}; "
                             "no budget was spent")
        if "seed" in macro:
            raise ValueError("Give macro_params.network or macro_params.seed, not both; "
                             "no budget was spent")
    if names is None or recorded_pool.pool_name(instance_spec) is not None:
        return
    hidden = registry(instance_spec)
    if NETWORK_KEY not in macro:
        from mini_bang.execution.planner import SimulationPlanner
        effective = SimulationPlanner().plan(dataclasses.replace(request, runs=1))[0].macro_params["seed"]
        if type(effective) is int and effective in hidden.seeds.values():
            raise ValueError("This macro seed belongs to a hidden test network; address test networks "
                             "by name; no budget was spent")
    sample = request.sample_params.get("seed")
    values = sample if isinstance(sample, list) else [sample]
    if any(type(value) is int and value in hidden.reserved_sample_seeds for value in values):
        raise ValueError("sample_params.seed is reserved for hidden evaluation; no budget was spent")


def backend_request(instance_spec, request):
    """The request the backend runs: the public name replaced by its hidden seed."""
    if NETWORK_KEY not in request.macro_params:
        return request
    hidden = registry(instance_spec)
    macro = {key: value for key, value in request.macro_params.items() if key != NETWORK_KEY}
    macro["seed"] = hidden.seeds[request.macro_params[NETWORK_KEY]]
    return dataclasses.replace(request, macro_params=macro)


def public_payload(instance_spec, request, payload):
    """The payload an agent sees: echoed macro parameters of the public request."""
    if NETWORK_KEY not in request.macro_params:
        return payload
    public = copy.deepcopy(payload)
    if isinstance(public, dict) and "macro_params" in public:
        public["macro_params"] = dict(request.macro_params)
    assert_public(public, registry(instance_spec))
    return public


def assert_public(value: Any, hidden: HiddenNetworks) -> None:
    """Fail closed if a hidden seed would reach the agent, as a number or inside text."""
    secret = hidden.secret_values()
    texts = [str(seed) for seed in secret]
    for item in _leaves(value):
        if isinstance(item, bool) or item is None:
            continue
        if isinstance(item, (int, float)) and item in secret:
            raise RuntimeError("A hidden network value would reach the agent; the response was withheld")
        if isinstance(item, str) and any(text in item for text in texts):
            raise RuntimeError("A hidden network value would reach the agent; the response was withheld")


def _leaves(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        yield value
    else:
        yield json.dumps(value, default=str)
