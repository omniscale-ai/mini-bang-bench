"""Network families with a hidden catalysis probability (closure vs size).

An instance lists public family names in ``visible.network_families``; its gold file
holds the server-side values::

    "hidden_families": {name: {"p": catalysis probability}, ...}

Every request names one family with ``macro_params.family``. With an observation
pool (``visible.observation_pool``) it selects one recorded network of the family
with ``macro_params.network_index`` and the hidden values are not used at runtime
(raf_common/recorded_pool.py). Otherwise it may choose ``macro_params.seed`` for
independent networks of that family, and the policy adds the
hidden values only in the backend request (framework/request_views.py): the
agent-visible request, its hash, the session trace and the echoed payload carry the
family name, never the hidden value. Agents tell families apart only by observing
them. The policy reads only this registry from the gold file, never the targets.

Hidden values are not searched for in payloads (unlike the 62-bit seeds of
raf_common/hidden_networks.py): a short number such as p can equal a snapshot time,
and a withheld response would then reveal it. Only ``macro_params`` echoes it, and
that echo is replaced by the public request's values.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

from mini_bang.framework.policy import PublicRequestError

FAMILY_KEY = "family"               # macro_params key of a public family name
PUBLIC_FIELD = "network_families"   # instance visible field: list of public names
REGISTRY_FIELD = "hidden_families"  # gold field: the server-side registry
HIDDEN_KEYS = ("p",)


def public_names(instance_spec) -> list[str] | None:
    names = instance_spec.visible_data.get(PUBLIC_FIELD)
    if names is None:
        return None
    if (not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError(f"visible.{PUBLIC_FIELD} must list at least two unique family names")
    return list(names)


def registry(instance_spec) -> dict[str, dict[str, float]]:
    """Server-side name -> hidden macro values, read from the instance's bound gold file.

    The gold file is small and read on every call, so no cache can outlive a gold change.
    """
    from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
    names = public_names(instance_spec)
    raw = BenchmarkArtifactStore().load_gold(instance_spec).get(REGISTRY_FIELD)
    if (not isinstance(raw, dict) or sorted(raw) != sorted(names)
            or any(not isinstance(values, dict) or sorted(values) != sorted(HIDDEN_KEYS)
                   or any(not isinstance(v, float) or not math.isfinite(v) for v in values.values())
                   for values in raw.values())):
        raise ValueError(f"Gold {REGISTRY_FIELD} must give every family a finite {list(HIDDEN_KEYS)}")
    return copy.deepcopy(raw)


def capability(instance_spec) -> dict[str, Any]:
    from mini_bang.tasks.raf_common import recorded_pool
    names = public_names(instance_spec)
    if names is None:
        return {}
    rule = ("Every request names one family. The families share every public macro parameter "
            "except the catalysis probability p, which is hidden on the server; macro_params.seed "
            "chooses independent networks within the family.")
    if recorded_pool.pool_name(instance_spec) is not None:
        rule = ("Every request names one family and one of its recorded networks "
                "(macro_params.network_index). The families share every public macro parameter except "
                "the catalysis probability p, which is not part of the benchmark.")
    return {"network_families": {"names": names, "address_with": f"macro_params.{FAMILY_KEY}", "rule": rule}}


def validate_access(instance_spec, request) -> None:
    """Reject a missing or unknown family and any hidden key; the messages name public values only."""
    names = public_names(instance_spec)
    macro = request.macro_params
    if names is None:
        if FAMILY_KEY in macro:
            raise PublicRequestError(f"macro_params.{FAMILY_KEY} is not available for this task instance")
        return
    if FAMILY_KEY not in macro:
        raise PublicRequestError(f"Every request names a network family: macro_params.{FAMILY_KEY} is one "
                                 f"of {names}; no budget was spent")
    if not isinstance(macro[FAMILY_KEY], str) or macro[FAMILY_KEY] not in names:
        raise PublicRequestError(f"Unknown network family; use one of {names}; no budget was spent")
    if any(key in macro for key in HIDDEN_KEYS):
        raise PublicRequestError(f"{list(HIDDEN_KEYS)} of a family is hidden and cannot be set; "
                                 "no budget was spent")


def without_family(request):
    """The request as the generic RAF checks see it: the family name is checked here."""
    if FAMILY_KEY not in request.macro_params:
        return request
    macro = {key: value for key, value in request.macro_params.items() if key != FAMILY_KEY}
    return dataclasses.replace(request, macro_params=macro)


def backend_request(instance_spec, request):
    """The request the backend runs: the family name replaced by its hidden values."""
    if FAMILY_KEY not in request.macro_params:
        return request
    hidden = registry(instance_spec)[request.macro_params[FAMILY_KEY]]
    resolved = without_family(request)
    return dataclasses.replace(resolved, macro_params={**resolved.macro_params, **hidden})


def public_payload(request, payload):
    """The payload an agent sees: echoed macro parameters of the public request."""
    if FAMILY_KEY not in request.macro_params:
        return payload
    public = copy.deepcopy(payload)
    if isinstance(public, dict) and "macro_params" in public:
        public["macro_params"] = dict(request.macro_params)
    return public
