"""Pure reference authority for the reviewed description route."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from mini_bang.rubric_judge.model_policy import (
    REVIEWED_JUDGE_PROFILE as JUDGE_PROFILE, REVIEWED_JUDGE_VERSION as JUDGE_VERSION)

# The bundle identifier is a data value bound by the hidden gold and the registry directory name.
# Annotation text over interpretation notes and tiered rubric weights.
TIERED_WEIGHTS_BUNDLE_VERSION = "tiered-weights"
REFERENCE_BUNDLE_VERSION = TIERED_WEIGHTS_BUNDLE_VERSION
TIERED_WEIGHT_POLICY = "tiered-shares-with-item-order-guard"
TIER_SHARES = {"S": 0.1, "D": 0.3, "T": 0.6}
# Item-order guard: when equal shares would let a lower tier's item weigh >= a higher tier's item,
# the pair's pooled share is split per item in these ratios (D/T first, then S/D).
TIER_GUARD_RATIOS = {("D", "T"): (1, 2), ("S", "D"): (1, 3)}
SOURCE_PROFILE = "scientific18-runtime3"
REGISTRY_ROOT = Path(__file__).parents[1] / "flow_lenia_describe/gold/reviewed_references"
AUTHORITY_FIELDS = {"id", "reference", "rubric_version", "interpretation_notes", "uncertain_claims", "traps"}


def default_registry_root():
    """Reference bundles of the describe package; under a gold root on the scoring server."""
    from mini_bang.framework.benchmark_artifacts import gold_root, task_gold_dir
    return task_gold_dir(REGISTRY_ROOT.parents[1]) / REGISTRY_ROOT.name if gold_root() else REGISTRY_ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tier_item_weights(total, counts):
    """Per-item weight of each present tier and the guard steps that fired.

    Target shares 10/30/60 % (renormalized over present tiers), equal within a tier. Then the per-item
    order S < D < T is enforced: if a D item would weigh >= a T item, the D and T shares are pooled and
    split per item 1:2; afterwards the same for S and D with 1:3. Event totals stay unchanged.
    """
    present = [tier for tier in TIER_SHARES if counts.get(tier)]
    norm = math.fsum(TIER_SHARES[tier] for tier in present)
    each = {tier: total * TIER_SHARES[tier] / norm / counts[tier] for tier in present}
    fired = []
    for (low, high), (a, b) in TIER_GUARD_RATIOS.items():
        if low in each and high in each and each[low] >= each[high]:
            pooled = each[low] * counts[low] + each[high] * counts[high]
            unit = pooled / (a * counts[low] + b * counts[high])
            each[low], each[high] = a * unit, b * unit
            fired.append(f"{low}/{high}")
    weights = [each[tier] for tier in present]
    if any(left >= right for left, right in zip(weights, weights[1:])):
        raise ValueError("Tier item order S < D < T cannot be met for this event")
    return each, fired


def check_tiered_weights(rubric, ledger):
    """Weights follow tier_item_weights: 10/30/60 % shares with the S < D < T per-item order guard."""
    guard = {"/".join(pair): list(ratio) for pair, ratio in TIER_GUARD_RATIOS.items()}
    if (ledger.get("policy") != TIERED_WEIGHT_POLICY or ledger.get("tier_shares") != TIER_SHARES
            or ledger.get("item_order_guard", {}).get("ratios") != guard):
        raise ValueError("Tiered rubric requires the declared tier weight policy")
    if [(row["id"], row["tier"]) for row in ledger["items"]] != [(row["id"], row.get("tier")) for row in rubric]:
        raise ValueError("Weight ledger tiers differ from the rubric")
    total = math.fsum(item["weight"] for item in rubric)
    counts = {tier: sum(item["tier"] == tier for item in rubric) for tier in TIER_SHARES}
    if sum(counts.values()) != len(rubric) or not math.isclose(total, ledger["proposed_total"], rel_tol=1e-12):
        raise ValueError("Tiered rubric total or tier label differs")
    each, fired = tier_item_weights(total, counts)
    if ledger["item_order_guard"].get("fired") != fired:
        raise ValueError("Weight ledger misreports the item-order guard")
    for item in rubric:
        if not math.isclose(item["weight"], each[item["tier"]], rel_tol=1e-12):
            raise ValueError("Tier weights differ from the tier policy: " + item["id"])


def bundle_index(version=REFERENCE_BUNDLE_VERSION, *, registry_root=None):
    if version != TIERED_WEIGHTS_BUNDLE_VERSION:
        raise ValueError("Unknown reviewed reference bundle")
    directory = (registry_root or default_registry_root()) / version
    index = json.loads((directory / "index.json").read_text())
    if (index["reference_bundle_version"] != version or index["judge_version"] != JUDGE_VERSION
            or index["judge_profile"] != JUDGE_PROFILE):
        raise ValueError("Reviewed reference/judge version or profile differs")
    expected = digest({"events": index["events"], "files_sha256": index["files_sha256"]})
    if index["authority_sha256"] != expected or len(index["events"]) != 22:
        raise ValueError("Reviewed reference authority manifest differs")
    return directory, index


def checked_member(directory, index, name):
    if index.get("member_bindings_file"):
        raise ValueError("Reference bundles do not inherit members")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
        raise ValueError("Reference member must stay inside its explicit bundle")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != index["files_sha256"].get(name):
        raise ValueError("Reviewed reference member changed: " + name)
    return json.loads(body)


def event_authority(annotation_id, *, version=REFERENCE_BUNDLE_VERSION, registry_root=None):
    directory, index = bundle_index(version, registry_root=registry_root)
    matches = [entry for entry in index["events"] if entry["annotation_id"] == annotation_id]
    if len(matches) != 1:
        raise ValueError("A reviewed source event must bind exactly one annotation")
    entry = matches[0]; prefix = entry["directory"] + "/"
    data = checked_member(directory, index, prefix + "authority.json")
    if set(data) != AUTHORITY_FIELDS or data["id"] != entry["event"] or data["rubric_version"] != entry["rubric_version"]:
        raise ValueError("Reference event identity/authority shape differs")
    data.update(rubric=checked_member(directory, index, prefix + "rubric.json"),
        rubric_semantics=checked_member(directory, index, prefix + "semantics.json"),
        rubric_weight_ledger=checked_member(directory, index, prefix + "weight-ledger.json"))
    if version == TIERED_WEIGHTS_BUNDLE_VERSION:
        check_tiered_weights(data["rubric"], data["rubric_weight_ledger"])
    return deepcopy(data), index
