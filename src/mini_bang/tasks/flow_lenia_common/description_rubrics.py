"""Hidden atomic description rubrics; the source annotations stay intact."""
from __future__ import annotations

import json
import math
from pathlib import Path
from mini_bang.rubric_judge.contract import ATOMIC_JUDGE_VERSION, HEALTH_PROFILE

ATOMIC_RUBRIC_VERSION = "atomic-facts"
REGISTRY_ROOT = Path(__file__).parents[1] / "flow_lenia_describe" / "gold" / "atomic_rubrics"


def atomic_event(event, *, version, registry_root=REGISTRY_ROOT, require_approved=True):
    if version != ATOMIC_RUBRIC_VERSION:
        raise ValueError("Unknown description rubric version")
    directory = registry_root / version
    index = json.loads((directory / "index.json").read_text())
    if index.get("rubric_version") != version or index.get("judge_version") != ATOMIC_JUDGE_VERSION:
        raise ValueError("Atomic registry protocol identity mismatch")
    if require_approved and index.get("review", {}).get("status") != "approved":
        raise ValueError("Atomic description rubrics are not approved for release")
    if require_approved and index["review"].get("judge_profile") != HEALTH_PROFILE:
        raise ValueError("Atomic registry approval does not cover the active judge profile")
    matches = [v for v in index["events"] if v["annotation_id"] == event["annotation_id"]]
    if len(matches) != 1:
        raise ValueError("Atomic registry needs exactly one entry for the annotated event")
    entry = matches[0]
    document = json.loads((directory / entry["file"]).read_text())
    if (document.get("rubric_version") != version or document.get("judge_version") != ATOMIC_JUDGE_VERSION
            or document.get("annotation_id") != event["annotation_id"]):
        raise ValueError("Atomic event document identity mismatch")
    if document["source_criteria"] != event["rubric"]:
        raise ValueError("Atomic rubric source does not match the frozen annotation")
    atoms = document["rubric"]
    if not atoms or len({v["id"] for v in atoms}) != len(atoms):
        raise ValueError("Atomic rubric requires distinct facts")
    if any(not isinstance(v.get("text"), str) or not v["text"].strip() for v in atoms):
        raise ValueError("Atomic facts require nonempty propositions")
    for source in event["rubric"]:
        group = [v for v in atoms if v.get("source_id") == source["id"]]
        if not group or any(isinstance(v["weight"], bool) or not isinstance(v["weight"], (int, float))
                            or not math.isfinite(v["weight"]) or v["weight"] <= 0 for v in group):
            raise ValueError("Invalid atomic fact weights")
        if not math.isclose(math.fsum(v["weight"] for v in group), source["weight"], rel_tol=1e-12, abs_tol=0):
            raise ValueError("Atomic weights must preserve each source criterion exactly")
    if {v["source_id"] for v in atoms} != {v["id"] for v in event["rubric"]}:
        raise ValueError("Atomic rubric added an unknown source criterion")
    return {**event, "source_rubric": event["rubric"], "rubric": atoms, "rubric_version": version}
