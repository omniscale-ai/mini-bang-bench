from __future__ import annotations

import hashlib
import gzip
import json
import math
import re
from decimal import Decimal, ROUND_HALF_UP
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from mini_bang.artifacts import ArtifactRef, get_default_artifact_resolver

REASONS = ("reversible", "recoloring", "limited_scope", "gradual_drift", "burn_in", "no_transition")
STEPS_PER_SECOND = 3000
LAST_STEP = 500000
# One reviewed pool of 100 golden-set trajectories feeds every Flow Lenia task.
POOL_ID = "golden100"
EVENT_KINDS = {"minibang", "minibang_2"}
NEGATIVE_KINDS = {"negative_1", "negative_minibang", "fake_minibang", "other_transition"}
# "unclear" intervals become labels only through an explicit curation decision.
CURATED_KINDS = {"unclear"}
# Language-specific markers of the human annotation live next to the annotation sources.
VOCABULARY_FILE = ("gold", "annotation_sources", "via-vocabulary.json")


@cache
def via_vocabulary() -> dict[str, Any]:
    return json.loads(files(__package__).joinpath(*VOCABULARY_FILE).read_text(encoding="utf-8"))


@cache
def stabilization_note_pattern() -> re.Pattern[str]:
    """Annotations record the onset of established dynamics in the video note with this marker."""
    return re.compile(via_vocabulary()["stabilization_note_pattern"])


def raw_step(seconds: float) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        raise ValueError("Annotation time must be a finite number")
    return int((Decimal(str(seconds)) * STEPS_PER_SECOND).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def import_via(source: Path, review: dict[str, Any]) -> dict[str, Any]:
    """Import explicitly reviewed trajectories; never infer review from empty labels."""
    body = gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
    via = json.loads(body)
    file_map = via["file"]
    view_map = via["view"]
    name_to_view = {}
    for vid, view in view_map.items():
        fids = view["fid_list"]
        if len(fids) != 1:
            raise ValueError("Only one video per annotation view is supported")
        name = Path(file_map[str(fids[0])]["fname"]).stem
        if name in name_to_view:
            raise ValueError(f"Duplicate trajectory {name}")
        name_to_view[name] = str(vid)
    trajectories = {}
    skipped = []
    for name, reviewed in review["trajectories"].items():
        if reviewed.get("complete") is not True:
            raise ValueError(f"Review completeness must be explicit for {name}")
        vid = name_to_view[name]
        events, negatives, notes = [], [], []
        stabilized = reviewed.get("stabilized_from_step")
        has_stabilization_marker = False
        for annotation_id, item in via["metadata"].items():
            if str(item["vid"]) != vid:
                continue
            times, attrs = item.get("z", []), item.get("av", {})
            noted = stabilization_note(str(attrs.get("3", ""))) if not times else None
            if len(times) == 1 and attrs.get("1") == "stabilization_start" or noted is not None:
                point = raw_step(times[0]) if times else noted
                if has_stabilization_marker or (stabilized is not None and stabilized != point):
                    raise ValueError(f"Duplicate or conflicting stabilization annotation for {name}")
                stabilized, has_stabilization_marker = point, True
                if times:
                    continue
            if not times:
                notes.append(str(attrs.get("3", "")))
                continue
            if len(times) != 2:
                skipped.append({"annotation_id": annotation_id, "reason": "point_marker"})
                continue
            label = str(attrs.get("2", ""))
            if via_vocabulary()["legacy_interval_marker"] in label:
                skipped.append({"annotation_id": annotation_id, "reason": "unreviewed_legacy_interval"})
                continue
            start, end = map(raw_step, times)
            if not 0 <= start < end <= LAST_STEP:
                raise ValueError(f"Invalid interval {annotation_id}: {times}")
            kind = attrs.get("1")
            curated = {annotation_id} & set(review["events"]), {annotation_id} & set(review["negatives"])
            if kind in CURATED_KINDS and sum(map(bool, curated)) == 1:
                kind = "minibang" if curated[0] else "negative_1"
            if kind not in EVENT_KINDS | NEGATIVE_KINDS:
                raise ValueError(f"Unresolved annotation kind {kind!r} in {annotation_id}")
            entry = {"annotation_id": annotation_id, "interval": [start, end], "seconds": times,
                     "description": label}
            if kind in EVENT_KINDS:
                entry.update(review["events"][annotation_id])
                if not entry.get("rubric"):
                    raise ValueError(f"Missing description rubric for {annotation_id}")
                events.append(entry)
            else:
                entry["reasons"] = review["negatives"][annotation_id]
                if not entry["reasons"] or set(entry["reasons"]) - set(REASONS):
                    raise ValueError(f"Invalid negative reasons for {annotation_id}")
                negatives.append(entry)
        events.sort(key=lambda item: item["interval"])
        for left, right in zip(events, events[1:]):
            if left["interval"][1] > right["interval"][0]:
                raise ValueError(f"Overlapping events require review for {name}")
        if len(events) != reviewed["event_count"]:
            raise ValueError(f"Unexpected event count for {name}")
        trajectories[name] = {"complete": True, "last_step": LAST_STEP, "events": events,
                              "negatives": negatives, "note": "\n".join(notes),
                              "stabilized_from_step": stabilized}
        stabilization_start(trajectories[name], require_review=False)
    return {"pool_id": POOL_ID, "gold_version": review["version"], "source_file": source.name,
            "source_sha256": hashlib.sha256(body).hexdigest(), "steps_per_second": STEPS_PER_SECOND,
            "video_fps": 30, "video_step_stride": 100, "video_first_step": 0,
            "time_intervals": "half_open", "curation": "human annotation; derived rubrics and reason mapping",
            "skipped_annotations": skipped, "trajectories": trajectories}


def stabilization_note(text: str) -> int | None:
    values = {raw_step(float(value.replace(",", "."))) for value in stabilization_note_pattern().findall(text)}
    if len(values) > 1:
        raise ValueError(f"Conflicting stabilization values in one note: {sorted(values)}")
    return values.pop() if values else None


def bundled_pool(gold_store=None, *, root=None) -> dict[str, Any]:
    """Read the annotation pool version named by the gold; without one, the bundled pool."""
    # Small bundled scoring labels are trusted package artifacts, like the
    # task's gold/*.json. Do not widen the external source resolver's S3-only
    # authority merely to read an installed package resource.
    if root is None:
        from mini_bang.framework.benchmark_artifacts import gold_root, task_gold_dir
        root = task_gold_dir(Path(__file__).parent) if gold_root() else files(__package__).joinpath("gold")
    original = json.loads(root.joinpath(f"{POOL_ID}.json").read_text())
    version = gold_store.get("annotation_version") if gold_store is not None else None
    if version is None or version == original["gold_version"]:
        return original
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("Invalid explicit bundled annotation version")
    path = root.joinpath("versioned_pools", version + ".json")
    if not path.is_file():
        raise ValueError("No bundled annotation authority for version: " + version)
    reviewed = json.loads(path.read_text())
    if reviewed.get("gold_version") != version:
        raise ValueError("Versioned annotation pool identity differs")
    return reviewed


def load_pool(gold_store=None) -> dict[str, Any]:
    ref = (gold_store.get("pool_ref") if gold_store is not None else None)
    if ref is None:
        return bundled_pool(gold_store)
    resolver = (getattr(gold_store, "artifact_resolver", None) or get_default_artifact_resolver())
    return resolver.load_json(ArtifactRef.from_dict(ref))


def overlap(left, right) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def case_id(trajectory: str, start: int, end: int, mode: str = "window") -> str:
    # Names describe the observation, never its hidden label or event identity.
    simulation = (f"simulation-{int(trajectory.removeprefix('traj_')) + 1:02d}"
                  if trajectory.startswith("traj_") and trajectory[5:].isdigit() else trajectory)
    # Sampling categories (e.g. curated vs easy-negative windows) must not be
    # encoded in public identifiers. The same observation has the same name.
    return f"{simulation}__steps-{start}-{end}"


def stabilization_start(trajectory, *, require_review=True):
    value = trajectory.get("stabilized_from_step")
    if value is None:
        if require_review:
            raise ValueError("Human annotation of stabilized_from_step is required before publishing observations")
        return 0
    if type(value) is not int or not 0 <= value < trajectory["last_step"]:
        raise ValueError("stabilized_from_step must be an integer inside the trajectory")
    if any(event["interval"][0] < value for event in trajectory["events"]):
        raise ValueError("Stabilization marker must not discard a reviewed minibang")
    return value


def stabilization_complete(pool):
    return all(t.get("stabilized_from_step") is not None for t in pool["trajectories"].values())


def make_case(name: str, start: int, end: int, mode="window", padding=2000,
              min_observation_step=0, last_step=LAST_STEP) -> dict[str, Any]:
    if not min_observation_step <= start < end <= last_step:
        raise ValueError("Case must lie entirely in the reviewed observation region")
    return {"case_id": case_id(name, start, end, mode), "trajectory": name, "t1": start, "t2": end,
            "observation_window": [max(min_observation_step, start - padding), min(last_step, end + padding)]}


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {key: case[key] for key in ("case_id", "t1", "t2", "observation_window", "inside_t") if key in case}


def gold_intervals(pool, case) -> list[list[int]]:
    window = [case["t1"], case["t2"]]
    return [[max(window[0], event["interval"][0]), min(window[1], event["interval"][1])]
            for event in pool["trajectories"][case["trajectory"]]["events"]
            if overlap(window, event["interval"]) > 0]
