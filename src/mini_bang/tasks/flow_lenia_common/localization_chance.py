"""Chance-adjusted soft F1 for interval localization.

Soft F1 pools all windows: 2 * (sum of IoU over one-to-one matched pairs) /
(number of predicted intervals + number of true intervals). No IoU threshold.

The chance level is the soft F1 of the SAME submission when the true events
are dealt to the windows at random: the events of window j are moved into
window i in window-relative coordinates (anchored at the window start and
scaled by the window length), or, in inside-point mode, anchored at the given
inside point. Window i receives the events of each window j with probability
1/n, so the expectation is computed exactly over all n * n pairs.

    score = (soft F1 - chance soft F1) / (1 - chance soft F1)

An answer that ignores the observations, i.e. the same window-relative
interval set in every window (or the same offsets around the inside point),
scores exactly zero. The oracle scores one; worse than chance is negative.

The primary score is max(0, chance-adjusted soft F1), so it lies in [0, 1];
the unclipped value stays in the diagnostics.

Inside point with edge-aware chance (OFFSET_OR_EDGE_CLIPPED_CHANCE_ADJUSTED_SOFT_F1):
the inside point lies in the middle half of its event and every event lies
inside its window, so answers built from the window edges alone (the whole
window, midpoint to midpoint) beat the offset placement, whose dealt events
may cross the window edges. A second placement keeps each dealt boundary at
the same fraction of [t1, inside_t] or [inside_t, t2]; there, answers built
from the edges and inside_t score about zero, while fixed offsets around
inside_t score above zero. The chance level is the larger of the two expected
soft F1 values, so both kinds of answer that ignore the observations score
about zero or below. Both per-placement values stay in the diagnostics.
"""
from __future__ import annotations

import numpy as np

EVENT_F1_LOCALIZATION_METRIC = "event_f1_iou_0.5"
CHANCE_ADJUSTED_SOFT_F1 = "chance_adjusted_soft_f1"
CLIPPED_CHANCE_ADJUSTED_SOFT_F1 = "chance_adjusted_soft_f1_clipped_at_zero"
OFFSET_OR_EDGE_CLIPPED_CHANCE_ADJUSTED_SOFT_F1 = "chance_adjusted_soft_f1_offset_or_edge_chance_clipped_at_zero"
KNOWN_METRICS = {EVENT_F1_LOCALIZATION_METRIC, CHANCE_ADJUSTED_SOFT_F1, CLIPPED_CHANCE_ADJUSTED_SOFT_F1,
                 OFFSET_OR_EDGE_CLIPPED_CHANCE_ADJUSTED_SOFT_F1}
CLIPPED_METRICS = {CLIPPED_CHANCE_ADJUSTED_SOFT_F1, OFFSET_OR_EDGE_CLIPPED_CHANCE_ADJUSTED_SOFT_F1}


def localization_metric(instance_spec, gold_store):
    """The gold or instance declaration chooses the metric; without one it is event F1."""
    declared = gold_store.get("localization_metric")
    visible = (getattr(instance_spec, "config", {}) or {}).get("localization_metric")
    if declared and visible and declared != visible:
        raise ValueError("Visible and hidden localization metric versions disagree")
    metric = declared or visible or EVENT_F1_LOCALIZATION_METRIC
    if metric not in KNOWN_METRICS:
        raise ValueError(f"Unknown localization metric protocol: {metric}")
    return metric


def interval_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]))


def matched_iou_sum(predicted, expected):
    """Largest total IoU over one-to-one matchings, without any threshold."""
    if not predicted or not expected:
        return 0.0
    from scipy.optimize import linear_sum_assignment

    ious = np.asarray([[interval_iou(a, b) for b in expected] for a in predicted])
    rows, cols = linear_sum_assignment(-ious)
    return float(ious[rows, cols].sum())


def _anchor(case):
    return case["inside_t"] if "inside_t" in case else case["t1"]


def moved_intervals(intervals, source, target):
    """Intervals of window ``source`` placed into window ``target`` at the same relative position."""
    scale = (target["t2"] - target["t1"]) / (source["t2"] - source["t1"])
    shift_from, shift_to = _anchor(source), _anchor(target)
    return [[shift_to + (start - shift_from) * scale, shift_to + (end - shift_from) * scale]
            for start, end in intervals]


def edge_placed_intervals(intervals, source, target):
    """Intervals of window ``source`` placed into window ``target`` between the edges and the inside point.

    A boundary before the inside point keeps its fraction of [t1, inside_t], a boundary after it its
    fraction of [inside_t, t2]. Dealt events so stay inside the target window and contain its inside point.
    """
    def place(step):
        before = step <= source["inside_t"]
        span_from = source["inside_t"] - source["t1"] if before else source["t2"] - source["inside_t"]
        span_to = target["inside_t"] - target["t1"] if before else target["t2"] - target["inside_t"]
        return target["inside_t"] + (step - source["inside_t"]) * (span_to / span_from if span_from else 0.0)

    return [[place(start), place(end)] for start, end in intervals]


OFFSET_PLACEMENT = {"offsets_around_anchor": moved_intervals}
OFFSET_OR_EDGE_PLACEMENTS = {"offsets_around_anchor": moved_intervals,
                             "fractions_between_edges_and_inside_t": edge_placed_intervals}


def placements_for(metric):
    return OFFSET_OR_EDGE_PLACEMENTS if metric == OFFSET_OR_EDGE_CLIPPED_CHANCE_ADJUSTED_SOFT_F1 else OFFSET_PLACEMENT


def chance_adjusted_soft_f1(cases, predicted, expected, placements=None):
    """``predicted``/``expected``: interval lists aligned with ``cases``.

    ``placements`` maps a name to a function (intervals, source case, target case) -> intervals; the
    chance level is the largest expected soft F1 over them. With more than one placement the
    per-placement chance levels are added to the result.
    """
    placements = placements or OFFSET_PLACEMENT
    n_pred, n_true = sum(map(len, predicted)), sum(map(len, expected))
    denominator = n_pred + n_true
    if denominator == 0:
        return {"soft_f1": 1.0, "soft_f1_with_shuffled_gold": 1.0, "chance_adjusted_soft_f1": 0.0}
    observed = sum(matched_iou_sum(p, e) for p, e in zip(predicted, expected))
    chances = {}
    for name, place in placements.items():
        chance = 0.0
        if n_pred:
            for case, prediction in zip(cases, predicted):
                if prediction:
                    chance += sum(matched_iou_sum(prediction, place(events, source, case))
                                  for source, events in zip(cases, expected) if events) / len(cases)
        chances[name] = 2 * chance / denominator
    soft, shuffled = 2 * observed / denominator, max(chances.values())
    adjusted = 0.0 if shuffled >= 1.0 else (soft - shuffled) / (1.0 - shuffled)
    result = {"soft_f1": soft, "soft_f1_with_shuffled_gold": shuffled, "chance_adjusted_soft_f1": adjusted}
    if len(placements) > 1:
        result["soft_f1_with_shuffled_gold_by_placement"] = chances
    return result
