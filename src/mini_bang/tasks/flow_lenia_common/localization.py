from __future__ import annotations

import numpy as np

from mini_bang.framework.task import ValidationResult
from mini_bang.tasks.flow_lenia_common.annotations import gold_intervals, load_pool, overlap
from mini_bang.tasks.flow_lenia_common.classification import prediction_map
from mini_bang.tasks.flow_lenia_common.localization_chance import (CHANCE_ADJUSTED_SOFT_F1, CLIPPED_METRICS,
    OFFSET_OR_EDGE_CLIPPED_CHANCE_ADJUSTED_SOFT_F1, chance_adjusted_soft_f1, localization_metric, placements_for)

# Inside point with event windows only lists windows with a minibang: every answer is one interval.
# Inside point with all windows lists every window: [] means "no minibang".
INSIDE_POINT_EVENT_WINDOWS = "event_windows_only"
INSIDE_POINT_ALL_WINDOWS = "all_windows"


def inside_point_windows(instance_spec, gold_store):
    """The gold or instance declaration wins; without one only windows with a minibang are listed."""
    declared = gold_store.get("inside_point_windows")
    visible = (getattr(instance_spec, "config", {}) or {}).get("inside_point_windows")
    if declared and visible and declared != visible:
        raise ValueError("Visible and hidden inside-point window protocols disagree")
    value = declared or visible or INSIDE_POINT_EVENT_WINDOWS
    if value not in {INSIDE_POINT_EVENT_WINDOWS, INSIDE_POINT_ALL_WINDOWS}:
        raise ValueError(f"Unknown inside-point window protocol: {value}")
    return value


def validate_intervals(value, case, limit=16, allow_empty=False):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError("Intervals must be a list within the published size limit")
    previous_end = case["t1"]
    for interval in value:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2 or any(type(x) is not int for x in interval):
            raise ValueError("Each interval must contain two integer simulation steps")
        start, end = interval
        if not case["t1"] <= start < end <= case["t2"] or start < previous_end:
            raise ValueError("Intervals must be sorted, non-overlapping and within the target window")
        previous_end = end
    if "inside_t" in case and allow_empty and not value:
        return value  # "No minibang in this window."
    if "inside_t" in case and (len(value) != 1 or not value[0][0] <= case["inside_t"] < value[0][1]):
        raise ValueError("Inside-point cases require one interval containing inside_t"
                         + (" or an empty list" if allow_empty else ""))
    return value


def match_intervals(predicted, expected, threshold=0.5):
    from scipy.optimize import linear_sum_assignment

    if not predicted or not expected:
        return {"quality": 1.0 if not predicted and not expected else 0.0, "matches": [], "tp": 0}
    ious = np.asarray([[overlap(a, b) / (a[1] - a[0] + b[1] - b[0] - overlap(a, b))
                        for b in expected] for a in predicted])
    # Cardinality first at the detection threshold; IoU breaks ties.
    rows, cols = linear_sum_assignment(-((ious >= threshold) * (max(ious.shape) + 1) + ious))
    matches = [(int(i), int(j), float(ious[i, j])) for i, j in zip(rows, cols) if ious[i, j] > 0]
    return {"quality": 2 * sum(iou for _, _, iou in matches) / (len(predicted) + len(expected)),
            "matches": matches, "tp": sum(iou >= threshold for _, _, iou in matches)}


class LocalizationScorer:
    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store):
        pool, cases = load_pool(gold_store), gold_store.get("cases")
        all_windows = inside_point_windows(instance_spec, gold_store) == INSIDE_POINT_ALL_WINDOWS
        try:
            raw = prediction_map(submission.answer, "intervals", [c["case_id"] for c in cases])
            predicted = [validate_intervals(raw[c["case_id"]], c, task_spec.config["validation"]["max_intervals"],
                                            allow_empty=all_windows) for c in cases]
        except (ValueError, TypeError) as exc:
            return ValidationResult(False, str(exc), {"score": 0.0, "valid_submission": False})
        qualities, boundary_errors, duration_errors, details = [], [], [], []
        tp = n_pred = n_true = 0
        for case, pred in zip(cases, predicted):
            true = gold_intervals(pool, case)
            if "inside_t" in case:
                # All-windows mode: a window without a minibang expects an empty answer.
                expected_events = 1 if true or not all_windows else 0
                true = [i for i in true if i[0] <= case["inside_t"] < i[1]]
                if len(true) != expected_events:
                    raise ValueError("Gold inside-point must identify exactly one event")
            matched = match_intervals(pred, true, task_spec.config["validation"]["iou_threshold"])
            qualities.append(matched["quality"])
            tp += matched["tp"]
            n_pred += len(pred)
            n_true += len(true)
            for i, j, _ in matched["matches"]:
                boundary_errors.extend([abs(pred[i][0] - true[j][0]), abs(pred[i][1] - true[j][1])])
                duration_errors.append(abs((pred[i][1] - pred[i][0]) - (true[j][1] - true[j][0])))
            details.append({"case_id": case["case_id"], "trajectory": case["trajectory"],
                            "quality": matched["quality"], "expected": true, "predicted": pred})
        mean_boundary = sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
        metrics = {"score": sum(qualities) / len(qualities), "mean_matched_iou_with_penalties": sum(qualities) / len(qualities),
                   "event_f1": 2 * tp / (n_pred + n_true) if n_pred + n_true else 1.0,
                   "event_precision": tp / n_pred if n_pred else float(n_true == 0),
                   "event_recall": tp / n_true if n_true else float(n_pred == 0),
                   "boundary_mae_steps": mean_boundary,
                   "boundary_mae_seconds": mean_boundary / 3000 if mean_boundary is not None else None,
                   "duration_mae_steps": sum(duration_errors) / len(duration_errors) if duration_errors else None,
                   "matched_boundary_count": len(boundary_errors) // 2, "n_events": n_true,
                   "n_predicted_events": n_pred, "valid_submission": True, "per_case": details}
        positives = [d["quality"] for d in details if d["expected"]]
        metrics["positive_case_quality"] = sum(positives) / len(positives) if positives else None
        metric = localization_metric(instance_spec, gold_store)
        if metric == OFFSET_OR_EDGE_CLIPPED_CHANCE_ADJUSTED_SOFT_F1 and not all("inside_t" in c for c in cases):
            raise ValueError("Edge-aware chance needs an inside point in every case")
        metrics.update(chance_adjusted_soft_f1(cases, predicted, [d["expected"] for d in details],
                                               placements_for(metric)))
        metrics["primary_metric"] = metric
        if metric in CLIPPED_METRICS:
            # Score range [0, 1]: worse than chance scores 0; the unclipped value stays a diagnostic.
            metrics["unclipped_score"] = metrics["chance_adjusted_soft_f1"]
            metrics["score"] = max(0.0, metrics["chance_adjusted_soft_f1"])
            return ValidationResult(True, "Chance-adjusted soft F1 clipped at zero; unclipped value in diagnostics", metrics)
        if metric == CHANCE_ADJUSTED_SOFT_F1:
            # Answers that ignore the observations score exactly zero (see localization_chance).
            metrics["score"] = metrics["chance_adjusted_soft_f1"]
            return ValidationResult(True, "Soft F1 adjusted for the same answer against shuffled gold", metrics)
        # Event F1 protocol: rewards matched events, never the prevalence of empty windows.
        metrics["score"] = metrics["event_f1"]
        return ValidationResult(True, "One-to-one interval matching; extra and missing intervals are penalized", metrics)
