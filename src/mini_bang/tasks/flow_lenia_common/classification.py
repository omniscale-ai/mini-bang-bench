from __future__ import annotations

import math

from mini_bang.framework.task import ValidationResult
from mini_bang.tasks.flow_lenia_common.annotations import REASONS, gold_intervals, load_pool

LOG_LOSS_METRIC = 'exp_negative_mean_log_loss'
BALANCED_BRIER_METRIC = 'balanced_brier_skill'
# Discriminate only; classification_metrics() refuses it, so presence fails closed.
PER_WINDOW_REASONS_METRIC = 'per_window_brier_reasons'
# Discriminate with joint reason probabilities (discriminate_joint.py); presence refuses it the same way.
JOINT_REASONS_METRIC = 'joint_reason_probabilities_quadratic'


def probability_metric(instance_spec, gold_store):
    """The gold or instance declaration chooses the metric, not the task defaults.

    An instance without a declaration is scored by exp(-mean log loss). Unknown
    declarations fail closed.
    """
    declared = gold_store.get('probability_metric')
    config = getattr(instance_spec, 'config', {}) or {}
    visible_declaration = config.get('probability_metric')
    if declared and visible_declaration and declared != visible_declaration:
        raise ValueError('Visible and hidden probability metric versions disagree')
    metric = declared or visible_declaration or LOG_LOSS_METRIC
    if metric not in {LOG_LOSS_METRIC, BALANCED_BRIER_METRIC, PER_WINDOW_REASONS_METRIC,
                      JOINT_REASONS_METRIC}:
        raise ValueError(f'Unknown probability metric protocol: {metric}')
    return metric


def classification_metrics(labels, predictions, epsilon=1e-6, *, metric=LOG_LOSS_METRIC):
    if metric == BALANCED_BRIER_METRIC:
        predictions = [probability(p) for p in predictions]
    result = binary_metrics(labels, predictions, epsilon)
    if metric == LOG_LOSS_METRIC:
        return result
    if metric != BALANCED_BRIER_METRIC:
        raise ValueError(f'Unknown probability metric protocol: {metric}')
    positive = [probability(p) for label, p in zip(labels, predictions) if label]
    negative = [probability(p) for label, p in zip(labels, predictions) if not label]
    if not positive or not negative:
        raise ValueError('Balanced Brier skill requires both positive and negative cases')
    positive_brier = sum((1-p)**2 for p in positive) / len(positive)
    negative_brier = sum(p*p for p in negative) / len(negative)
    balanced = .5 * positive_brier + .5 * negative_brier
    raw_skill = 1 - balanced / .25
    if len(set(predictions)) == 1:
        # Algebraically identical for a constant; avoid roundoff turning a
        # constant near .5 into a tiny positive skill.
        raw_skill = -4 * (predictions[0] - .5) ** 2
    skill = max(0.0, raw_skill)
    result.update(score=skill, primary_metric=BALANCED_BRIER_METRIC,
                  balanced_brier_score=balanced, positive_brier_score=positive_brier,
                  negative_brier_score=negative_brier, balanced_brier_skill=skill,
                  unclipped_balanced_brier_skill=raw_skill)
    return result


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probabilities must be finite numbers in [0,1]")
    return float(value)


def prediction_map(answer, key, case_ids):
    if not isinstance(answer, dict) or not isinstance(answer.get(key), dict):
        raise ValueError(f"Answer must contain a {key} mapping")
    result = answer[key]
    if set(result) != set(case_ids):
        raise ValueError("Answer must contain exactly every requested case_id")
    return result


def binary_metrics(labels, predictions, epsilon=1e-6):
    if not labels or len(labels) != len(predictions):
        raise ValueError("Classification requires non-empty aligned observations")
    losses = [-math.log(max(epsilon, p if label else 1 - p)) for label, p in zip(labels, predictions)]
    positives = [p for label, p in zip(labels, predictions) if label]
    negatives = [p for label, p in zip(labels, predictions) if not label]
    auc = (sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives)
           / (len(positives) * len(negatives))) if positives and negatives else None
    tp = sum(label and p >= 0.5 for label, p in zip(labels, predictions))
    fp = sum(not label and p >= 0.5 for label, p in zip(labels, predictions))
    fn = sum(label and p < 0.5 for label, p in zip(labels, predictions))
    loss = sum(losses) / len(labels)
    return {"score": math.exp(-loss), "log_loss": loss,
            "brier_score": sum((p - int(label)) ** 2 for label, p in zip(labels, predictions)) / len(labels),
            "auroc": auc, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "accuracy": sum(bool(label) == (p >= 0.5) for label, p in zip(labels, predictions)) / len(labels),
            "n_cases": len(labels), "n_positive": len(positives), "n_negative": len(negatives)}


class PresenceScorer:
    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store):
        pool, cases = load_pool(gold_store), gold_store.get("cases")
        try:
            raw = prediction_map(submission.answer, "predictions", [c["case_id"] for c in cases])
            predictions = [probability(raw[c["case_id"]]) for c in cases]
        except (ValueError, TypeError) as exc:
            return ValidationResult(False, str(exc), {"score": 0.0, "valid_submission": False})
        labels = [bool(gold_intervals(pool, c)) for c in cases]
        metrics = classification_metrics(labels, predictions, task_spec.config["validation"]["epsilon"],
                                         metric=probability_metric(instance_spec, gold_store))
        metrics["per_case"] = [{"case_id": c["case_id"], "trajectory": c["trajectory"], "label": label,
                                "prediction": p} for c, label, p in zip(cases, labels, predictions)]
        metrics["valid_submission"] = True
        return ValidationResult(True, "Valid probabilistic submission; quality is measured continuously", metrics)


class DiscriminateScorer:
    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store):
        pool, cases = load_pool(gold_store), gold_store.get("cases")
        if probability_metric(instance_spec, gold_store) == JOINT_REASONS_METRIC:
            # Joint reason probabilities have their own reader and vocabulary; other metrics continue below.
            from mini_bang.tasks.flow_lenia_common.discriminate_joint import joint_reason_result
            return joint_reason_result(cases, [bool(gold_intervals(pool, c)) for c in cases],
                                       submission.answer, gold_store)
        try:
            raw = prediction_map(submission.answer, "predictions", [c["case_id"] for c in cases])
            predictions, reasons = [], []
            for case in cases:
                item = raw[case["case_id"]]
                if not isinstance(item, dict) or set(item) != {"probability", "reasons"}:
                    raise ValueError("Each prediction requires probability and reasons")
                predictions.append(probability(item["probability"]))
                selected = item["reasons"]
                if (not isinstance(selected, list) or any(not isinstance(r, str) for r in selected)
                        or len(selected) != len(set(selected)) or set(selected) - set(REASONS)):
                    raise ValueError("Reasons must be distinct entries from the published vocabulary")
                reasons.append(set(selected))
        except (ValueError, TypeError, KeyError) as exc:
            return ValidationResult(False, str(exc), {"score": 0.0, "valid_submission": False})
        labels = [bool(gold_intervals(pool, c)) for c in cases]
        metric = probability_metric(instance_spec, gold_store)
        per_window = metric == PER_WINDOW_REASONS_METRIC
        # The per-window protocol keeps the balanced Brier skill as a diagnostic.
        metrics = classification_metrics(labels, predictions, metric=BALANCED_BRIER_METRIC if per_window else metric)
        for difficulty in ("easy_negative", "hard_negative", "positive"):
            subset = [i for i, c in enumerate(cases) if c["difficulty"] == difficulty]
            detail = binary_metrics([labels[i] for i in subset], [predictions[i] for i in subset]) if subset else None
            if detail and metric != LOG_LOSS_METRIC:
                # Single-class groups cannot define the balanced primary score.
                detail.pop('score')
            metrics[difficulty] = detail
        scores = []
        for c, selected in zip(cases, reasons):
            expected = set(c.get("reasons", []))
            if expected:
                scores.append(2 * len(expected & selected) / (len(expected) + len(selected)))
        metrics["reason_macro_f1"] = sum(scores) / len(scores) if scores else None
        metrics["classification_score"] = metrics["score"]
        if per_window:
            metrics.update(per_window_reason_score(cases, labels, predictions, reasons))
            metrics["valid_submission"] = True
            return ValidationResult(True, "Classification and explanations scored together in every window", metrics)
        # Keep classification primary, with an explicit secondary reason component.
        metrics["score"] *= 0.5 + 0.5 * (metrics["reason_macro_f1"] or 0.0)
        if metric == BALANCED_BRIER_METRIC:
            metrics['primary_metric'] = 'balanced_brier_skill_times_reason_factor'
        metrics["valid_submission"] = True
        return ValidationResult(True, "Classification and negative explanations scored separately", metrics)


def per_window_reason_score(cases, labels, predictions, reasons):
    """Per-window product: c = 1 - 4 (p - y)^2 and reason agreement r.

    Negative window: r = 2 |selected & gold| / (|selected| + |gold|). Minibang
    window: r = 1 without reasons, 0 with any. Window score c (0.5 + 0.5 r) if
    c > 0, else c, so reasons never deepen a penalty. Score = max(0, mean).
    """
    rows = []
    for case, label, p, selected in zip(cases, labels, predictions, reasons):
        skill = 1 - 4 * (p - int(label)) ** 2
        if label:
            agreement = 0.0 if selected else 1.0
        else:
            expected = set(case.get("reasons", []))
            if not expected:
                raise ValueError("Per-window reason scoring requires gold reasons on every negative window")
            agreement = 2 * len(expected & selected) / (len(expected) + len(selected))
        window = skill * (0.5 + 0.5 * agreement) if skill > 0 else skill
        rows.append({"case_id": case["case_id"], "label": label, "probability": p, "reasons": sorted(selected),
                     "window_skill": skill, "reason_agreement": agreement, "window_score": window})
    mean = sum(row["window_score"] for row in rows) / len(rows)
    negative = [row["reason_agreement"] for row in rows if not row["label"]]
    return {"score": max(0.0, mean), "unclipped_score": mean, "primary_metric": PER_WINDOW_REASONS_METRIC,
            "mean_window_skill": sum(row["window_skill"] for row in rows) / len(rows),
            "mean_negative_reason_f1": sum(negative) / len(negative) if negative else None,
            "positives_with_reasons": sum(bool(row["label"] and row["reasons"]) for row in rows),
            "per_case": rows}
