"""Discriminate: one probability distribution over 33 outcomes in every window.

Answer per window: p = P(minibang) and, for each reason j, pi_j = P(reason j applies |
no minibang). Outcomes: "minibang", or "no minibang with reason set S" for each of
the 32 subsets S of JOINT_REASONS:

    P(minibang) = p
    P(no minibang, S) = (1 - p) * prod_j (pi_j if j in S else 1 - pi_j)

Window score: quadratic (Brier) score at the true outcome,
QS = 2 P(truth) - sum_z P(z)^2 with sum_z P(z)^2 = p^2 + (1 - p)^2 prod_j (pi_j^2 + (1 - pi_j)^2).
QS lies in [-1, 1]; a perfect answer gets 1. Task score:
min(1, max(0, (mean QS - C) / (1 - C))), where C is the mean QS of the best constant
answer (same p and pi in every window) on the hidden gold. C stays in gold: the best
constant pi reveal reason frequencies.

Answer reading: a list of reasons gives pi = 1 to listed reasons; a mapping gives
probabilities. A reason absent from the list or the mapping has pi = 0, so a list and
a mapping of ones are the same answer. Names outside JOINT_REASONS make the answer invalid.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from mini_bang.framework.task import ValidationResult
from mini_bang.tasks.flow_lenia_common.classification import (
    BALANCED_BRIER_METRIC, JOINT_REASONS_METRIC, classification_metrics, prediction_map, probability,
)

# burn_in is not offered: initial settling lies before every observation window and no gold uses it.
JOINT_REASONS = ("reversible", "recoloring", "limited_scope", "gradual_drift", "no_transition")


def reason_probabilities(value) -> dict[str, float]:
    if isinstance(value, list):
        if (any(not isinstance(r, str) for r in value) or len(value) != len(set(value))
                or set(value) - set(JOINT_REASONS)):
            raise ValueError("A reasons list must hold distinct names from the published vocabulary")
        named = {r: 1.0 for r in value}
    elif isinstance(value, dict):
        if any(not isinstance(r, str) for r in value) or set(value) - set(JOINT_REASONS):
            raise ValueError("Reason probabilities must be keyed by names from the published vocabulary")
        named = {r: probability(v) for r, v in value.items()}
    else:
        raise ValueError("Reasons must be a list of names or a mapping of reason probabilities")
    return {r: named.get(r, 0.0) for r in JOINT_REASONS}


def truth_probability(p, pi, label, gold_reasons) -> float:
    if label:
        return p
    return (1 - p) * math.prod(pi[r] if r in gold_reasons else 1 - pi[r] for r in JOINT_REASONS)


def sum_of_squares(p, pi) -> float:
    return p * p + (1 - p) ** 2 * math.prod(pi[r] ** 2 + (1 - pi[r]) ** 2 for r in JOINT_REASONS)


def window_score(p, pi, label, gold_reasons) -> float:
    return 2 * truth_probability(p, pi, label, gold_reasons) - sum_of_squares(p, pi)


def gold_reason_sets(cases, labels) -> list[frozenset]:
    sets = []
    for case, label in zip(cases, labels):
        reasons = frozenset(case.get("reasons", []))
        if label and reasons:
            raise ValueError("Minibang windows must not carry gold reasons")
        if not label and (not reasons or reasons - set(JOINT_REASONS)):
            raise ValueError("Every negative window needs gold reasons from the published vocabulary")
        sets.append(reasons)
    return sets


def mean_window_score(predictions, reason_probs, labels, gold_sets) -> float:
    scores = [window_score(p, pi, label, gold) for p, pi, label, gold
              in zip(predictions, reason_probs, labels, gold_sets)]
    return sum(scores) / len(scores)


def constant_anchor(gold_store) -> float:
    anchor = gold_store.get("constant_anchor") or {}
    value = anchor.get("best_constant_mean_window_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value >= 1:
        raise ValueError("Joint reason scoring requires the hidden best-constant anchor")
    return float(value)


def joint_reason_result(cases, labels, answer, gold_store) -> ValidationResult:
    try:
        raw = prediction_map(answer, "predictions", [c["case_id"] for c in cases])
        predictions, reason_probs = [], []
        for case in cases:
            item = raw[case["case_id"]]
            if not isinstance(item, dict) or set(item) != {"probability", "reasons"}:
                raise ValueError("Each prediction requires probability and reasons")
            predictions.append(probability(item["probability"]))
            reason_probs.append(reason_probabilities(item["reasons"]))
    except (ValueError, TypeError, KeyError) as exc:
        return ValidationResult(False, str(exc), {"score": 0.0, "valid_submission": False})
    metrics = joint_reason_score(cases, labels, predictions, reason_probs, constant_anchor(gold_store))
    metrics["valid_submission"] = True
    return ValidationResult(True, "Presence and reasons scored as one distribution in every window", metrics)


def joint_reason_score(cases, labels, predictions, reason_probs, anchor) -> dict:
    gold_sets = gold_reason_sets(cases, labels)
    # Presence-only diagnostics (balanced Brier skill, AUROC) use the probability alone.
    metrics = classification_metrics(labels, predictions, metric=BALANCED_BRIER_METRIC)
    rows = []
    for case, label, p, pi, gold in zip(cases, labels, predictions, reason_probs, gold_sets):
        brier = None if label else sum((pi[r] - (r in gold)) ** 2 for r in JOINT_REASONS) / len(JOINT_REASONS)
        rows.append({"case_id": case["case_id"], "label": label, "probability": p, "reason_probabilities": pi,
                     "truth_probability": truth_probability(p, pi, label, gold),
                     "window_score": window_score(p, pi, label, gold), "reason_brier": brier})
    mean = sum(row["window_score"] for row in rows) / len(rows)
    positive = [row["window_score"] for row in rows if row["label"]]
    negative = [row["window_score"] for row in rows if not row["label"]]
    briers = [row["reason_brier"] for row in rows if not row["label"]]
    unclipped = (mean - anchor) / (1 - anchor)
    metrics.update(
        score=min(1.0, max(0.0, unclipped)), unclipped_score=unclipped, primary_metric=JOINT_REASONS_METRIC,
        mean_window_score=mean, best_constant_mean_window_score=anchor,
        mean_window_score_positive=sum(positive) / len(positive) if positive else None,
        mean_window_score_negative=sum(negative) / len(negative) if negative else None,
        mean_negative_reason_brier=sum(briers) / len(briers) if briers else None,
        per_case=rows)
    return metrics


def best_constant_search(labels, gold_sets, *, p_grid=(.1, .3, .5, .7, .9), reason_grid=(.1, .5, .9),
                         rounds=20000, tolerance=1e-15):
    """Best constant answer: coordinate ascent with exact one-dimensional maxima from a start grid.

    For a constant answer the mean score is S = 2 q p + 2 (1 - p) B - p^2 - (1 - p)^2 G, with
    q the minibang share, B = sum over negatives of P(gold set) / n and G = prod_j g_j. S is a
    concave quadratic in each single parameter, so every coordinate step solves its maximum
    exactly and S never decreases. Returns (S, p, pi) of the best start after convergence.
    """
    n = len(labels)
    q = sum(labels) / n
    distinct = sorted({s for s, label in zip(gold_sets, labels) if not label}, key=sorted)
    member = np.array([[r in s for r in JOINT_REASONS] for s in distinct], dtype=float)
    weight = np.array([sum(1 for g, label in zip(gold_sets, labels) if not label and g == s) / n
                       for s in distinct])
    starts = np.array(list(itertools.product(p_grid, *[reason_grid] * len(JOINT_REASONS))))
    p, pi = starts[:, 0].copy(), starts[:, 1:].copy()
    for _ in range(rounds):
        before = np.concatenate([p[:, None], pi], axis=1)
        factors = member[None] * pi[:, None] + (1 - member[None]) * (1 - pi[:, None])
        g = pi ** 2 + (1 - pi) ** 2
        b, big_g = (factors.prod(axis=2) * weight).sum(axis=1), g.prod(axis=1)
        p = np.clip((q - b + big_g) / (1 + big_g), 0, 1)
        for k in range(len(JOINT_REASONS)):
            factors = member[None] * pi[:, None] + (1 - member[None]) * (1 - pi[:, None])
            rest = np.delete(factors, k, axis=2).prod(axis=2) * weight
            b1, b0 = (rest * member[:, k]).sum(axis=1), (rest * (1 - member[:, k])).sum(axis=1)
            g_rest = np.delete(pi ** 2 + (1 - pi) ** 2, k, axis=1).prod(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.clip(.5 + (b1 - b0) / (2 * (1 - p) * g_rest), 0, 1)
            pi[:, k] = np.where(p < 1, step, pi[:, k])
        if np.abs(np.concatenate([p[:, None], pi], axis=1) - before).max() < tolerance:
            break
    best = None
    for start_p, start_pi in zip(p.tolist(), pi.tolist()):
        reasons = dict(zip(JOINT_REASONS, start_pi))
        value = mean_window_score([start_p] * n, [reasons] * n, labels, gold_sets)
        if best is None or value > best[0]:
            best = (value, start_p, reasons)
    return best
