"""Population fate per chain: per-chain quantile forecast, scored between two anchors.

Gold holds, per set, per test chain, per later level (size)
and per target species, the log1p-EMA targets of independent micro rollouts on
that chain's actual network. Pinball loss is averaged over rollouts and
quantiles, then species, then levels, then chains (each mean has equal weights),
giving one loss L per set. Each chain is compared only with its own rollouts.

Per set: score = min(1, max(0, 1 - (L - L1) / (L0 - L1))); the unclipped value is
reported in the diagnostics. L0 is the lowest gold loss of
the answers that use no information about the test chain's later growth
(population quantiles; copying level 1), L1 is the gold loss of the
information limit (growth continued from the exact state at T1 with fresh
randomness). Both anchors are fixed in gold. The final score is the mean over
sets. There is no pass threshold: as in the Flow Lenia probability tasks,
``success`` only means a valid, scored submission.
"""
from __future__ import annotations

import numpy as np

from mini_bang.framework.task import ValidationResult

SCORE = 'score'


def is_per_chain_gold(gold_store) -> bool:
    # Per-chain gold has 'sets'; test chains are keyed by public name or by seed.
    return 'sets' in gold_store.data


def _invalid(message):
    return ValidationResult(False, message, {SCORE: 0.0, 'valid_submission': False})


def pinball(values, prediction, quantile):
    delta = np.asarray(values, dtype=np.float64) - prediction
    return np.where(delta >= 0, quantile * delta, (quantile - 1.0) * delta)


def _species_predictions(raw, where):
    from mini_bang.tasks.raf_mechanism_seq.scorer import _coerce_quantile_predictions
    if not isinstance(raw, dict):
        raise ValueError(f'{where} must be a mapping of species to quantile mappings')
    try:
        return _coerce_quantile_predictions(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{where}: {exc}') from exc


def set_loss(predictions, chains, quantiles, *, set_id='set'):
    """Mean pinball loss of one set; returns (loss, per-level mean losses)."""
    if not isinstance(predictions, dict):
        raise ValueError(f'Predictions for {set_id} must be a mapping of test chains')
    predictions = {str(key): value for key, value in predictions.items()}
    chain_losses, level_losses = [], {}
    for chain, levels in chains.items():
        chain_answer = predictions.get(chain)
        if not isinstance(chain_answer, dict):
            raise ValueError(f'Missing predictions for test chain {chain} of {set_id}')
        chain_answer = {str(key): value for key, value in chain_answer.items()}
        per_level = []
        for size, species_values in levels.items():
            where = f'{set_id} chain {chain} size {size}'
            answer = _species_predictions(chain_answer.get(size), where)
            per_species = []
            for species, values in species_values.items():
                quantile_map = answer.get(species, {})
                missing = [q for q in quantiles if q not in quantile_map]
                if missing:
                    raise ValueError(f'Missing q={missing[0]:g} for species {species} in {where}')
                per_species.append(np.mean([pinball(values, quantile_map[q], q).mean()
                                            for q in quantiles]))
            level = float(np.mean(per_species))
            per_level.append(level)
            level_losses.setdefault(size, []).append(level)
        chain_losses.append(float(np.mean(per_level)))
    return float(np.mean(chain_losses)), {size: float(np.mean(v)) for size, v in level_losses.items()}


def scaled_score(loss, anchors):
    """Anchor-scaled score of one set clipped to [0, 1], and its unclipped value."""
    unclipped = 1.0 - (loss - anchors['L1']) / (anchors['L0'] - anchors['L1'])
    return min(1.0, max(0.0, unclipped)), unclipped


def score_answer(predictions, gold, quantiles):
    """Metrics for a mapping {set: {chain: {size: {species: {q: value}}}}}; raises ValueError."""
    if not isinstance(predictions, dict):
        raise ValueError("Submission must contain 'predicted_quantiles' mapping")
    per_set = {}
    for set_id, info in gold['sets'].items():
        anchors = info['anchors']
        if not anchors['L0'] > anchors['L1']:
            raise ValueError(f'Gold anchors of {set_id} have no gap; the instance is invalid')
        loss, levels = set_loss(predictions.get(set_id), info['chains'], quantiles, set_id=set_id)
        score, unclipped = scaled_score(loss, anchors)
        per_set[set_id] = {'mean_pinball_loss': loss, 'per_level_mean_pinball_loss': levels,
                           'L0': anchors['L0'], 'L0_answer': anchors['L0_answer'], 'L1': anchors['L1'],
                           SCORE: score, f'{SCORE}_unclipped': unclipped}
    count = len(per_set)
    return {SCORE: sum(row[SCORE] for row in per_set.values()) / count,
            f'{SCORE}_unclipped': sum(row[f'{SCORE}_unclipped'] for row in per_set.values()) / count,
            'mean_pinball_loss': sum(row['mean_pinball_loss'] for row in per_set.values()) / count,
            'per_set': per_set, 'valid_submission': True}


def score(*, quantiles, submission, gold_store):
    answer = submission.answer
    if not isinstance(answer, dict):
        return _invalid('Submission answer must be a mapping')
    try:
        metrics = score_answer(answer.get('predicted_quantiles'), gold_store.data, quantiles)
    except ValueError as exc:
        return _invalid(str(exc))
    parts = ', '.join(f"{set_id} {row[SCORE]:.3f} (loss {row['mean_pinball_loss']:.4f}, "
                      f"L0 {row['L0']:.4f}, L1 {row['L1']:.4f})" for set_id, row in metrics['per_set'].items())
    return ValidationResult(True, f'Score {metrics[SCORE]:.3f}: {parts}', metrics)
