"""Public validation probe; the hidden scorer uses this exact implementation."""
from __future__ import annotations

PROBE_PARAMETERS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "fit_intercept": True,
    "intercept_scaling": 1.0,
    "class_weight": None,
    "max_iter": 1000,
    "random_state": 0,
}
PROBE_DESCRIPTION = (
    "StandardScaler fitted only on training examples, followed by "
    "LogisticRegression(C=1.0, penalty='l2', solver='liblinear', "
    "fit_intercept=True, intercept_scaling=1.0, class_weight=None, "
    "max_iter=1000, random_state=0). The same fitted scaler and classifier "
    "evaluate every thinning factor; neither is refitted on test examples."
)
LEAVE_PAIR_OUT_DESCRIPTION = (
    "StandardScaler followed by LogisticRegression(C=1.0, penalty='l2', "
    "solver='liblinear', fit_intercept=True, intercept_scaling=1.0, "
    "class_weight=None, max_iter=1000, random_state=0), evaluated by grouped "
    "leave-pair-out over the hidden windows: for every pair of one hidden window "
    "with a minibang and one without, a fresh probe is fitted on all other "
    "windows (the training windows plus the remaining hidden windows, with their "
    "labels), excluding every window from the same simulation as either window "
    "of the pair, and predicts both windows of the pair. Training windows are "
    "always in the fit and never in pairs. AUROC is the fraction of pairs in "
    "which the window with a minibang gets the higher probability; ties count "
    "0.5."
)
THINNING_DESCRIPTION = (
    "Thinning keeps every 2nd or 4th checkpoint and appends the final checkpoint "
    "if that stride missed it. Both endpoints and the observation duration are "
    "preserved; the final sampling gap may be shorter than the regular stride."
)


def make_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # The published protocol declares L2 regularization. Current and older
    # sklearn default to L2; omitting the removed API keyword preserves the
    # same numerical probe without changing the published parameter contract.
    parameters = dict(PROBE_PARAMETERS)
    if parameters.pop('penalty') != 'l2':
        raise ValueError('The fixed benchmark probe requires L2 regularization')
    return make_pipeline(StandardScaler(), LogisticRegression(**parameters))


def validate_probe(training_features, training_labels, validation_features):
    """Fit on visible training data and predict a caller's held-out examples.

    This function has no access to hidden cases, files, services, or annotations.
    Labels must contain both classes. Split by trajectory when grouping is known.
    """
    import numpy as np

    labels = np.asarray(training_labels)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Training labels must contain both 0 and 1")
    model = make_probe()
    model.fit(training_features, labels)
    return model.predict_proba(validation_features)[:, 1]


def leave_pair_out(features, labels, groups, *, in_pairs=None, also_predict=()):
    """Grouped leave-pair-out with the fixed probe; the hidden scorer calls this.

    Every (positive, negative) pair of windows marked by in_pairs (default: all)
    is held out together. A fresh probe is fitted on every other window except
    those sharing a group (simulation) with either held-out window, then predicts
    both. Windows outside in_pairs are never held out. also_predict lists extra
    matrices aligned with features (e.g. thinned copies): the same per-pair probe
    predicts their pair rows too. Returns (positive index, negative index,
    [(p_positive, p_negative) for features and each extra matrix]) per pair.
    """
    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    groups = list(groups)
    if x.ndim != 2 or len(x) != len(y) or len(groups) != len(y) or set(y.tolist()) - {0, 1}:
        raise ValueError("features, 0/1 labels and groups must align")
    selected = np.ones(len(y), bool) if in_pairs is None else np.asarray(in_pairs, bool)
    matrices = [x, *(np.asarray(m, dtype=np.float64) for m in also_predict)]
    positives = np.flatnonzero(selected & (y == 1))
    negatives = np.flatnonzero(selected & (y == 0))
    if not len(positives) or not len(negatives):
        raise ValueError("Leave-pair-out needs at least one positive and one negative window in pairs")
    pairs = []
    for i in positives:
        for j in negatives:
            keep = np.array([group not in (groups[i], groups[j]) for group in groups])
            if set(y[keep].tolist()) != {0, 1}:
                raise ValueError("Every leave-pair-out fit needs both labels after removing the pair's groups")
            model = make_probe().fit(x[keep], y[keep])
            pairs.append((int(i), int(j), [tuple(model.predict_proba(m[[i, j]])[:, 1].tolist()) for m in matrices]))
    return pairs


def pair_auroc(pairs, matrix=0):
    """Fraction of pairs where the positive window scores higher; ties count 0.5."""
    wins = [1.0 if p > n else 0.5 if p == n else 0.0 for p, n in (row[2][matrix] for row in pairs)]
    return sum(wins) / len(wins)


def validate_probe_leave_pair_out(features, labels, groups=None):
    """Reproduce the hidden protocol on windows you supply (e.g. the training windows).

    Every window takes part in pairs. groups are simulation identifiers; by
    default each window is its own group. No hidden data is reachable from here.
    """
    groups = list(range(len(labels))) if groups is None else list(groups)
    auroc = pair_auroc(leave_pair_out(features, labels, groups))
    return {"auroc": auroc, "unclipped_score": 2 * auroc - 1, "score": max(0.0, 2 * auroc - 1)}
