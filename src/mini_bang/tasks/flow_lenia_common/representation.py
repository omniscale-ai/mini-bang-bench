from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from mini_bang.execution.payloads import PAYLOAD_REF_SENTINEL
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.task import ValidationResult
from mini_bang.tasks.flow_lenia_common.annotations import gold_intervals, load_pool
from mini_bang.tasks.flow_lenia_common.api import observation_request, observation_settings
from mini_bang.tasks.flow_lenia_common.classification import binary_metrics


def checkpoint_ranges(case, *, preferred_step=3000, max_states=64, max_raw_steps=180000):
    """Cover a full window, keeping its endpoints and every planned checkpoint.

    The final short cadence gap is a separate request when necessary. Chunking
    changes delivery only: the executable receives one continuous ordered stream.
    """
    start, end = (case['t1'] + 99) // 100 * 100, (case['t2'] - 1) // 100 * 100
    if start >= end:
        raise ValueError('Feature windows require at least two available checkpoints')
    span = end - start
    step = max(preferred_step, ((span + max_states - 3) // (max_states - 2) + 99) // 100 * 100)
    points = list(range(start, end + 1, step))
    if points[-1] != end:
        points.append(end)
    ranges, index = [], 0
    while index < len(points):
        last = index
        while (last + 1 < len(points) and last - index + 1 < max_states
               and points[last + 1] - points[index] <= max_raw_steps
               and points[last + 1] - points[last] == step):
            last += 1
        ranges.append((points[index], points[last], step))
        index = last + 1
    return ranges


TRAIN_ONCE_PROBE_PROTOCOL = "train_once_hidden_test"
LEAVE_PAIR_OUT_PROTOCOL = "leave_pair_out_grouped"
MAX_FEATURES = {TRAIN_ONCE_PROBE_PROTOCOL: 128, LEAVE_PAIR_OUT_PROTOCOL: 8}


def probe_protocol(instance_spec, gold_store):
    """The gold or instance declaration chooses the probe; without one it is the train-once probe."""
    declared = gold_store.get("probe_protocol")
    visible = (getattr(instance_spec, "config", {}) or {}).get("probe_protocol")
    if declared and visible and declared != visible:
        raise ValueError("Visible and hidden probe protocols disagree")
    protocol = declared or visible or TRAIN_ONCE_PROBE_PROTOCOL
    if protocol not in MAX_FEATURES:
        raise ValueError(f"Unknown probe protocol: {protocol}")
    return protocol


def validate_features(output, count, max_dimension=128):
    if not isinstance(output, dict) or set(output) != {"features"}:
        raise ValueError("Missing feature matrices")
    matrices = output["features"]
    if not isinstance(matrices, dict) or set(matrices) != {"1", "2", "4"}:
        raise ValueError("All three thinning matrices are required")
    dimension = None
    for matrix in matrices.values():
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != count or not 1 <= array.shape[1] <= max_dimension:
            raise ValueError("Feature matrix has the wrong shape")
        if not np.isfinite(array).all() or np.max(np.abs(array)) > 1e100:
            raise ValueError("Features must be finite and numerically usable")
        dimension = array.shape[1] if dimension is None else dimension
        if dimension != array.shape[1]:
            raise ValueError("Inconsistent feature dimensions")


def probe_metrics(features, cases, pool):
    from mini_bang.tasks.flow_lenia_common.probe import make_probe, PROBE_DESCRIPTION

    train = np.array([case["split"] == "train" for case in cases])
    labels = np.array([int(bool(gold_intervals(pool, case))) for case in cases])
    for split in (train, ~train):
        if set(labels[split]) != {0, 1}:
            raise ValueError("Probe requires both labels in train and hidden test")
    if ({c["trajectory"] for c in cases if c["split"] == "train"}
            & {c["trajectory"] for c in cases if c["split"] == "test"}):
        raise ValueError("Representation splits must be trajectory-disjoint")
    # Fit ONCE on unthinned training windows. Thinning is a robustness test of
    # the same frozen encoder+probe, not permission to train three probes.
    classifier = make_probe()
    full = np.asarray(features["1"], dtype=np.float64)
    classifier.fit(full[train], labels[train])
    metrics, predictions_by_factor = {}, {}
    for factor in ("1", "2", "4"):
        matrix = np.asarray(features[factor], dtype=np.float64)
        predictions = classifier.predict_proba(matrix)[:, 1].tolist()
        predictions_by_factor[factor] = predictions
        metrics[factor] = binary_metrics(labels[~train].tolist(), np.asarray(predictions)[~train].tolist())
    return {"score": metrics["1"]["auroc"], "feature_dimension": full.shape[1],
            "probe": PROBE_DESCRIPTION,
            "per_case": feature_diagnostics(cases, labels, predictions_by_factor),
            "thinning": metrics, "thinning_2_auroc_drop": metrics["1"]["auroc"] - metrics["2"]["auroc"],
            "thinning_4_auroc_drop": metrics["1"]["auroc"] - metrics["4"]["auroc"],
            "n_train": int(train.sum()), "n_test": int((~train).sum()),
            "n_train_trajectories": len({c["trajectory"] for c in cases if c["split"] == "train"}),
            "n_test_trajectories": len({c["trajectory"] for c in cases if c["split"] == "test"})}


def leave_pair_out_metrics(features, cases, pool):
    """Grouped leave-pair-out over hidden windows; training windows always stay in the fit."""
    from mini_bang.tasks.flow_lenia_common.probe import LEAVE_PAIR_OUT_DESCRIPTION, leave_pair_out, pair_auroc

    hidden = np.array([case["split"] == "test" for case in cases])
    labels = np.array([int(bool(gold_intervals(pool, case))) for case in cases])
    for split in (~hidden, hidden):
        if set(labels[split]) != {0, 1}:
            raise ValueError("Probe requires both labels in train and hidden test")
    if ({c["trajectory"] for c in cases if c["split"] == "train"}
            & {c["trajectory"] for c in cases if c["split"] == "test"}):
        raise ValueError("Representation splits must be trajectory-disjoint")
    factors = ("1", "2", "4")
    full = np.asarray(features["1"], dtype=np.float64)
    pairs = leave_pair_out(full, labels, [case["trajectory"] for case in cases], in_pairs=hidden,
                           also_predict=[features[factor] for factor in factors[1:]])
    thinning = {}
    for index, factor in enumerate(factors):
        auroc = pair_auroc(pairs, index)
        thinning[factor] = {"auroc": auroc, "unclipped_score": 2 * auroc - 1,
                            "pair_ties": sum(row[2][index][0] == row[2][index][1] for row in pairs)}
    per_case = []
    for i in np.flatnonzero(hidden):
        own = [(row[2], 0 if row[0] == i else 1) for row in pairs if i in row[:2]]
        per_case.append({"case_id": cases[i]["case_id"], "trajectory": cases[i]["trajectory"],
                         "window": [cases[i]["t1"], cases[i]["t2"]], "split": "test", "label": bool(labels[i]),
                         "mean_pair_probability_by_thinning": {factor: float(np.mean([p[k][side] for p, side in own]))
                                                               for k, factor in enumerate(factors)}})
    auroc = thinning["1"]["auroc"]
    return {"score": max(0.0, 2 * auroc - 1), "unclipped_score": 2 * auroc - 1, "auroc": auroc,
            "primary_metric": "leave_pair_out_auroc_skill", "probe_protocol": LEAVE_PAIR_OUT_PROTOCOL,
            "feature_dimension": full.shape[1], "max_feature_dimension": MAX_FEATURES[LEAVE_PAIR_OUT_PROTOCOL],
            "probe": LEAVE_PAIR_OUT_DESCRIPTION, "n_pairs": len(pairs), "per_case": per_case,
            "thinning": thinning, "thinning_2_auroc_drop": auroc - thinning["2"]["auroc"],
            "thinning_4_auroc_drop": auroc - thinning["4"]["auroc"],
            "n_train": int((~hidden).sum()), "n_test": int(hidden.sum()),
            "n_train_trajectories": len({c["trajectory"] for c in cases if c["split"] == "train"}),
            "n_test_trajectories": len({c["trajectory"] for c in cases if c["split"] == "test"})}


class RepresentationScorer:
    entrypoint = "featurize"
    evaluate_features = staticmethod(probe_metrics)

    def feature_protocol(self, instance_spec, gold_store):
        """(evaluate, max features); without a declaration the train-once probe and 1..128."""
        protocol = probe_protocol(instance_spec, gold_store)
        if protocol == LEAVE_PAIR_OUT_PROTOCOL:
            return leave_pair_out_metrics, MAX_FEATURES[protocol]
        return self.evaluate_features, MAX_FEATURES[protocol]

    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store, scoring_services=None):
        answer = submission.answer
        if (not isinstance(answer, dict) or set(answer) != {"code"} or not isinstance(answer["code"], str)
                or not 1 <= len(answer["code"].encode()) <= 100000):
            return ValidationResult(False, "Submit Python code (at most 100 kB)", {"score": 0.0, "valid_submission": False})
        if scoring_services is None:
            raise RuntimeError("Representation requires isolated executable and simulation services")
        if observation_settings()[1] == "video":
            raise RuntimeError("Representation requires raw checkpoints or verified replay, not video-only media")
        evaluate, max_features = self.feature_protocol(instance_spec, gold_store)
        pool, cases = load_pool(gold_store), list(gold_store.get("cases"))
        random.SystemRandom().shuffle(cases)
        requests, request_cases = [], []
        limits = task_spec.config['api']
        for index, case in enumerate(cases):
            plan = checkpoint_ranges(case, max_states=limits['max_states_per_request'],
                                     max_raw_steps=limits['max_request_raw_steps'])
            for start, end, step in plan:
                requests.append(observation_request(case['case_id'], start, end, step, ['A', 'P', 'Food'],
                                                    instance_id=instance_spec.instance_id))
                request_cases.append(index)
        results = scoring_services.run_batch(requests)
        if len(results) != len(requests):
            raise RuntimeError('Scoring observations lost a requested chunk')
        staged, files, uncompressed = [], [[] for _ in cases], 0
        for index, (case_index, result) in enumerate(zip(request_cases, results)):
            payload = result.materialize_payload()
            if payload["observation"].get(PAYLOAD_REF_SENTINEL) is not True:
                raise RuntimeError("Observation worker returned no NPZ payload reference")
            ref = payload["observation"]["ref"]
            name = f"window_{index:03d}.npz"
            files[case_index].append(name)
            # Absolute timestamps would reveal the annotated-window sampling
            # pattern. The trusted stager strips them before code can read NPZ.
            sample = requests[index].sample_params
            expected_steps = list(range(sample['start_step'], sample['end_step'] + 1, sample['step']))
            staged.append({"name": name, "ref": ref, "fields": ["A", "P", "Food"],
                           "expected_steps": expected_steps})
            # Count original numerical precision, not ZIP compression or an
            # invented fp32 size for the original fp16 observations.
            uncompressed += sum(int(np.prod(shape)) * np.dtype(payload["dtypes"][key]).itemsize
                                for key, shape in payload["shapes"].items() if key in {"A", "P", "Food"})
        harness = Path(__file__).with_name("representation_worker.py").read_text()
        config = {"code": answer["code"], "files": files, "staged_artifacts": staged, "entrypoint": self.entrypoint}
        if max_features != MAX_FEATURES[TRAIN_ONCE_PROBE_PROTOCOL]:
            config["max_features"] = max_features  # The train-once input has no max_features key.
        result = scoring_services.run_executable({"code": harness}, config,
            ExecutionLimits(timeout_seconds=600, memory_mb=4096, cpu_count=2, network_enabled=False),
            output_schema=lambda output: validate_features(output, len(cases), max_features))
        if result.exit_code or result.timed_out:
            return ValidationResult(False, "Feature extraction failed or exceeded sandbox limits",
                                    {"score": 0.0, "valid_submission": False, "timed_out": result.timed_out})
        metrics = evaluate(result.output["features"], cases, pool)
        metrics.update(valid_submission=True, compression_ratio=uncompressed / (len(cases) * metrics["feature_dimension"] * 8))
        return ValidationResult(True, "Submitted representation or scalar evaluated on trajectory-disjoint hidden windows", metrics)


def feature_diagnostics(cases, labels, predictions):
    return [{"case_id": case["case_id"], "trajectory": case["trajectory"],
             "window": [case["t1"], case["t2"]], "split": case["split"], "label": bool(labels[i]),
             "values_by_thinning": {factor: float(values[i]) for factor, values in predictions.items()}}
            for i, case in enumerate(cases)]


def scalar_metrics(features, cases, pool):
    """Original scalar metric task: rank held-out windows, without a fitted probe."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    test = np.array([c["split"] == "test" for c in cases])
    all_labels = np.array([int(bool(gold_intervals(pool, c))) for c in cases])
    labels = all_labels[test]
    if set(labels) != {0, 1}:
        raise ValueError("Scalar metric needs both hidden labels")
    if ({c["trajectory"] for c in cases if c["split"] == "train"}
            & {c["trajectory"] for c in cases if c["split"] == "test"}):
        raise ValueError("Metric splits must be trajectory-disjoint")
    metrics = {}
    for factor, matrix in features.items():
        values = np.asarray(matrix)
        if values.shape != (len(cases), 1):
            raise ValueError("score(window) must return exactly one scalar")
        values = values[test, 0]
        metrics[factor] = {"auroc": float(roc_auc_score(labels, values)),
                          "spearman": float(spearmanr(labels, values).statistic) if np.ptp(values) else 0.0}
    return {"score": metrics["1"]["auroc"], "feature_dimension": 1, "probe": "none; submitted scalar, larger means minibang",
            "per_case": feature_diagnostics(cases, all_labels,
                           {factor: np.asarray(matrix)[:, 0] for factor, matrix in features.items()}),
            "thinning": metrics, "thinning_2_auroc_drop": metrics["1"]["auroc"] - metrics["2"]["auroc"],
            "thinning_4_auroc_drop": metrics["1"]["auroc"] - metrics["4"]["auroc"],
            "n_train": int((~test).sum()), "n_test": int(test.sum())}


RAW_AUROC_SCALAR_METRIC = "hidden_test_auroc"
AUROC_SKILL_SCALAR_METRIC = "hidden_test_auroc_skill"


def scalar_metric(instance_spec, gold_store):
    """The gold or instance declaration chooses the metric; without one it is raw AUROC."""
    declared = gold_store.get("scalar_metric")
    visible = (getattr(instance_spec, "config", {}) or {}).get("scalar_metric")
    if declared and visible and declared != visible:
        raise ValueError("Visible and hidden scalar metric versions disagree")
    metric = declared or visible or RAW_AUROC_SCALAR_METRIC
    if metric not in {RAW_AUROC_SCALAR_METRIC, AUROC_SKILL_SCALAR_METRIC}:
        raise ValueError(f"Unknown scalar metric protocol: {metric}")
    return metric


def auroc_skill_metrics(features, cases, pool):
    """Same hidden AUROC; score = max(0, 2*AUROC - 1) so constant and random metrics score about 0."""
    metrics = scalar_metrics(features, cases, pool)
    auroc = metrics["score"]
    return {**metrics, "score": max(0.0, 2 * auroc - 1), "unclipped_score": 2 * auroc - 1, "auroc": auroc,
            "primary_metric": AUROC_SKILL_SCALAR_METRIC, "scalar_metric": AUROC_SKILL_SCALAR_METRIC}


class MinibangMetricScorer(RepresentationScorer):
    entrypoint = "score"
    evaluate_features = staticmethod(scalar_metrics)

    def feature_protocol(self, instance_spec, gold_store):
        # Metric design has its own scalar metrics; probe protocols do not apply.
        if scalar_metric(instance_spec, gold_store) == AUROC_SKILL_SCALAR_METRIC:
            return auroc_skill_metrics, MAX_FEATURES[TRAIN_ONCE_PROBE_PROTOCOL]
        return self.evaluate_features, MAX_FEATURES[TRAIN_ONCE_PROBE_PROTOCOL]
