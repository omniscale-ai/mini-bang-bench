"""Synthetic hidden gold, so tests drive session and scoring code independently of the real gold.

A test that needs some gold only to exercise code calls use_synthetic_gold(monkeypatch):

- instance SMOKE_INSTANCE of raf_mechanism_prob and raf_timing is the small
  visible instance below;
- gold of raf_mechanism_prob, flow_lenia_representation and
  flow_lenia_minibang_metric is synthetic for every instance (for network
  families, with synthetic hidden p values);
- the Flow Lenia scorers read SYNTHETIC_POOL instead of the bundled annotations.

A subprocess gets the same synthetic gold with
python -m tests.synthetic_gold -m <module> [arguments].
"""
from __future__ import annotations

import copy
from pathlib import Path
import runpy
import sys

from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore, GoldStore
from mini_bang.framework.task import InstanceSpec
from mini_bang.tasks.flow_lenia_common import annotations

SMOKE_INSTANCE = "synthetic-smoke"

SMOKE_INSTANCES = {
    "raf_mechanism_prob": {
        "description": "Estimate RAF emergence probabilities for fixed saturation values.",
        "instructions": "Use generate_samples(saturation=T, runs=..., extras=[\"is_raf\"]), "
                        "then return {'probabilities': {T: probability}}.",
        "metadata": {"level": 5, "category": "Mechanism Inference"},
        "visible": {
            "simulator_id": "raf",
            "saturations": [6, 7, 8, 9],
            "snapshot_times": [0.25, 0.5, 0.75, 1.0],
            "train_seeds": [411, 412, 413, 414, 415],
        },
    },
    "raf_timing": {
        "description": "Estimate first-appearance distributions for a fixed RAF saturation.",
        "instructions": "Use generate_samples(T, runs, snapshot_times, extras=[\"first_hits\"]), then return "
                        "{'distributions': {seed_or_default: {species: {time_or_none: probability}}}}.",
        "metadata": {"level": 2, "category": "Anomaly Detection", "api_limits": {"max_generate_calls": 10}},
        "visible": {
            "simulator_id": "raf",
            "saturation": 6,
            "snapshot_times": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "simulator_seeds": ["111", "222", "333"],
        },
    },
}


def _closed_form(size, p, m0):
    return (1 - (1 - p) ** (size - m0)) ** (size - m0)


def raf_probability_gold(instance: InstanceSpec) -> dict:
    visible = instance.visible_data
    macro = visible.get("macro_params")
    if macro is None:
        # The smoke instance: every hidden network is autocatalytic.
        return {"probabilities": {str(size): 1.0 for size in visible["saturations"]}}
    families = visible.get("network_families")
    if families is not None:
        # Network families: synthetic hidden p values in name order, not the real ones.
        values = {name: 0.1 + 0.05 * index for index, name in enumerate(families)}
        return {"hidden_families": {name: {"p": p} for name, p in values.items()},
                "probabilities": {name: {str(size): _closed_form(size, p, macro["M0"])
                                         for size in visible["saturations"]} for name, p in values.items()}}
    # Other instances: the published closed form of their public macro regime.
    return {"probabilities": {str(size): _closed_form(size, macro["p"], macro["M0"])
                              for size in visible["saturations"]}}


# Eight simulations with one minibang each; four train and four hidden test simulations.
FLOW_EVENT = [100000, 110000]
FLOW_WINDOWS = {"with-event": (95000, 115000), "quiet": (300000, 320000)}
FLOW_SPLITS = {f"traj_{number:05d}": "train" if number < 4 else "test" for number in range(8)}

SYNTHETIC_POOL = {
    "pool_id": annotations.POOL_ID,
    "gold_version": "synthetic",
    "trajectories": {
        name: {"complete": True, "last_step": annotations.LAST_STEP, "stabilized_from_step": 0,
               "events": [{"annotation_id": f"{name}-event", "interval": FLOW_EVENT}], "negatives": []}
        for name in FLOW_SPLITS
    },
}


def flow_window_gold(instance: InstanceSpec) -> dict:
    return {"cases": [
        {"case_id": f"synthetic-{name}-{kind}__steps-{start}-{end}", "trajectory": name,
         "t1": start, "t2": end, "observation_window": [start, end], "split": split}
        for name, split in FLOW_SPLITS.items() for kind, (start, end) in FLOW_WINDOWS.items()
    ]}


SYNTHETIC_GOLD = {
    "raf_mechanism_prob": raf_probability_gold,
    "flow_lenia_representation": flow_window_gold,
    "flow_lenia_minibang_metric": flow_window_gold,
}


def package_name(spec) -> str:
    return spec.package.rpartition(".")[2]


def use_synthetic_gold(monkeypatch) -> None:
    real_instance, real_gold = BenchmarkArtifactStore.load_instance, BenchmarkArtifactStore.load_gold

    def load_instance(store, task_spec, instance_id=None):
        config = SMOKE_INSTANCES.get(package_name(task_spec)) if instance_id == SMOKE_INSTANCE else None
        if config is None:
            return real_instance(store, task_spec, instance_id)
        return InstanceSpec.from_config(task_id=task_spec.task_id, package=task_spec.package,
                                        instance_id=SMOKE_INSTANCE, config=copy.deepcopy(config),
                                        task_metadata=task_spec.metadata)

    def load_gold(store, instance_spec):
        build = SYNTHETIC_GOLD.get(package_name(instance_spec))
        if build is None:
            return real_gold(store, instance_spec)
        path = Path("synthetic-gold", package_name(instance_spec), f"{instance_spec.instance_id}.json")
        return GoldStore(task_id=instance_spec.task_id, instance_id=instance_spec.instance_id, path=path,
                         data=build(instance_spec), artifact_resolver=store.artifact_resolver)

    monkeypatch.setattr(BenchmarkArtifactStore, "load_instance", load_instance)
    monkeypatch.setattr(BenchmarkArtifactStore, "load_gold", load_gold)
    monkeypatch.setattr(annotations, "bundled_pool",
                        lambda gold_store=None, *, root=None: copy.deepcopy(SYNTHETIC_POOL))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        raise SystemExit("usage: python -m tests.synthetic_gold -m <module> [arguments]")
    import pytest

    use_synthetic_gold(pytest.MonkeyPatch())
    module = sys.argv[2]
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
