"""RAF hidden objects served from recorded observation pools.

Every RAF task's default (test) and dev instances serve their hidden objects from the
pool shipped in the task package. Covered here with open data only (instances, pools,
dev gold): pool format, allowed and rejected requests, and complete runs in process and
through the official HTTP runner. Pool-versus-simulation identity and the leak check
against test gold are in test_raf_recorded_pool_identity.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.official import OfficialRunServer
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore, instance_split
from mini_bang.framework.runtime import RunManager
from mini_bang.tasks.loader import TaskLoader
from mini_bang.tasks.raf_common import hidden_networks, recorded_pool
from mini_bang.tasks.raf_mechanism_prob import families
from tests.test_official_boundaries import _server

TASKS = ("raf/timing", "raf/signature", "raf/mechanism-prob", "raf/mechanism-seq", "raf/predictive")


def instance_ids(task_id):
    spec = TaskLoader().get_spec(task_id)
    return spec.default_instance_id, spec.config["instance_variants"]["dev"]


def instance(task_id, instance_id):
    return BenchmarkArtifactStore().load_instance(TaskLoader().get_spec(task_id), instance_id)


def valid_request(task_id, spec):
    """One admissible request on the first recorded object of an instance."""
    pool = recorded_pool.load(spec)
    group = sorted(pool.groups)[0]
    sizes = sorted(pool.rollouts(group))
    times = pool.grid[-3:]
    request = {"saturation": sizes[0], "runs": 2, "snapshot_times": times, "extras": ["first_hits"]}
    if pool.address.get("member"):
        request.update(runs=1, macro_params={"family": group, "network_index": 1}, extras=["is_raf"])
    else:
        request["macro_params"] = {"network": group}
    if task_id == "raf/predictive":
        request.update(saturation=[sizes[0], sizes[-1]], extras=["first_hits", "is_raf"])
    return request


def constant_answer(task_id, visible):
    if task_id == "raf/timing":
        outcomes = [str(t) for t in visible["snapshot_times"]] + ["none"]
        return {"distributions": {net: {s: {o: 1 / len(outcomes) for o in outcomes} for s in species}
                                  for net, species in visible["target_species"].items()}}
    if task_id == "raf/signature":
        return {"predictions": {size: {name: 0.5 for name in names}
                                for size, names in visible["cases_by_saturation"].items()}}
    if task_id == "raf/mechanism-prob":
        return {"probabilities": {family: {str(t): 0.5 for t in visible["saturations"]}
                                  for family in visible["network_families"]}}
    if task_id == "raf/predictive":
        return {"probabilities": {chain: {str(t): 0.5 for t in visible["saturations"]}
                                  for chain in visible["test_networks"]}}
    quantiles = {"0.1": 0.5, "0.5": 1.5, "0.9": 3.0}
    return {"predicted_quantiles": {set_id: {chain: {size: {s: dict(quantiles) for s in species}
                                                     for size, species in levels.items()}
                                             for chain, levels in chains.items()}
                                    for set_id, chains in visible["target_species"].items()}}


@pytest.fixture
def no_hidden_registry(monkeypatch):
    """A run of a recorded instance must not need the hidden seed registries."""
    def forbidden(*args, **kwargs):
        raise AssertionError("hidden registry read during an agent run")
    monkeypatch.setattr(hidden_networks, "registry", forbidden)
    monkeypatch.setattr(families, "registry", forbidden, raising=False)


@pytest.mark.parametrize("task_id", TASKS)
def test_defaults_are_recorded_test_instances_with_a_dev_variant(task_id):
    test_id, dev_id = instance_ids(task_id)
    test, dev = instance(task_id, test_id), instance(task_id, dev_id)
    assert instance_split(test) == "test" and instance_split(dev) == "dev"
    test_pool, dev_pool = recorded_pool.load(test), recorded_pool.load(dev)
    assert test_pool.name != dev_pool.name and not set(test_pool.groups) & set(dev_pool.groups)
    for spec in (test, dev):
        pool = recorded_pool.load(spec)
        assert pool.name in spec.instructions
        assert re.search(r"sample_params\.seed (is|are) not available", spec.instructions)
        assert spec.visible_data["observation_pool"] == pool.name


@pytest.mark.parametrize("task_id", TASKS)
def test_pool_files_hold_only_counts_species_and_labels(task_id):
    for instance_id in instance_ids(task_id):
        spec = instance(task_id, instance_id)
        directory = recorded_pool.pool_directory(spec)
        manifest = json.loads((directory / f"{recorded_pool.pool_name(spec)}.json").read_text())
        assert set(manifest) == {"format", "name", "description", "snapshot_times", "address", "sequences",
                                 "groups", "arrays"}
        with np.load(directory / manifest["arrays"], allow_pickle=False) as arrays:
            for key in arrays.files:
                kind, group = key.split("__")[:2]
                assert kind in {"counts", "species", "labels"} and group in manifest["groups"]
                # Integer counts of at most 32 bits cannot hold a 63-bit seed; no floats at all.
                assert arrays[key].dtype.kind == "i" and arrays[key].dtype.itemsize <= 4


@pytest.mark.parametrize("task_id", TASKS)
def test_recorded_objects_reject_requests_the_pool_cannot_answer(task_id):
    session = RunManager(backend=LocalBackend()).prepare(task_id)
    api, pool = session.environment.api, recorded_pool.load(session.instance_spec)
    good = valid_request(task_id, session.instance_spec)
    named = "network" in good["macro_params"]
    group = good["macro_params"]["network" if named else "family"]
    size = good["saturation"] if type(good["saturation"]) is int else good["saturation"][0]
    rollouts = pool.rollouts(group)[size]
    bad = [
        {"sample_params": {"seed": 3}},
        {"runs": rollouts + 1},
        {"snapshot_times": [pool.grid[0], pool.grid[0]]},
        {"snapshot_times": [0.0123]},
        {"saturation": max(pool.rollouts(group)) + 1},
    ]
    if named:
        bad.append({"macro_params": {"network": group, "seed": 7}})
    else:
        bad.append({"macro_params": {"family": group, "network_index": pool.groups[group]["members"]}})
    before = dict(session.budget_counters)
    for change in bad:
        with pytest.raises(ValueError):
            api.generate_samples_batch([{**good, **change}])
    assert session.budget_counters == before
    payload = api.generate_samples_batch([good])[0]
    assert payload["runs"] == good["runs"] and "seed" not in payload["macro_params"]
    assert group in payload["macro_params"].values() and "seed" not in payload["sample_params"]
    assert "recorded_network" not in json.dumps(payload) and "recorded_network" not in json.dumps(session.trace)


@pytest.mark.parametrize("task_id", TASKS)
def test_dev_instance_runs_and_scores_in_process(task_id, no_hidden_registry):
    dev_id = instance_ids(task_id)[1]

    def agent(task):
        spec = task.api._session.instance_spec
        task.api.generate_samples_batch([valid_request(task_id, spec)])
        return constant_answer(task_id, spec.visible_data)

    result = RunManager(backend=LocalBackend()).run(agent, task_id, instance_id=dev_id)
    assert result.validation.metrics and result.session.budget_counters["generate_samples"] == 1


@pytest.mark.parametrize("task_id", TASKS)
def test_default_instance_runs_through_the_official_http_runner(task_id, tmp_path, monkeypatch, no_hidden_registry):
    monkeypatch.setenv("MINI_BANG_OBSERVATION_RECEIPTS_DIR", str(tmp_path / "receipts"))
    session = RunManager(backend=LocalBackend()).prepare(task_id)
    server = OfficialRunServer(session)
    token = server.session_token(session.run_id)
    good = valid_request(task_id, session.instance_spec)
    try:
        with _server(server) as url:
            run = get_json(url + "/v1/run", token=token)
            assert "recorded_observations" in run["capabilities"]
            response = post_json(url + "/v1/generate_samples", good, token=token)
            with pytest.raises(AgentRuntimeError):
                post_json(url + "/v1/generate_samples", {**good, "sample_params": {"seed": 5}}, token=token)
            answer = constant_answer(task_id, session.instance_spec.visible_data)
            post_json(url + "/v1/submission", {"answer": answer}, token=token)
            submission = server._submissions[str(session.run_id)]
    finally:
        server.observations.close()
    payload = response["payload"]
    assert payload["runs"] == good["runs"] and "recorded_network" not in json.dumps(response)
    assert session.budget_counters["generate_samples"] == 1
    # The organizer scores the submitted answer against the test gold (absent from the open tree).
    gold_path = session.instance_spec.gold_path
    if gold_path is None or not Path(gold_path).is_file():
        return
    store = BenchmarkArtifactStore()
    result = session.task_spec.create_scorer().score(
        task_spec=session.task_spec, instance_spec=session.instance_spec, submission=submission,
        run_record=None, gold_store=store.load_gold(session.instance_spec))
    assert result.metrics
