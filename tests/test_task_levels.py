"""Every task config records its level in the five levels of understanding."""
from mini_bang.agent_runtime import get_json
from mini_bang.client.cli.official import OfficialRunServer
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.runtime import RunManager
from mini_bang.framework.task import TaskSubmission
from mini_bang.tasks.loader import TaskLoader
from tests.test_official_boundaries import _server

LEVEL_SCHEME = "five levels of understanding"
ALLOWED_LEVELS = {"1", "2", "3", "4", "5a", "5b"}
LEVEL_KEYS = ("level", "level_scheme")


class NoSimulationBackend:
    def describe(self):
        return {"backend": "no-simulation"}

    def run(self, request):
        raise AssertionError("preparing a run must not simulate")

    def run_batch(self, requests):
        raise AssertionError("preparing a run must not simulate")


def _levels(metadata):
    return {key: metadata.get(key) for key in LEVEL_KEYS}


def test_every_task_config_has_a_level_from_the_five_level_scheme():
    specs = {spec.task_id: spec for spec in TaskLoader().specs()}
    assert len(specs) == 11
    for task_id, spec in specs.items():
        assert spec.metadata.get("level") in ALLOWED_LEVELS, task_id
        assert spec.metadata.get("level_scheme") == LEVEL_SCHEME, task_id


def test_agent_visible_metadata_and_run_record_show_the_config_level():
    manager = RunManager(backend=NoSimulationBackend())
    for spec in TaskLoader().specs():
        expected = _levels(spec.metadata)
        session = manager.prepare(spec.task_id)  # the config's default instance
        assert session.instance_spec.instance_id == spec.default_instance_id
        assert _levels(session.environment.metadata) == expected, spec.task_id
        assert _levels(session.instance_spec.metadata) == expected, spec.task_id
        record = RunRecord.from_parts(run_id=session.run_id, task_id=session.task_id,
                                      instance_id=session.instance_id,
                                      submission=TaskSubmission(answer={}), trace=session.trace).to_dict()
        prepared = next(event for event in record["trace"] if event["event"] == "session_prepared")
        assert _levels(prepared) == expected, spec.task_id
        server = OfficialRunServer(session)
        with _server(server) as base:
            run = get_json(f"{base}/v1/run", token=server.session_token(session.run_id))
        assert _levels(run["metadata"]) == expected, spec.task_id


def test_released_instances_keep_loading_without_presenting_their_own_level():
    # An instance file may store its own level number; only the task config level is shown.
    store = BenchmarkArtifactStore()
    for spec in TaskLoader().specs():
        for instance_id in store.list_instance_ids(spec):
            instance = store.load_instance(spec, instance_id)
            assert _levels(instance.metadata) == _levels(spec.metadata), (spec.task_id, instance_id)
            assert _levels(instance.to_public_metadata()) == _levels(spec.metadata)
