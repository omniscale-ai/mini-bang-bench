"""Run ids name the date-time, task and exact instance version of every current instance."""
import re

from mini_bang.client.cli.observation_journal import ObservationJournal
from mini_bang.naming import ELISION, MAX_ID_LENGTH, readable_id, shorten_label
from mini_bang.pool.protocol import validate_name
from mini_bang.tasks.loader import TaskLoader

STAMP = r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}"


def current_instances():
    for spec in TaskLoader().specs():
        for path in sorted((spec.package_dir / "instances").glob("*.json")):
            yield spec.task_id, path.stem


def test_every_current_instance_gets_a_distinct_readable_run_id(tmp_path):
    instances = list(current_instances())
    assert instances
    names = {}
    journal = ObservationJournal(tmp_path)
    for task_id, instance_id in instances:
        run_id = readable_id("run", task_id, instance_id)
        stamp, role, task_label, instance_label = run_id.split("--")
        assert re.fullmatch(STAMP, stamp) and role == "run"
        assert task_label == re.sub(r"[^a-z0-9-]+", "-", task_id.lower())
        words = re.findall(r"[a-z0-9]+", instance_id.lower())
        if ELISION in instance_label:
            assert instance_label.startswith(words[0]) and instance_label.endswith(words[-1])
            assert len(instance_label.split(ELISION)[1]) > len(instance_label.split(ELISION)[0])
        else:
            assert re.findall(r"[a-z0-9]+", instance_label) == words
        assert len(run_id) <= MAX_ID_LENGTH and ".." not in run_id
        journal.session_directory(run_id)
        validate_name("agent--" + run_id)
        validate_name(f"scoring--{task_label}--run-{run_id}")
        names.setdefault(f"{task_label}--{instance_label}", []).append(f"{task_id}/{instance_id}")
    assert {name: sources for name, sources in names.items() if len(sources) > 1} == {}


def test_instance_version_survives_and_short_names_stay_whole():
    run_id = readable_id("run", "sample/describe-events",
                         "event-01-disjoint-windows-population-brier-annotated-tiered-weights-rubric")
    assert run_id.endswith("--sample-describe-events--event-01_windows-population-brier-annotated-tiered-weights-rubric")
    run_id = readable_id("run", "sample/localization-cases", "inside-point-disjoint-windows-population-brier-"
                         "annotated-events-held-out-cases.localization-chance-adjusted-soft-f1-1")
    assert run_id.endswith("--inside-point_held-out-cases.localization-chance-adjusted-soft-f1-1")
    assert readable_id("run", "example/timing", "no-interventions-sizes").endswith(
        "--run--example-timing--no-interventions-sizes")
    assert len(readable_id("scoring", "x" * 300, "y-" * 300)) == MAX_ID_LENGTH
    assert shorten_label("golden100-" + "z" * 300, 40) == "golden100_" + "z" * 30


def test_ids_cut_at_a_fixed_length_remain_valid(tmp_path):
    old = "2000-01-01-00-00-00-000000--run--flow-lenia-presence--windows-balanced-brier-skill-held-out-c"
    assert ObservationJournal(tmp_path).session_directory(old) == tmp_path / old
    assert validate_name("agent--" + old)
