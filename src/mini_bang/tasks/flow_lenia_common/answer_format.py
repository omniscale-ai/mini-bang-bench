"""Published answer formats of the Flow Lenia tasks, checked from the visible instance only.

Each format mirrors the answer reader of its scorer, with the visible cases in
place of the gold cases (the same case ids). Referenced from task configs as
``answer_format``.
"""
from __future__ import annotations

from mini_bang.framework.answer_format import FormatProblem, require_keys, require_mapping
from mini_bang.tasks.flow_lenia_common.classification import (
    JOINT_REASONS_METRIC, PER_WINDOW_REASONS_METRIC, probability, probability_metric)
from mini_bang.tasks.flow_lenia_common.discriminate_joint import reason_probabilities
from mini_bang.tasks.flow_lenia_common.localization import (
    INSIDE_POINT_ALL_WINDOWS, inside_point_windows, validate_intervals)

MAX_CODE_BYTES = 100000
MAX_DESCRIPTION_WORDS = 250
MAX_DESCRIPTION_CHARACTERS = 16000


class NoGold:
    """Stands in for gold in declaration readers: only the instance declaration counts."""

    def get(self, key, default=None):
        return default


def visible_cases(instance_spec) -> list[dict]:
    cases = instance_spec.visible_data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise FormatProblem("The instance publishes no cases")
    return cases


def case_map(answer, key, instance_spec) -> dict:
    given = require_mapping(answer.get(key), f"'{key}'")
    require_keys(given, [case["case_id"] for case in visible_cases(instance_spec)], f"'{key}'")
    return given


def checked(function, *args):
    try:
        return function(*args)
    except (ValueError, TypeError) as exc:
        raise FormatProblem(str(exc)) from exc


class PresenceFormat:
    """{'predictions': {case_id: probability}}."""

    def problems(self, task_spec, instance_spec, answer):
        for case_id, value in case_map(answer, "predictions", instance_spec).items():
            checked(probability, value)
        return []

    def example(self, task_spec, instance_spec):
        return {"predictions": {case["case_id"]: 0.5 for case in visible_cases(instance_spec)}}


class LocalizationFormat:
    """{'intervals': {case_id: [[start_step, end_step], ...]}}."""

    def problems(self, task_spec, instance_spec, answer):
        given = case_map(answer, "intervals", instance_spec)
        limit = task_spec.config["validation"]["max_intervals"]
        allow_empty = inside_point_windows(instance_spec, NoGold()) == INSIDE_POINT_ALL_WINDOWS
        for case in visible_cases(instance_spec):
            checked(validate_intervals, given[case["case_id"]], case, limit, allow_empty)
        return []

    def example(self, task_spec, instance_spec):
        intervals = {}
        for case in visible_cases(instance_spec):
            if "inside_t" in case:
                inside = case["inside_t"]
                intervals[case["case_id"]] = [[max(case["t1"], inside - 1000), min(case["t2"], inside + 1000)]]
            else:
                intervals[case["case_id"]] = []
        return {"intervals": intervals}


class DiscriminateFormat:
    """{'predictions': {case_id: {'probability': p, 'reasons': [names] or {name: probability}}}}."""

    def problems(self, task_spec, instance_spec, answer):
        metric = checked(probability_metric, instance_spec, NoGold())
        for case_id, item in case_map(answer, "predictions", instance_spec).items():
            if not isinstance(item, dict) or set(item) != {"probability", "reasons"}:
                raise FormatProblem(f"Prediction for {case_id} requires exactly probability and reasons")
            checked(probability, item["probability"])
            if metric == JOINT_REASONS_METRIC:
                checked(reason_probabilities, item["reasons"])
            elif metric == PER_WINDOW_REASONS_METRIC or isinstance(item["reasons"], list):
                if not isinstance(item["reasons"], list) or any(not isinstance(r, str) for r in item["reasons"]):
                    raise FormatProblem(f"Reasons for {case_id} must be a list of names")
        return []

    def example(self, task_spec, instance_spec):
        return {"predictions": {case["case_id"]: {"probability": 0.5, "reasons": []}
                                for case in visible_cases(instance_spec)}}


class CodeFormat:
    """{'code': Python source} defining the published entrypoint."""

    def __init__(self, entrypoint: str):
        self.entrypoint = entrypoint

    def problems(self, task_spec, instance_spec, answer):
        if set(answer) != {"code"} or not isinstance(answer["code"], str):
            raise FormatProblem("The answer must be {'code': Python source}")
        if not 1 <= len(answer["code"].encode()) <= MAX_CODE_BYTES:
            raise FormatProblem(f"Submitted code must hold 1 to {MAX_CODE_BYTES} bytes")
        return []

    def example(self, task_spec, instance_spec):
        result = "0.0" if self.entrypoint == "score" else "[0.0]"
        return {"code": f"def {self.entrypoint}(window):\n    return {result}\n"}


class DescriptionFormat:
    """{'descriptions': {case_id: text of at most 250 words}}."""

    def problems(self, task_spec, instance_spec, answer):
        for case_id, text in case_map(answer, "descriptions", instance_spec).items():
            if (not isinstance(text, str) or not text.strip() or len(text.split()) > MAX_DESCRIPTION_WORDS
                    or len(text) > MAX_DESCRIPTION_CHARACTERS):
                raise FormatProblem(f"Description of {case_id} must be nonempty and at most "
                                    f"{MAX_DESCRIPTION_WORDS} words")
        return []

    def example(self, task_spec, instance_spec):
        return {"descriptions": {case["case_id"]: "The population changes." for case in visible_cases(instance_spec)}}


PRESENCE = PresenceFormat()
LOCALIZATION = LocalizationFormat()
DISCRIMINATE = DiscriminateFormat()
SCALAR_METRIC = CodeFormat("score")
REPRESENTATION = CodeFormat("featurize")
DESCRIPTION = DescriptionFormat()
