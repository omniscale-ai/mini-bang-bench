from types import SimpleNamespace

from mini_bang.tasks.flow_lenia_common import description
from mini_bang.tasks.flow_lenia_common.description import description_metrics


class SavedJudge:
    def __init__(self, claims):
        self.claims = claims

    def evaluate(self, kind, data):
        return {
            "coverage": [{"id": "change", "present": True, "quote": "A changes", "reason": "Explicit change."}],
            "traps": [],
            "claims": self.claims,
        }


def claim(text, label, evidence=""):
    return {"claim": text, "quote": text, "label": label,
            "reference_quote": evidence, "reason": "Saved semantic decision."}


EVENT = {"reference": "A changes. B remains.",
         "rubric": [{"id": "change", "weight": 1, "text": "A changes"}], "traps": []}


def test_high_precision_exposes_how_little_the_reference_can_check():
    text = "A changes. C moves. D rotates."
    judge = SavedJudge([claim("A changes", "supported", "A changes"),
                        claim("C moves", "unverifiable"), claim("D rotates", "unverifiable")])
    result = description_metrics(text, EVENT, judge, seed="reference-coverage")
    assert result["factual_precision"] == 1.0
    assert result["reference_checkable_fraction"] == 1 / 3
    assert result["supported_claim_count"] == 1
    assert result["contradicted_claim_count"] == 0
    assert result["unverifiable_claim_count"] == 2
    assert result["score"] == 1.0


def test_contradictions_enter_precision_but_unverifiable_claims_do_not():
    text = "A changes. B disappears. C moves. D rotates."
    judge = SavedJudge([claim("A changes", "supported", "A changes"),
                        claim("B disappears", "contradicted", "B remains"),
                        claim("C moves", "unverifiable"), claim("D rotates", "unverifiable")])
    result = description_metrics(text, EVENT, judge, seed="reference-coverage")
    assert result["reference_checkable_fraction"] == 0.5
    assert result["factual_precision"] == 0.5
    assert result["score"] == 2 / 3
    assert result["claim_count"] == sum(result[key] for key in
        ("supported_claim_count", "contradicted_claim_count", "unverifiable_claim_count"))


def test_scorer_pools_claim_counts_without_changing_case_scores(monkeypatch):
    event = {**EVENT, "annotation_id": "change"}
    monkeypatch.setattr(description, "load_pool", lambda gold: {
        "trajectories": {"example": {"events": [event]}}})

    class CaseJudge(SavedJudge):
        model = "saved-judge"

        def evaluate(self, kind, data):
            self.claims = [claim("A changes", "supported", "A changes")]
            if "C moves" in data["description"]:
                self.claims += [claim("C moves", "unverifiable"), claim("D rotates", "unverifiable")]
            return super().evaluate(kind, data)

    monkeypatch.setattr(description, "RubricJudgeClient", lambda context, **kwargs: CaseJudge([]))
    cases = [{"case_id": key, "trajectory": "example", "annotation_id": "change"} for key in ("first", "second")]
    result = description.DescriptionScorer().score(
        task_spec=None, instance_spec=None, run_record=None, gold_store={"cases": cases},
        submission=SimpleNamespace(answer={"descriptions": {
            "first": "A changes. C moves. D rotates.", "second": "A changes."}}))
    assert result.success
    assert result.metrics["score"] == 1.0
    assert result.metrics["factual_precision"] == 1.0
    assert result.metrics["reference_checkable_fraction"] == 0.5
    assert result.metrics["claim_count"] == 4
    assert result.metrics["supported_claim_count"] == 2
    assert result.metrics["unverifiable_claim_count"] == 2
