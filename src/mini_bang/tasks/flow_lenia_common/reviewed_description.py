"""Reviewed description scoring: source-bound reference authority and calibrated metrics."""
from copy import deepcopy

from mini_bang.framework.task import ValidationResult
from mini_bang.tasks.flow_lenia_common.classification import prediction_map
from mini_bang.tasks.flow_lenia_common.description import DescriptionScorer, RubricJudgeClient, description_metrics
from mini_bang.tasks.flow_lenia_common.reviewed_reference import event_authority
from mini_bang.rubric_judge.model_policy import REVIEWED_JUDGE_VERSION
from mini_bang.rubric_judge.ambiguity_history import apply_hold


class CompletedEncoding:
    """Replay the one actual encoding through the unchanged physical metric."""
    version = REVIEWED_JUDGE_VERSION

    def __init__(self, answer):
        self.answer = answer

    def evaluate(self, kind, data):
        return self.answer


class ReviewedDescriptionScorer:
    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store, scoring_services=None):
        bundle = gold_store.get("reference_bundle_version")
        declared = instance_spec.visible_data
        if bundle is None:
            if declared.get("reference_bundle_version"):
                raise ValueError("The instance declares a reference bundle but the hidden gold binds none")
            # Instances and gold without a reference bundle keep their declared protocol.
            return DescriptionScorer().score(task_spec=task_spec, instance_spec=instance_spec,
                submission=submission, run_record=run_record, gold_store=gold_store, scoring_services=scoring_services)
        if (declared.get("reference_bundle_version") != bundle or declared.get("rubric_version") != bundle
                or gold_store.get("rubric_version") != bundle
                or declared.get("judge_version") != REVIEWED_JUDGE_VERSION
                or gold_store.get("judge_version") != REVIEWED_JUDGE_VERSION):
            raise ValueError("Public/hidden reviewed description protocol identities disagree")
        cases = gold_store.get("cases")
        try:
            texts = prediction_map(submission.answer, "descriptions", [case["case_id"] for case in cases])
            if any(not isinstance(value, str) or not value.strip() or len(value.split()) > 250
                   or len(value) > 16000 for value in texts.values()):
                raise ValueError("Descriptions must be nonempty and at most250 words")
        except (ValueError, TypeError) as error:
            return ValidationResult(False, str(error), {"score": 0.0, "valid_submission": False})
        # Resolve and verify every source before constructing a paid judge client.
        authorities = [event_authority(case["annotation_id"], version=bundle) for case in cases]
        for _, index in authorities:
            if (gold_store.get("reference_authority_sha256") != index["authority_sha256"]
                    or declared.get("reference_authority_sha256") != index["authority_sha256"]):
                raise ValueError("Reviewed reference authority digest differs")
        context = scoring_services.scoring_context if scoring_services else None
        judge = RubricJudgeClient(context, version=REVIEWED_JUDGE_VERSION)
        rows = []
        for case, (authority, index) in zip(cases, authorities):
            data = {**deepcopy(authority), "description": texts[case["case_id"]]}
            # The full source payload has exactly the calibrated request shape.
            answer = judge.evaluate("description", data)
            native = judge.audit[-1]["metadata"].get("actual_native_identity", {})
            if (native.get("all_native_turns_match") is not True or native.get("selected_model") != judge.model
                    or native.get("selected_effort") != "high"):
                raise ValueError("Reviewed scoring requires the retained native model/effort proof")
            event = {key: data[key] for key in ("reference", "rubric", "rubric_semantics", "rubric_version",
                "rubric_weight_ledger", "interpretation_notes", "uncertain_claims")}
            event["traps"] = [item["text"] for item in data["traps"]]
            metrics = description_metrics(data["description"], event, CompletedEncoding(answer), seed=data["id"])
            history = judge.audit[-1]["metadata"].get("scope_ambiguity_history")
            if not isinstance(history, dict) or "scientific_grade_held" not in history:
                raise ValueError("Judge transport must preserve all completed-encoding ambiguity history")
            metrics = apply_hold(metrics, history)
            rows.append({"case_id": case["case_id"], "source_event": data["id"], **metrics})
        metrics = {key: sum(row[key] for row in rows) / len(rows)
            for key in ("score", "rubric_recall", "factual_precision", "unverifiable_fraction", "trap_count")}
        for key in ("claim_count", "supported_claim_count", "contradicted_claim_count", "unverifiable_claim_count"):
            metrics[key] = sum(row[key] for row in rows)
        held = any(row["scientific_grade_held"] for row in rows)
        metrics.update(valid_submission=True, judge_model=judge.model, judge_version=REVIEWED_JUDGE_VERSION,
            rubric_version=bundle, reference_bundle_version=bundle,
            reference_authority_sha256=authorities[0][1]["authority_sha256"], per_case=rows,
            scientific_grade_held=held, score_is_diagnostic=held,
            reference_checkable_fraction=(metrics["supported_claim_count"]+metrics["contradicted_claim_count"])
                / metrics["claim_count"] if metrics["claim_count"] else 0.0)
        if scoring_services:
            scoring_services.trace.append({"event": "description_judged", "hidden": True,
                "model": judge.model, "judge_version": REVIEWED_JUDGE_VERSION,
                "reference_bundle_version": bundle, "judge_audit": judge.audit})
        return ValidationResult(True, "Source-backed reviewed reference and calibrated scoring", metrics)
