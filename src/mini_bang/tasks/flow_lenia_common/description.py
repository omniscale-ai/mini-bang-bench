from __future__ import annotations

import json
import math
import os
import random
import urllib.request

from mini_bang.framework.task import ValidationResult
from mini_bang.tasks.flow_lenia_common.annotations import load_pool
from mini_bang.tasks.flow_lenia_common.classification import prediction_map
from mini_bang.rubric_judge.contract import JUDGE_VERSION, DEFAULT_JUDGE_VERSION, BASE_JUDGE_VERSION, LABELS, HEALTH_PROFILE
from mini_bang.rubric_judge.validation import validate_comparisons
from mini_bang.rubric_judge.contract import REVIEWED_JUDGE_VERSION, profile_for
from mini_bang.rubric_judge.model_policy import model_authority, AUTHORIZATION_ENV
from mini_bang.tasks.flow_lenia_common.description_rubrics import (
    ATOMIC_JUDGE_VERSION, ATOMIC_RUBRIC_VERSION, atomic_event,
)


class RubricJudgeClient:
    """HTTP client of the configured rubric judge service used by description scoring."""

    def __init__(self, context=None, *, version=None):
        environment = context.env if context is not None else os.environ
        self.url = environment.get("MINI_BANG_JUDGE_URL", "")
        self.token = environment.get("MINI_BANG_JUDGE_TOKEN", "")
        self.model = environment.get("MINI_BANG_JUDGE_MODEL", "")
        self.evaluated_model = environment.get("MINI_BANG_EVALUATED_MODEL", "")
        self.version = version or environment.get("MINI_BANG_JUDGE_VERSION", DEFAULT_JUDGE_VERSION)
        try:
            timeout = float(environment.get("MINI_BANG_JUDGE_HTTP_TIMEOUT", "1000"))
        except (TypeError, ValueError) as error:
            raise ValueError("Judge HTTP timeout must be a finite nonnegative number") from error
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Judge HTTP timeout must be a finite nonnegative number")
        # Zero waits for this request without a timeout; bounded model calls and
        # the independent latency monitor still detect a stalled judge.
        self.http_timeout_seconds = None if timeout == 0 else timeout
        self.audit = []
        if not all((self.url, self.token, self.model, self.evaluated_model)):
            raise RuntimeError("Description scoring requires the configured rubric judge service")
        self.model_authority = model_authority(version=self.version, profile=profile_for(self.version),
            model=self.model, evaluated_model=self.evaluated_model, authorization=environment.get(AUTHORIZATION_ENV))

    def evaluate(self, kind, data):
        request = urllib.request.Request(self.url.rstrip("/") + "/judge",
                    data=json.dumps({"kind": kind, "data": data, "judge_version": self.version}).encode(),
                    headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.http_timeout_seconds) as response:
            result = json.load(response)
        if result.get("metadata", {}).get("model") != self.model or result["metadata"].get("auth") != "model_subscription":
            raise RuntimeError("Judge model or authentication does not match the evaluation protocol")
        if kind == "description" and result["metadata"].get("judge_version") != self.version:
            raise RuntimeError("Judge prompt version does not match the scorer")
        if self.version == ATOMIC_JUDGE_VERSION and result["metadata"].get("judge_profile") != HEALTH_PROFILE:
            raise RuntimeError("Judge profile does not match the approved atomic scorer")
        if self.version == REVIEWED_JUDGE_VERSION and (result["metadata"].get("judge_profile") != profile_for(self.version)
                or result["metadata"].get("model_authority") != self.model_authority):
            raise RuntimeError("Reviewed judge profile or explicit model authority audit differs")
        self.audit.append({"kind": kind, "metadata": result["metadata"], **result.get("audit", {})})
        return result["answer"]


def normalized(text):
    return " ".join(text.split())


def checked_decisions(items, expected, text):
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("Judge must encode every supplied criterion exactly once")
    result = {}
    for item in items:
        if (not isinstance(item, dict) or item.get("id") not in expected or item["id"] in result
                or type(item.get("present")) is not bool or not isinstance(item.get("quote"), str)
                or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
            raise ValueError("Invalid judge decision")
        quote = normalized(item["quote"])
        if (item["present"] and not quote) or (quote and quote not in normalized(text)):
            raise ValueError("Judge decision must quote the candidate verbatim")
        result[item["id"]] = {**item, "criterion": expected[item["id"]]}
    return result


def description_evidence(text, event, *, seed):
    order = list(event["rubric"])
    random.Random(seed).shuffle(order)
    traps = [{"id": f"trap-{i + 1:02d}", "text": value} for i, value in enumerate(event["traps"])]
    result = {"description": text, "reference": event["reference"],
        "rubric": order, "traps": traps, "uncertain_claims": event.get("uncertain_claims", []),
        "interpretation_notes": event.get("interpretation_notes", [])}
    for key in ("rubric_semantics", "rubric_weight_ledger"):
        if key in event:
            result[key] = event[key]
    return result


def description_metrics(text, event, judge, *, seed):
    evidence = description_evidence(text, event, seed=seed)
    order, traps = evidence["rubric"], evidence["traps"]
    encoding = judge.evaluate("description", evidence)
    version = getattr(judge, "version", BASE_JUDGE_VERSION)
    validate_comparisons(encoding, version=version)
    if version == REVIEWED_JUDGE_VERSION:
        from mini_bang.rubric_judge.reviewed_controls import validate_scope_ambiguities
        validate_scope_ambiguities(encoding["coverage"])
    coverage = checked_decisions(encoding.get("coverage"), {i["id"]: i["text"] for i in order}, text)
    trap_decisions = checked_decisions(encoding.get("traps"), {i["id"]: i["text"] for i in traps}, text)
    recall = math.fsum(item["weight"] * coverage[item["id"]]["present"] for item in order) / math.fsum(item["weight"] for item in order)
    claims = encoding.get("claims")
    if not isinstance(claims, list) or len(claims) > 40:
        raise ValueError("Judge returned invalid atomic claims")
    references = [event["reference"], *[i["text"] for i in order], *event.get("interpretation_notes", []),
                  *event.get("uncertain_claims", [])]
    for claim in claims:
        if (not isinstance(claim, dict) or claim.get("label") not in LABELS or any(
                not isinstance(claim.get(k), str) or len(claim[k]) > 2000
                for k in ("claim", "quote", "reference_quote", "reason"))):
            raise ValueError("Judge returned invalid evidence claim")
        quote, evidence = normalized(claim["quote"]), normalized(claim["reference_quote"])
        if not quote or quote not in normalized(text):
            raise ValueError("Judge claim must quote the candidate verbatim")
        if evidence and not any(evidence in normalized(r) for r in references):
            raise ValueError("Judge fabricated a reference quote")
        if claim["label"] in {"supported", "contradicted"} and not evidence:
            raise ValueError("A supported or contradicted claim requires explicit reference evidence")
    labels = [claim["label"] for claim in claims]
    supported, contradicted = labels.count("supported"), labels.count("contradicted")
    precision = supported / (supported + contradicted) if supported + contradicted else 0.0
    trap_count = sum(item["present"] for item in trap_decisions.values())
    harmonic = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    result = {"score": max(0.0, harmonic - 0.1 * trap_count), "rubric_recall": recall,
            "factual_precision": precision, "unverifiable_fraction": labels.count("unverifiable") / len(labels) if labels else 0.0,
            "supported_claim_count": supported, "contradicted_claim_count": contradicted,
            "unverifiable_claim_count": labels.count("unverifiable"),
            "reference_checkable_fraction": (supported + contradicted) / len(labels) if labels else 0.0,
            "trap_count": trap_count, "claim_count": len(claims), "coverage": coverage,
            "claims": claims, "traps": trap_decisions, "judge_version": version}
    if version == REVIEWED_JUDGE_VERSION:
        ambiguous = {item["id"] for item in encoding["coverage"] if item["scope_ambiguity"]}
        result.update(scope_ambiguity_policy="fixed_absence_scientific_hold_v1",
            scope_ambiguity_count=len(ambiguous), scope_ambiguity_ids=sorted(ambiguous),
            scope_ambiguity_weight=math.fsum(item["weight"] for item in order if item["id"] in ambiguous),
            scientific_grade_held=bool(ambiguous), score_is_diagnostic=bool(ambiguous))
    if event.get("rubric_version"):
        groups = {}
        for atom in order:
            group = groups.setdefault(atom["source_id"], {"weight": 0.0, "covered_weight": 0.0, "atoms": 0, "covered_atoms": 0})
            present = coverage[atom["id"]]["present"]
            group["weight"] += atom["weight"]
            group["covered_weight"] += atom["weight"] * present
            group["atoms"] += 1
            group["covered_atoms"] += present
        for source_id, group in groups.items():
            members = [atom for atom in order if atom["source_id"] == source_id]
            group["weight"] = math.fsum(atom["weight"] for atom in members)
            group["covered_weight"] = math.fsum(atom["weight"] * coverage[atom["id"]]["present"] for atom in members)
        result["source_criterion_coverage"] = groups
        result["rubric_version"] = event["rubric_version"]
    if event.get("rubric_weight_ledger"):
        ledger = event["rubric_weight_ledger"]
        result["rubric_weight_policy"] = ledger["policy"]
        result["original_rubric_total_weight"] = ledger["original_total"]
        result["scored_rubric_total_weight"] = ledger["proposed_total"]
        result["removed_duplicate_weight"] = ledger["removed_duplicate_weight"]
        result["removed_unsupported_weight"] = ledger.get("removed_unsupported_weight", 0)
        result["removed_redundant_weight"] = ledger.get("removed_redundant_weight", 0)
        result["shared_source_assessments"] = {row["source_id"]: {
            "original_weight": row["original_weight"], "owned_scored_weight": row["proposed_total"],
            "shared_atom_decisions": {key: coverage[key]["present"] for key in row["shared_assessment_ids"]},
            "shared_references_add_no_scoring_weight": True} for row in ledger["source_criteria"]}
    return result


class DescriptionScorer:
    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store, scoring_services=None):
        pool, cases = load_pool(gold_store), gold_store.get("cases")
        try:
            texts = prediction_map(submission.answer, "descriptions", [c["case_id"] for c in cases])
            for value in texts.values():
                if not isinstance(value, str) or not value.strip() or len(value.split()) > 250 or len(value) > 16000:
                    raise ValueError("Descriptions must be non-empty texts of at most 250 words")
        except (ValueError, TypeError) as exc:
            return ValidationResult(False, str(exc), {"score": 0.0, "valid_submission": False})
        rubric_version = gold_store.get("rubric_version")
        context = scoring_services.scoring_context if scoring_services else None
        if rubric_version:
            declared = instance_spec.visible_data
            if (rubric_version != ATOMIC_RUBRIC_VERSION or gold_store.get("judge_version") != ATOMIC_JUDGE_VERSION
                    or declared.get("rubric_version") != rubric_version
                    or declared.get("judge_version") != ATOMIC_JUDGE_VERSION):
                raise ValueError("Public instance and hidden description protocol identities disagree")
            judge = RubricJudgeClient(context, version=ATOMIC_JUDGE_VERSION)
        else:
            # Gold without a rubric version is always scored with the base
            # protocol, whatever default the service or package declares.
            judge = RubricJudgeClient(context, version=BASE_JUDGE_VERSION)
        per_case = []
        for case in cases:
            event = next(event for event in pool["trajectories"][case["trajectory"]]["events"]
                         if event["annotation_id"] == case["annotation_id"])
            if rubric_version:
                event = atomic_event(event, version=rubric_version)
            metrics = description_metrics(texts[case["case_id"]], event, judge, seed=case["case_id"])
            per_case.append({"case_id": case["case_id"], **metrics})
        metrics = {key: sum(c[key] for c in per_case) / len(per_case)
                   for key in ("score", "rubric_recall", "factual_precision", "unverifiable_fraction", "trap_count")}
        for key in ("claim_count", "supported_claim_count", "contradicted_claim_count", "unverifiable_claim_count"):
            metrics[key] = sum(c[key] for c in per_case)
        checkable = metrics["supported_claim_count"] + metrics["contradicted_claim_count"]
        metrics["reference_checkable_fraction"] = checkable / metrics["claim_count"] if metrics["claim_count"] else 0.0
        metrics.update(valid_submission=True, judge_model=judge.model, per_case=per_case)
        metrics["rubric_version"] = rubric_version or "legacy-annotation"
        metrics["judge_version"] = judge.version if hasattr(judge, "version") else BASE_JUDGE_VERSION
        if scoring_services:
            scoring_services.trace.append({"event": "description_judged", "hidden": True, "model": judge.model,
                                           "auth": "model_subscription", "n_cases": len(cases),
                                           "judge_version": judge.version, "judge_audit": judge.audit})
        return ValidationResult(True, "Rubric recall, factual precision and traps judged separately", metrics)
