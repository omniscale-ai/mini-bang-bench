"""Preserve scores while holding source proposals that await human review."""
from dataclasses import replace

from mini_bang.framework.scoring import score_submission
from mini_bang.importing import import_from_path


class SourceAuthorityScorer:
    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store, scoring_services=None):
        authority = gold_store.get("scientific_authority")
        if authority and (authority.get("status") != "pending_human_review"
                or authority.get("scientific_publication_authorized") is not False):
            raise ValueError("Unknown experimental annotation authority; no implicit publication")
        original = import_from_path(task_spec.config["source_authority_base_scorer_class"])()
        result = score_submission(original, task_spec=task_spec, instance_spec=instance_spec,
            submission=submission, run_record=run_record, gold_store=gold_store,
            scoring_services=scoring_services)
        if not authority:
            return result  # Gold without a pending authority record is scored as is.
        return replace(result, metrics={**(result.metrics or {}), "scientific_grade_held": True,
            "score_is_diagnostic": True, "scientific_publication_authorized": False,
            "annotation_authority_status": "pending_human_review",
            "annotation_authority": authority})
