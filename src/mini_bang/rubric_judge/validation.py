"""Structural checks of judge encodings; they never substitute semantic labels."""
from mini_bang.rubric_judge.contract import COMPARISON_JUDGE_VERSION


def validate_comparisons(answer, *, version):
    if version != COMPARISON_JUDGE_VERSION:
        return
    if not isinstance(answer, dict) or not isinstance(answer.get("claims"), list) or len(answer["claims"]) > 40:
        raise ValueError("Comparison judge requires a bounded list of claims")
    expected = {"supported": "entailed", "contradicted": "incompatible", "unverifiable": "not_established"}
    fields = {"entity", "property", "phase", "scope", "counting_unit", "reference_property", "relation"}
    seen = set()
    for claim in answer.get("claims", []):
        if not isinstance(claim, dict) or claim.get("label") not in expected:
            raise ValueError("Invalid comparison judge claim")
        comparison = claim.get("comparison")
        if not isinstance(comparison, dict) or set(comparison) != fields:
            raise ValueError("Comparison judge requires an explicit proposition comparison")
        if any(not isinstance(value, str) or len(value) > 2000 for value in comparison.values()):
            raise ValueError("Invalid judge proposition comparison")
        if comparison["relation"] != expected.get(claim.get("label")):
            raise ValueError("Judge label does not match its proposition comparison")
        if claim["label"] != "contradicted":
            continue
        if any(not comparison[key].strip() for key in fields):
            raise ValueError("Contradiction requires aligned entity, property, phase, scope and unit")
        if comparison["counting_unit"].strip().lower() in {"unknown", "unresolved", "ambiguous"}:
            raise ValueError("An unresolved counting unit cannot justify a contradiction")
        normalized = lambda value: " ".join(value.lower().split())
        if normalized(comparison["property"]) == normalized(comparison["reference_property"]):
            raise ValueError("Identical predicates cannot justify a contradiction")
        key = tuple(normalized(comparison[field]) for field in sorted(fields))
        if key in seen:
            raise ValueError("Judge duplicated the same incompatibility")
        seen.add(key)


def evidence_quote_issues(answer, data):
    """Index every defect, so one repair can address independent problems."""
    issues = []
    normalize = lambda value: " ".join(value.split())
    candidate = normalize(data.get("description", ""))
    references = [data.get("reference", ""), *[v["text"] for v in data.get("rubric", [])],
                  *data.get("interpretation_notes", []), *data.get("uncertain_claims", [])]
    for name in ("coverage", "traps"):
        expected = {v["id"] for v in data.get("rubric" if name == "coverage" else "traps", [])}
        decisions = answer.get(name)
        if not isinstance(decisions, list) or len(decisions) != len(expected):
            issues.append((name, "Every criterion must have exactly one decision"))
            continue
        if {v.get("id") for v in decisions} != expected:
            issues.append((name, "Unknown or duplicated criterion decision"))
        for index, decision in enumerate(decisions):
            quote = normalize(decision.get("quote", ""))
            if decision.get("present") and not quote or quote and quote not in candidate:
                issues.append((f"{name}[{index}].quote", "Decision quotation is not verbatim candidate evidence"))
    for index, claim in enumerate(answer["claims"]):
        quote = normalize(claim.get("quote", ""))
        reference = normalize(claim.get("reference_quote", ""))
        if not quote or quote not in candidate:
            issues.append((f"claims[{index}].quote", "Claim quotation is not verbatim candidate evidence"))
        if reference and not any(reference in normalize(v) for v in references):
            issues.append((f"claims[{index}].reference_quote", "Reference quotation is not one contiguous supplied passage"))
        if claim["label"] in {"supported", "contradicted"} and not reference:
            issues.append((f"claims[{index}].reference_quote", "A checkable claim needs a verbatim reference quotation"))
    return issues


def validate_evidence_quotes(answer, data):
    """Raise the first quotation defect reported by evidence_quote_issues."""
    issues = evidence_quote_issues(answer, data)
    if issues:
        raise ValueError(issues[0][1])


class AtomicEncodingError(ValueError):
    """A completed encoding needs its one permitted comprehensive format repair."""


def validate_atomic_encoding(answer, data, *, require_epistemic_witness=False):
    from mini_bang.rubric_judge.claim_boundaries import claim_boundary_issues
    from mini_bang.rubric_judge.entities import validate_alignments
    issues = evidence_quote_issues(answer, data)
    for index, decision in enumerate(answer["coverage"]):
        try:
            validate_alignments({"coverage": [decision]})
        except ValueError as error:
            issues.append((f"coverage[{index}].entity_alignment", str(error)))
    for index, claim in enumerate(answer["claims"]):
        issues.extend((f"claims[{index}]", issue) for issue in claim_boundary_issues(claim, data))
        if require_epistemic_witness:
            from mini_bang.rubric_judge.epistemic_support import epistemic_support_issues
            issues.extend((f"claims[{index}].reference_quote", issue)
                          for issue in epistemic_support_issues(claim, data))
    if issues:
        raise AtomicEncodingError("\n".join(path + ": " + message for path, message in issues))


FORMAT_REPAIR_GUIDANCE = (
    "\nAddress EVERY indexed defect in this ONE repair. The indices are zero-based; they "
    "identify format/evidence-boundary failures, not target semantic labels. When a claim "
    "retains a time, number or qualifier stated in a preceding sentence or paragraph heading, "
    "expand its contiguous candidate quote to include that source context. Do not borrow "
    "context from the reference. If an ordinary synonym is rejected by the finite feature "
    "guard, keep the original candidate wording and qualification in the atomic claim. "
    "Do not drop claims or semantic qualifications merely to evade validation. Recheck ALL "
    "claims and citations, including items without an indexed error, before returning. "
    "Keep genuine same-unit contradictions; never change labels merely to pass formatting.\n"
)
