"""Quote/unit validation for supplied tests, without access to their private oracle."""
from mini_bang.rubric_judge.claim_boundaries import claim_boundary_issues
from mini_bang.rubric_judge.validation import AtomicEncodingError
from mini_bang.rubric_judge.epistemic_support import epistemic_support_issues

FIELDS = {"id", "kind", "reference", "interpretation_notes", "uncertain_claims",
          "criterion", "candidate", "claim"}


def validate_scope_ambiguities(rows, *, controls=False, cases=None):
    for index, row in enumerate(rows):
        flag, reason = row.get("scope_ambiguity"), row.get("scope_ambiguity_reason")
        if type(flag) is not bool or not isinstance(reason, str):
            raise AtomicEncodingError(f"coverage[{index}]: explicit source ambiguity fields required")
        if flag and (not reason.strip() or (row.get("result") != "absent" if controls else row.get("present") is not False)):
            raise AtomicEncodingError(f"coverage[{index}]: source ambiguity requires explained fixed absence")
        if not flag and reason.strip():
            raise AtomicEncodingError(f"coverage[{index}]: clear source scope requires empty ambiguity reason")
        if controls and cases[row["id"]]["kind"] != "coverage" and flag:
            raise AtomicEncodingError(f"decisions[{index}]: evidence uncertainty uses its factual label")


def model_cases(cases):
    """Never copy a fixture's expected labels, prior outputs, scores or provenance."""
    return [{key: value for key, value in case.items() if key in FIELDS} for case in cases]


def validate_controls(answer, data):
    cases = {row["id"]: row for row in data["cases"]}
    rows = answer.get("decisions")
    if not isinstance(rows, list) or len(rows) != len(cases) or {v.get("id") for v in rows} != set(cases):
        raise AtomicEncodingError("decisions: every supplied case id must occur exactly once")
    validate_scope_ambiguities(rows, controls=True, cases=cases)
    issues = []
    for index, row in enumerate(rows):
        case = cases[row["id"]]
        kind, result = case["kind"], row["result"]
        allowed = {"present", "absent"} if kind == "coverage" else {"supported", "contradicted", "unverifiable"}
        if result not in allowed:
            issues.append(f"decisions[{index}]: result is not valid for this case kind")
            continue
        quote = row["quote"]
        if quote and quote not in case["candidate"] or result in {"present", "supported", "contradicted"} and not quote:
            issues.append(f"decisions[{index}].quote: a contiguous verbatim candidate quote is required")
        references = [case["reference"], case.get("criterion", ""),
                      *case.get("interpretation_notes", []), *case.get("uncertain_claims", [])]
        reference = row["reference_quote"]
        if reference and not any(reference in text for text in references):
            issues.append(f"decisions[{index}].reference_quote: quotation is not a contiguous supplied passage")
        if result in {"supported", "contradicted"} and not reference:
            issues.append(f"decisions[{index}].reference_quote: checkable claims require authoritative evidence")
        if kind == "coverage":
            if row["count_comparison"] is not None:
                issues.append(f"decisions[{index}].count_comparison: coverage uses null")
        else:
            evidence = {"description": case["candidate"], "reference": case["reference"],
                "rubric": [], "interpretation_notes": case.get("interpretation_notes", []),
                "uncertain_claims": case.get("uncertain_claims", [])}
            claim = {"claim": case["claim"], "quote": quote, "label": result,
                     "count_comparison": row["count_comparison"], "reference_quote": reference}
            issues.extend(f"decisions[{index}]: " + message for message in claim_boundary_issues(claim, evidence))
            issues.extend(f"decisions[{index}].reference_quote: " + message
                          for message in epistemic_support_issues(claim, evidence))
    if issues:
        raise AtomicEncodingError("\n".join(issues))
