from copy import deepcopy
import pytest

from mini_bang.rubric_judge.claim_boundaries import validate_claim_boundaries, protected_content


def assertion(text, label="contradicted"):
    return {"claim": text, "quote": text, "label": label, "count_comparison": {
        "is_numerical_count_assertion": True, "candidate_unit": "coherent_organisms",
        "reference_unit": "coherent_organisms", "counting_unit": "coherent_organisms",
        "unit_relation": "same", "entity_relation": "same",
        "phase_relation": "same", "candidate_unit_quote": "organisms", "reference_unit_quote": "organism"}}


def test_reference_dependent_contrast_is_not_a_candidate_claim():
    text = "A pair stays in the corner."
    claim = assertion(text)
    claim["claim"] = "Two organisms, rather than exactly one, stay in the corner."
    with pytest.raises(ValueError, match="verbatim"):
        validate_claim_boundaries({"claims": [claim]}, {"description": text,
            "reference": "One organism stays in the corner.", "rubric": []})


@pytest.mark.parametrize("field,value", [("candidate_unit", "organism_parts"),
    ("candidate_unit", "visual_structures_unspecified"), ("unit_relation", "unresolved"),
    ("entity_relation", "different"), ("phase_relation", "different")])
def test_incompatible_unit_entity_or_phase_cannot_be_a_numerical_contradiction(field, value):
    text = "Two organisms remain."
    claim = assertion(text)
    claim["count_comparison"][field] = value
    with pytest.raises(ValueError, match="same established unit"):
        validate_claim_boundaries({"claims": [claim]}, {"description": text,
            "reference": "Exactly one organism remains.", "rubric": []})


def test_genuine_same_unit_contradiction_remains_available():
    text = "Two organisms remain."
    claim = assertion(text)
    claim["count_comparison"].update(candidate_unit_quote="Two organisms", reference_unit_quote="one organism")
    validate_claim_boundaries({"claims": [claim]}, {"description": text,
        "reference": "Exactly one organism remains.", "rubric": []})


def test_unknown_unit_can_be_unverifiable_without_changing_the_assertion():
    text = "Two structures remain."
    claim = assertion(text, "unverifiable")
    claim["count_comparison"].update(candidate_unit="visual_structures_unspecified",
        candidate_unit_quote="structures", unit_relation="unresolved")
    validate_claim_boundaries({"claims": [claim]}, {"description": text,
        "reference": "One organism remains.", "rubric": []})


@pytest.mark.parametrize("paraphrase,quote", [("Two lobes remain", "A pair of lobes remain"),
    ("At least two rovers", "No fewer than 2 rovers")])
def test_quantifier_guard_recognizes_normalized_equivalents(paraphrase, quote):
    numbers, features = protected_content(paraphrase)
    source_numbers, source_features = protected_content(quote)
    assert not numbers - source_numbers
    assert features <= source_features


def test_atomic_supported_paraphrase_can_resolve_a_pronoun_without_copying_the_whole_sentence():
    data = {"description": "A red rover is present. It moves while a blue part flexes.",
            "reference": "A red rover moves.", "rubric": []}
    value = assertion("It moves while a blue part flexes.", "supported")
    value["claim"] = "The red rover moves."
    value["count_comparison"].update(is_numerical_count_assertion=False,
        candidate_unit="not_a_count", reference_unit="not_a_count", counting_unit="not_a_count",
        candidate_unit_quote="", reference_unit_quote="", unit_relation="not_applicable")
    validate_claim_boundaries({"claims": [value]}, data)


def test_supported_paraphrase_cannot_add_reference_only_quantifiers():
    data = {"description": "A pair of lobes remain.", "reference": "Only one organism remains.", "rubric": []}
    value = assertion(data["description"], "supported")
    value["claim"] = "Exactly one organism remains rather than a pair."
    with pytest.raises(ValueError, match="added a number"):
        validate_claim_boundaries({"claims": [value]}, data)
