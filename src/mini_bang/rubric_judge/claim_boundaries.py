"""Verbatim claim fidelity and explicit-unit guards."""
from copy import deepcopy
import re
import unicodedata

from mini_bang.rubric_judge import lexicon
from mini_bang.rubric_judge.base_contract import object_schema, STRING
from mini_bang.rubric_judge.quantity_syntax import explicit_quantity_witness, same_head_existence, same_quantity_kind
from mini_bang.rubric_judge.quantity_syntax import POPULATION_EXCLUSIVE, exclusive_existence_subject, exclusive_subject_quantity
from mini_bang.rubric_judge.quantity_syntax import numeric_projection

RIGHT_QUOTE = "\N{RIGHT SINGLE QUOTATION MARK}"

UNIT = {"type": "string", "enum": ["coherent_organisms", "organism_parts", "groups",
    "population_types", "visual_structures_unspecified", "activity_events", "spatial_patterns",
    "numerical_channel_values", "time", "other_explicit_unit", "not_a_count", "unresolved"]}
ALIGNMENT = {"type": "string", "enum": ["same", "different", "unresolved", "not_applicable"]}
COMPARISON = object_schema({"is_numerical_count_assertion": {"type": "boolean"},
    "candidate_unit": UNIT, "reference_unit": UNIT, "counting_unit": UNIT, "unit_relation": ALIGNMENT,
    "entity_relation": ALIGNMENT, "phase_relation": ALIGNMENT,
    "candidate_unit_quote": STRING, "reference_unit_quote": STRING})
RULES = """CLAIM FIDELITY BOUNDARY:
Every contradicted claim must itself be a contiguous VERBATIM candidate quotation: claim == quote exactly.
Supported/unverifiable claims may normalize an independently asserted atomic proposition from a longer
compound quotation or resolve its pronouns. Preserve every relevant qualification, negation and temporal
context. Do not turn a compound containing an unsupported component into a supported whole claim.
Every quotation must remain contiguous VERBATIM candidate text. Every number, quantifier, comparator,
contrast or negation in an atomic paraphrase must be conserved by that quote (ordinary synonyms such
as """ + " and ".join(["pair/two", *lexicon.items("synonym_examples")]) + """ are allowed). Do not add rather-than, exactly, only or a new negation from gold.
Do not add a contrast, count, causal link, implication or interpretation from the reference. In particular,
the reference's different number is not something the candidate asserted. Keep the text of a quoted
assertion separate from the evidence used to classify it. Do not turn a noun phrase into a new assertion.

For every claim fill count_comparison. Count assertions need an explicit counting unit, entity and phase
in BOTH accounts before contradiction is possible. Structure, blob or part count does not establish
organism count. A generic pair of visible structures can be two parts of one organism. Do not assign a
biological unit from color alone or from the number you want to compare. If the unit is unresolved, such
a numerical contradiction is not established; retain unverifiable. Other explicit quantitative facts and
genuine equal-unit/equal-entity/equal-phase incompatible counts still require contradicted. A different
phase or actor is not a contradiction. The candidate_unit_quote must be verbatim candidate evidence;
reference_unit_quote must be verbatim reference/rubric/notes evidence. For a non-count
assertion ONLY counting_unit takes not_a_count; candidate_unit and reference_unit
still name the actual actor units when identity alignment is needed.
Do not invent units, quotes or missing source facts merely to retain a label.
A count contradiction requires an explicit quantity tied to the proposed counting unit in
each unit quotation. A time stamp is not an organism count. A group is not automatically
several coherent organisms; plural bodies in an earlier phase do not fix a later group's
organism count. Do not import 'only one' from the reference into a candidate claim about
localization or persistence. Quote compatible explicit count/unit/phase evidence from BOTH
accounts, or keep the count inference unverifiable. Preserve genuine incompatible explicit
counts when independent context establishes the same unit, actor and phase. Localization
and movement remain separate predicates."""

NUMBER_WORDS = {value: lexicon.words("number_words", str(value), base=english) for value, english in {
    0: "zero", 1: "one", 2: "two pair paired", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten"}.items()}
NUMBER_WORDS[1] = lexicon.words("number_one_emphatic", base=NUMBER_WORDS[1] + " a_single")
PHRASES = [
    ("lower_bound", r"\bat least\b|\bno fewer than\b" + lexicon.alternatives("bound_phrases", "lower") + r"|or more\b"),
    ("upper_bound", r"\bat most\b|\bno more than\b" + lexicon.alternatives("bound_phrases", "upper") + r"|or fewer\b"),
    ("contrast", r"\brather than\b|\binstead of\b" + lexicon.alternatives("contrast_phrases")),
]
WORDS = {feature: lexicon.words("feature_words", feature, base=english) for feature, english in {
    "negation": "not no none never neither without cannot can't isn't aren't wasn't weren't doesn't don't",
    "exact": "exactly precisely",
    "only": "only solely",
    "all": "all every each",
    "most": "most majority",
    "some": "some",
    "few": "few",
    "almost": "almost nearly",
    "increase": "more greater higher increased increases increasing increase stronger",
    "decrease": "less fewer lower reduced decreases decreased decreasing decrease",
}.items()}
WORDS["some"] += " several multiple"


def protected_content(text):
    text = lexicon.fold(unicodedata.normalize("NFKC", text).casefold()).replace(RIGHT_QUOTE, "'")
    numbers = numeric_projection(text, NUMBER_WORDS)
    features = set()
    for feature, pattern in PHRASES:
        if re.search(pattern, text):
            features.add(feature)
            text = re.sub(pattern, " ", text)
    text = re.sub(r"\ba single\b", "a_single", text)
    tokens = re.findall(r"[\w']+", text)
    for feature, words in WORDS.items():
        equivalents = set(lexicon.fold(words).split())
        if any(token in equivalents for token in tokens):
            features.add(feature)
    return numbers, features


def schema_for(base):
    schema = deepcopy(base)
    claim = schema["properties"]["claims"]["items"]
    claim["properties"]["count_comparison"] = COMPARISON
    claim["required"].append("count_comparison")
    return schema


BATTERY_RULES = ("\nFor this classification battery, do not extract or rewrite supplied claims. "
    "Return count_comparison=null for coverage decisions; for every evidence/claim decision return "
    "the complete count_comparison object. For non-count claims only counting_unit uses not_a_count; "
    "candidate_unit and reference_unit retain actual actor units when alignment is needed. The explicit-unit "
    "contradiction guard applies equally to these already supplied claims. No expected labels are given.")


def battery_schema(base):
    schema = deepcopy(base)
    decision = schema["properties"]["decisions"]["items"]
    decision["properties"]["count_comparison"] = {"anyOf": [{"type": "null"}, COMPARISON]}
    decision["required"].append("count_comparison")
    return schema


COLLECTIVE_NOUN = (r"(?:groups?|clusters?|communit(?:y|ies)|collections?|flocks?|bodies|organisms|cells"
    + lexicon.alternatives("collective_nouns") + ")")
PROPERTY = r"(?:shape|color|colour|outline|size|brightness|form" + lexicon.alternatives("property_nouns") + ")"
PREDICATE = (r"(?:remain\w*|stay\w*|persist\w*|gather\w*|move\w*|appear\w*|surviv\w*|exist\w*|occup\w*|is|are|was|were"
    + lexicon.alternatives("collective_predicates") + ")")
COLLECTIVE_SUBJECT = re.compile(r"(?:^|[.;:,\n])\s*(?:(?:the|a|an|this|that|these|those|their|its)\s+)?"
    r"(?:(?!" + PROPERTY + r"\b)[\w-]+\s+){0,5}" + COLLECTIVE_NOUN +
    r"(?!-)\s+(?:of\s+(?:[\w-]+\s+){1,8})?" + PREDICATE + r"\b", re.IGNORECASE)
COLLECTIVE_PROPERTY = re.compile(r"\b" + PROPERTY + r"\s+(?:of\s+(?:the\s+)?)?"
    r"(?:[\w-]+\s+){0,4}" + COLLECTIVE_NOUN + r"\b(?!-)", re.IGNORECASE)
COLLECTIVE_POSSESSIVE = re.compile(r"\b" + COLLECTIVE_NOUN + r"(?:'s|'|" + RIGHT_QUOTE + r"s|" + RIGHT_QUOTE + r")\s+"
    + PROPERTY + r"\b", re.IGNORECASE)
DOMAIN_NOUN = re.compile(r"\b(?:organisms?|body|bodies|cells?|units?|entities|entity"
    + lexicon.alternatives("domain_nouns") + r")\b", re.IGNORECASE)
SINGULAR = re.compile(r"\b(?:one" + lexicon.alternatives("singular_words") + r")\b", re.IGNORECASE)
INCIDENTAL_ONE = re.compile(r"\s+(?:side|point(?!-like)|place|end|corner|moment|time|occasion|step|frame|direction|aspect|hand"
    + lexicon.alternatives("incidental_one_nouns") + r")\b(?!-)", re.IGNORECASE)
EXISTENCE_FORMS = {
    "remain": r"remain(?:s)?" + lexicon.alternatives("existence_forms", "remain"),
    "survive": r"surviv(?:e|es)" + lexicon.alternatives("existence_forms", "survive"),
    "persist": r"persist(?:s)?" + lexicon.alternatives("existence_forms", "persist"),
    "exist": (r"exist(?:s)?" + lexicon.alternatives("existence_forms", "exist")
        + r"|(?:is|are) present" + lexicon.alternatives("existence_forms", "present")),
}
EXCLUSIVE_SUBJECT = re.compile(r"^\s*(?:the\s+)?" + POPULATION_EXCLUSIVE, re.IGNORECASE)


def distinct_exclusive_scope(claim, comparison):
    """An exclusion counterexample need not be the same individual.

    A quantity inside that subject still needs the strict count path when the
    alleged conflict is only the same affirmative existence predicate.
    """
    text = claim.get("claim", "")
    exclusive = EXCLUSIVE_SUBJECT.search(text) or exclusive_existence_subject(text, EXISTENCE_FORMS, NUMBER_WORDS)
    return (comparison["entity_relation"] == "different" and bool(exclusive)
        and (not matching_existence_predicate(claim, comparison)
            or not exclusive_subject_quantity(text, EXISTENCE_FORMS, NUMBER_WORDS)))


def pure_existence_clause(text, *, collective, complement=False):
    """Recognize a complete affirmative actor/existence proposition.

    Optional complements are returned intact for literal comparison. Negation
    and coordination stay outside this check. Matching returns syntax, never a factual S/C/U decision.
    Unrecognized paraphrases stay the semantic judge's responsibility.
    """
    text = lexicon.fold(unicodedata.normalize("NFKC", text).casefold()).strip()
    if re.search(r"\b(?:not|no|never" + lexicon.alternatives("negation_particles") + r"|and|or|but"
                 + lexicon.alternatives("coordinators") + r")\b", text):
        return None
    actor = COLLECTIVE_NOUN if collective else DOMAIN_NOUN.pattern.removeprefix(r"\b").removesuffix(r"\b")
    prefix = r"(?:(?:the|these|those)\s+)?" if collective else (
        r"(?:(?:only\s+)?(?:one" + lexicon.alternatives("singular_subject_words") + r")\s+)")
    # Whole-string anchoring prevents a number in another clause from becoming
    # the subject quantity, and prevents a reference disappearance from matching.
    subject = prefix + r"(?:[\w-]+\s+){0,8}" + actor
    if collective:
        subject = r"(?:(?:the|these|those)\s+)?(?:[\w-]+\s+){0,8}" + actor
    for lemma, forms in EXISTENCE_FORMS.items():
        tail = r"(?P<tail>[^.!?;:\n]*)" if complement else ""
        match = re.fullmatch(subject + r"\s+(?:" + forms + r")\b" + tail + r"[.!?\s]*", text)
        if match:
            return (lemma, " ".join(match['tail'].split())) if complement else lemma
    return None


def matching_existence_predicate(claim, comparison):
    """Require the same predicate AND literal complement; never erase location.

    Different/negated predicates and different complements stay semantic work.
    This also catches count-only claims with an identical location qualifier.
    """
    candidate = pure_existence_clause(claim.get("claim", ""), collective=True, complement=True)
    return any(same_head_existence(claim.get("claim", ""), text, EXISTENCE_FORMS, NUMBER_WORDS) or (
        candidate is not None and pure_existence_clause(text, collective=False, complement=True) == candidate)
        for text in (claim.get("reference_quote", ""), comparison["reference_unit_quote"]))


def pure_cardinality_mismatch(claim, comparison):
    """Block false-count C for the same positive existence predicate.

    Unit relabeling cannot make this pure plural-versus-one comparison into a
    property conflict. A real negative/different property is never matched.
    """
    if claim["label"] != "contradicted" or comparison["is_numerical_count_assertion"]:
        return False
    if distinct_exclusive_scope(claim, comparison):
        return False
    return matching_existence_predicate(claim, comparison)


def collective_subject(text):
    """Conservative predicate-subject syntax, not a biological count inference."""
    text = lexicon.fold(unicodedata.normalize("NFKC", text))
    return bool(COLLECTIVE_SUBJECT.search(text) or COLLECTIVE_PROPERTY.search(text)
        or COLLECTIVE_POSSESSIVE.search(text))


def singular_domain_witness(text):
    """Detect one domain member, including anaphoric 'organisms: one remains'.

    One group of bodies counts the group, not one body. This necessary wording
    guard does not resolve identity, parts, phase or the full source entailment.
    """
    text = lexicon.fold(unicodedata.normalize("NFKC", text).casefold())
    if not DOMAIN_NOUN.search(text):
        return False
    for match in SINGULAR.finditer(text):
        tail = text[match.end():]
        if (re.match(r"\s+another\b", tail) or INCIDENTAL_ONE.match(tail)
                or re.search(r"\b(?:no|not" + lexicon.alternatives("preceding_negation") + r")\s*$", text[:match.start()])):
            continue
        # Do not collapse a counted collective or part into its whole members.
        prefix = re.split(r"[.;:,]", tail, maxsplit=1)[0]
        words = re.findall(r"[\w-]+", prefix)[:5]
        noun = next((word for word in words if DOMAIN_NOUN.fullmatch(word)), None)
        before_noun = words[:words.index(noun)] if noun else words[:3]
        if any(re.fullmatch(COLLECTIVE_NOUN + r"|parts?|lobes?" + lexicon.alternatives("collective_part_nouns"), word)
               for word in before_noun):
            continue
        return True
    return False


def validate_collective_support(claim, data, comparison):
    """Reject missing alignment evidence, never manufacture a factual label.

    Already-unverifiable encodings stay conservative. The source guard
    prevents unsafe support or contradiction through not_a_count.
    """
    texts = (claim.get("claim", ""), claim.get("quote", ""))
    if (claim["label"] not in {"supported", "contradicted"}
            or not any(collective_subject(text) for text in texts)):
        return
    witnesses = (claim.get("reference_quote", ""), comparison["reference_unit_quote"])
    if not any(singular_domain_witness(witness) for witness in witnesses):
        return
    uncertain = {"unresolved", "not_a_count"}
    # An exclusive population assertion can be refuted by a distinct member of
    # that SAME population scope. It does not assert individual actor identity.
    # This admits an honest encoding, never supplies its factual label. The
    # semantic judge must still establish population scope and incompatibility.
    if (claim["label"] == "contradicted" and not comparison["is_numerical_count_assertion"]
            and distinct_exclusive_scope(claim, comparison)
            and comparison["candidate_unit"] == comparison["reference_unit"] not in uncertain
            and comparison["unit_relation"] == comparison["phase_relation"] == "same"
            and comparison["entity_relation"] == "different"
            and comparison["counting_unit"] == "not_a_count"
            and comparison["candidate_unit_quote"] and comparison["reference_unit_quote"]):
        return
    if (any(comparison[key] in uncertain for key in ("candidate_unit", "reference_unit"))
            or comparison["unit_relation"] not in {"same", "different"}
            or comparison["entity_relation"] != "same" or comparison["phase_relation"] != "same"
            or not comparison["candidate_unit_quote"] or not comparison["reference_unit_quote"]):
        raise ValueError("Truth-asserting collective-versus-single claim requires explicit identity/unit/phase evidence; a property may remain non-count and must not invent numerical units")
    if not comparison["is_numerical_count_assertion"] and comparison["counting_unit"] != "not_a_count":
        raise ValueError("A non-count property identity comparison must keep counting_unit=not_a_count")


def validate_count_comparison(claim, data):
    references = [data["reference"], *[v["text"] for v in data["rubric"]],
                  *data.get("interpretation_notes", []), *data.get("uncertain_claims", [])]
    comparison = claim.get("count_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("Every classified claim requires a counting-unit comparison")
    candidate_unit = comparison["candidate_unit_quote"]
    reference_unit = comparison["reference_unit_quote"]
    if candidate_unit and candidate_unit not in data["description"]:
        raise ValueError("Count unit was not quoted from the candidate")
    quote = claim.get("quote", "")
    if (claim["label"] == "contradicted" and comparison["is_numerical_count_assertion"]
            and candidate_unit and candidate_unit not in quote and (not quote or quote not in candidate_unit)):
        raise ValueError("Candidate quantity witness must belong to the classified claim quotation")
    if (claim["label"] == "contradicted" and comparison["is_numerical_count_assertion"]
            and reference_unit and "reference_quote" in claim and reference_unit not in claim["reference_quote"]):
        raise ValueError("Reference quantity witness must belong to the classified claim reference quotation")
    if reference_unit and not any(reference_unit in text for text in references):
        raise ValueError("Reference count unit lacks contiguous source evidence")
    if (claim["label"] == "contradicted" and not comparison["is_numerical_count_assertion"]
            and distinct_exclusive_scope(claim, comparison)):
        if (comparison['candidate_unit'] != comparison['reference_unit']
                or comparison['candidate_unit'] in {'unresolved', 'not_a_count'}
                or comparison['unit_relation'] != 'same' or comparison['phase_relation'] != 'same'
                or comparison['counting_unit'] != 'not_a_count' or not candidate_unit or not reference_unit):
            raise ValueError("An exclusive counterexample requires actual same identity/unit/phase scope and literal unit witnesses; different individual identity does not permit different phase")
    validate_collective_support(claim, data, comparison)
    if pure_cardinality_mismatch(claim, comparison):
        raise ValueError("A pure collective-versus-single cardinality contradiction has the same positive existence predicate on both sides; explicit quantities must use the strict count path, while plurality or same-actor scope alone does not prove contradiction")
    if claim["label"] != "contradicted" or not comparison["is_numerical_count_assertion"]:
        return
    uncertain = {"unresolved", "not_a_count"}
    if (comparison["candidate_unit"] in uncertain or comparison["reference_unit"] in uncertain
            or comparison["candidate_unit"] != comparison["reference_unit"]
            or comparison["counting_unit"] != comparison["candidate_unit"]
            or any(comparison[key] != "same" for key in ("unit_relation", "entity_relation", "phase_relation"))
            or not candidate_unit or not reference_unit):
        raise ValueError("A numerical contradiction requires the same established unit, entity and phase")
    local_candidate = candidate_unit if candidate_unit in quote else quote
    if (not all(explicit_count_witness(v, comparison["counting_unit"]) for v in (local_candidate, reference_unit))
            or not same_quantity_kind(local_candidate, reference_unit, comparison["counting_unit"], NUMBER_WORDS)):
        raise ValueError("A numerical count contradiction requires an explicit quantity tied to the proposed counting-unit nominal phrase in BOTH unit witnesses; incidental quantities and timestamps alone are insufficient")


def explicit_organism_count(text):
    """Necessary organism-count syntax boundary."""
    return explicit_count_witness(text, "coherent_organisms")


def explicit_count_witness(text, counting_unit):
    """Unit-aware nominal, scalar, duration/timestamp or event-frequency syntax."""
    return explicit_quantity_witness(text, counting_unit, NUMBER_WORDS)


def claim_boundary_issues(claim, data):
    """Return all independent defects without changing any acceptance condition."""
    issues = []
    quote = claim["quote"]
    if not quote or quote not in data["description"]:
        issues.append("A claim requires a contiguous verbatim candidate quotation")
    if claim["label"] == "contradicted" and claim["claim"] != quote:
        issues.append("A contradicted claim must equal its verbatim candidate quotation")
    claim_numbers, claim_features = protected_content(claim["claim"])
    quote_numbers, quote_features = protected_content(quote)
    if claim_numbers - quote_numbers or not claim_features <= quote_features:
        issues.append("Atomic paraphrase added a number, quantifier, comparator, contrast or negation absent from its quote"
            + "; absent numbers=" + str(dict(claim_numbers - quote_numbers))
            + "; absent features=" + str(sorted(claim_features - quote_features)))
    try:
        validate_count_comparison(claim, data)
    except ValueError as error:
        issues.append(str(error))
    return issues


def validate_claim_boundaries(answer, data):
    for claim in answer["claims"]:
        issues = claim_boundary_issues(claim, data)
        if issues:
            raise ValueError(issues[0])
