"""Atom-level rubric semantics; the hidden registry preserves the source criterion weights."""
from copy import deepcopy

from mini_bang.rubric_judge.minimal_contract import PROMPTS as BASE_PROMPTS, SCHEMAS as BASE_SCHEMAS
from mini_bang.rubric_judge.base_contract import object_schema, STRING
from mini_bang.rubric_judge.claim_boundaries import RULES as CLAIM_RULES, schema_for

ATOMIC_RULES = (
    "\nATOMIC RUBRIC PROTOCOL:\n"
    "Each supplied rubric item is one separately scored fact. Judge that atom only. Its source_id "
    "is provenance, not a request to grade the entire former compound criterion. A missing fact "
    "in another atom must not cancel a correctly communicated current fact. Preserve the atom's "
    "own entity, phase, quantifier, degree, relation and negation. Resolve pronouns from the whole "
    "candidate and reference; do not require the same noun or surface wording.\n"
    "Before awarding coverage, identify the semantic dimension of its predicate. Frequency, "
    "brightness/amplitude, spatial extent, relative population share, spatial order and number "
    "of organisms are different dimensions. A later activity being recurrent, strong or dominant "
    "does not by itself establish that its frequency increased from before. A before/after "
    "description can establish an increase without the word 'increase', but only if it actually "
    "communicates that comparison on the SAME dimension. Continued activity is not elevated activity.\n"
    "Use entailment rather than plausible completion: a shape nickname supplies an appearance, "
    "not every physical part, gap, membrane, connection or coordinated motion associated with "
    "that nickname. Do not invent independent components or object identities. In a genuine "
    "disjunction A-or-B, a fact follows only if BOTH alternatives establish it; do not pick the "
    "convenient alternative and discard the other. Hedging is not a positive assertion.\n"
    "For each positive coverage decision, briefly state the matching proposition and quote "
    "candidate evidence. For a negative decision, name the particular missing semantic fact, "
    "not merely a missing word. For a contradiction, name the explicit incompatible proposition; "
    "never use low recall, an unfamiliar paraphrase or a missing detail as the contradiction.\n"
)
ENTITY_RULES = (
    "\nENTITY ALIGNMENT AND PREDICATE CHECK:\n"
    "The reference and candidate entity ledgers were extracted separately; neither saw the "
    "other text. They are aids, never new factual authority. First align each current atom's "
    "subject using independently stated role, phase, location or an explicit coreference chain. "
    "A background field is not automatically an organism. Shared color/change alone cannot "
    "establish this alignment. Mark genuinely unresolved identity unresolved, explain the two "
    "possible readings, and do not award coverage. Do not infer alignment from the desired fact.\n"
    "Separate the atom's target predicate from its entity description. A modifier separately "
    "scored by another supplied atom is entity context here, not a duplicate requirement to "
    "repeat that shape/color/count word. Once independent role/phase/coreference establishes "
    "the same actor, a missing secondary nickname must not erase its clearly stated location, "
    "color or motion. If the shape/number IS the current target predicate, it is required. "
    "No partial match to the target predicate suffices.\n"
    "For every coverage item encode entity_alignment and predicate_check. Compare the candidate "
    "and required predicate on the SAME semantic dimension, phase and scope. For an alleged "
    "increase, quote what communicates lower-before/higher-after on that dimension: a conspicuous "
    "or dominant later pattern alone has a countermodel with unchanged frequency. A stronger "
    "covered atom logically entails its weaker atom on the same actor/phase/dimension; check "
    "the complete decision set for inconsistencies before returning it. For every positive "
    "decision, try a concrete reading where its cited candidate remains true and the required "
    "predicate is false. If such a countermodel survives the candidate's actual assertions, "
    "mark absent. Do not borrow a missing premise from the gold reference.\n"
)
REFERENTIAL_RULES = (
    "\nORDINARY REFERENTIAL CORRESPONDENCE:\n"
    "These are two accounts of the SAME observed scene, not unrelated possible worlds. "
    "An ordinary unambiguous role+phase+location or coreference chain is sufficient to align "
    "actors. Do not invent an unmentioned duplicate actor solely to deny a paraphrase. An "
    "identity objection must identify an actually distinct candidate actor, incompatible "
    "phase/role, or real semantic class conflict, such as background field versus body. "
    "A reference's indefinite singular is not an exactly-one restriction. Multiple described "
    "formations or parts can correspond to one organism/group for non-count predicates when "
    "their role and phase chain match; their unresolved counting unit affects the count atom, "
    "not independently established motion, location, color or persistence. One-to-many group "
    "correspondence is allowed. A generic visual noun such as formation need not repeat the "
    "word organism. This does not license a background field as a body solely because both "
    "become green, nor infer missing physical parts or a gap from a nickname.\n"
    "Judge-only synthetic illustration: reference 'A violet body remains in the lower-right' "
    "with separate count, color and location atoms; candidate 'Two violet bodies remain in the "
    "lower-right'. If this is the same uniquely identified actor class and phase, location "
    "and color are communicated while the exactly-one count is not. Do not reject location "
    "because the separate count fails. No such example is supplied to the research agent.\n"
)
PROFILE = "atomic-complete-format-diagnostics-5g"
USE_FIXED_VERIFIER = False
OBSERVATIONAL_RULES = (
    "\nOBSERVATIONAL TASK MEANING:\n"
    "These annotations describe visible biological/morphodynamic behavior, not microscopic "
    "material particles. An organism's identity is its coherent visual/morphodynamic phenotype. "
    "Explicit continuity of a population or coherent form does not require literal fluid lineage "
    "or unchanged material in every constituent. Do not turn 'same blobs' into a requirement "
    "that every individual particle or constituent survives. Loss of a described dynamic type "
    "can coexist with its material persisting in another form; it does not require zero numerical "
    "mass or disappearance of every similarly colored fragment.\n"
    "Generic ongoing descriptions such as 'continually/constantly' express a characteristic "
    "habitual regime unless the text explicitly quantifies no pauses or every instant. Read "
    "generic present-tense accounts of repeated stretching, breakup and reformation accordingly; "
    "do not invent silent intervals merely to deny ordinary habitual wording. Explicit occasional "
    "versus frequent activity, numerical rates, before/after comparisons and degree still matter.\n"
    "Observable deformation, breakup and reorganization count as activity/movement unless the "
    "source specifically requires translation, locomotion or center displacement. Do not add a "
    "translation-only requirement to an otherwise general motion/dynamics predicate. A resulting "
    "replacement across a transition need not be its immediate first breakup product unless "
    "the target explicitly requires immediacy. Do not add material-origin causation to observed "
    "replacement. Read all stages of the complete caption, including the later outcome.\n"
    "A countermodel cannot invent unmentioned recoloring, population turnover or another hidden "
    "mechanism solely to erase a clear visual coreference/shape-change chain. Preserve the "
    "ordinary descriptive correspondence; this is not evidence for a microscopic mechanism. "
    "Genuine ambiguous correspondence, a field/body class conflict, explicit changed color or "
    "actor, different numerical/comparative dimensions, and A-or-B requiring BOTH branches "
    "remain strict. No property may be filled in merely because the reference needs it.\n"
)
PROMPTS = {kind: prompt + ATOMIC_RULES + ENTITY_RULES + REFERENTIAL_RULES + OBSERVATIONAL_RULES + "\n" + CLAIM_RULES
           for kind, prompt in BASE_PROMPTS.items()}
SCHEMAS = deepcopy(BASE_SCHEMAS)
SCHEMAS["description"]["properties"]["coverage"]["items"] = object_schema({
    "id": STRING, "entity_alignment": object_schema({
        "status": {"type": "string", "enum": ["aligned", "not_applicable", "unresolved"]},
        "independent_basis": STRING}),
    "predicate_check": object_schema({"dimension": STRING, "required": STRING,
        "candidate_assertion": STRING, "countermodel": STRING}),
    "present": {"type": "boolean"}, "quote": STRING, "reason": STRING})
SCHEMAS["description"] = schema_for(SCHEMAS["description"])
