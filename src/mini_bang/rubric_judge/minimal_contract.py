"""Minimal protocol: the base prompts plus semantic cross-check clarifications."""
from mini_bang.rubric_judge.base_contract import PROMPTS as BASE_PROMPTS, SCHEMAS

JUDGE_VERSION = "text-evidence-minimal"
CLARIFICATIONS = (
    "\nSemantic cross-check before returning your answer:\n"
    "Keep comparative degree, temporal phase and epistemic qualifications in each extracted claim. "
    "A change becoming stronger does not assert that it did not exist before. A lasting outcome "
    "does not assert that the transition began slowly. Do not turn an uncertain conclusion into "
    "an asserted denial. Read the whole candidate when resolving a comparison or pronoun; an "
    "ordinary paraphrase need not repeat the rubric's nouns or concluding formula.\n"
    "Separate what happened from what the candidate says defines the event. If the reference "
    "explicitly says X does NOT make it the event, positively identifying X as what makes it "
    "the event conflicts with that definition, even if X physically happened. A statement that "
    "a cause is unknown does not assert a cause or deny an observed transformation.\n"
    "Extract one count or identity assertion once. Do not reclassify every motion/shape predicate "
    "as contradicted solely because its noun phrase repeats that same disputed count. Two parts "
    "of an organism are not necessarily two organisms; unresolved identity is unverifiable.\n"
    "Judge-only semantic controls (these are language examples, not observations of the supplied system): "
    "Reference: 'The patterned array becomes steadier.' Candidate: 'Its loose order becomes more "
    "coherent.' These can both be true; no contradiction about the array's existence follows. "
    "Reference: 'The change starts abruptly and the new regime persists.' Candidate: 'The shift "
    "is sustained.' Duration does not contradict onset. Reference: 'Several units remain alive.' "
    "Candidate: 'Every unit has died.' This is a genuine contradiction. Reference: 'The event is "
    "increased activity, not the loss of one unit.' Candidate: 'That unit's loss is what makes "
    "this the event.' This is a contradictory event interpretation. Candidate: 'Its cause is "
    "not established.' This does not assert or contradict a physical mechanism.\n"
)
PROMPTS = {kind: value + CLARIFICATIONS for kind, value in BASE_PROMPTS.items()}
