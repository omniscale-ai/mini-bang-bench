"""Necessary source-witness checks; never infer or rewrite a semantic label.

This finite wording guard is intentionally not an entailment engine. The model
must still compare the entire epistemic proposition and the witness's scope.
Optional additional-language wording comes from the package lexicon.
"""
import re

from mini_bang.rubric_judge import lexicon


RULES = """EPISTEMIC CLAIMS REQUIRE POSITIVE SOURCE WITNESSES
A claim such as 'the observations do not establish X' is about the evidence,
not an assertion that X did not occur. Label it supported ONLY when the reference,
interpretation_notes or uncertain_claims explicitly states that X, or a class
containing X, is unknown, not established, unverifiable or a hypothesis. Quote
that positive uncertainty statement, preserving its subject and epistemic force.
Selective silence about X is not such a statement. An outcome such as no births
or deaths is not a statement about evidence for reproduction or predation.
Uncertainty about a narrower subject does not establish a broader uncertainty:
unknown material origin of particular organisms does not establish that the
cause of the whole event is unknown. Check every part of a compound claim.
When only silence or a narrower statement is available, use unverifiable,
never contradicted on that basis. Do not turn uncertainty into an established
world fact or assert that the causal phenomenon is absent.
The deterministic witness guard checks necessary wording and coarse topic
compatibility only. Passing it does not establish semantic entailment. Never
invent, expand or change a source quotation to satisfy it.
"""

# Epistemic negation, not an ordinary negation of an observable event.
UNCERTAINTY = re.compile(
    r"\b(?:unknown|uncertain|unverifiable|unverified|hypothes(?:is|es)|unproven)\b"
    r"|\b(?:not|never)\s+(?:\w+\s+){0,3}(?:established|known|verified|determined|explained|proven)\b"
    r"|\b(?:do|does|did|can|could)\s+not\s+(?:\w+\s+){0,3}(?:establish|determine|verify|explain|prove)\b"
    r"|\b(?:cannot|can't|doesn't|don't|didn't)\s+(?:\w+\s+){0,3}(?:establish|determine|verify|explain|prove)\b"
    r"|\bwithout\s+(?:\w+\s+){0,3}establishing\b"
    + lexicon.alternatives("epistemic_uncertainty"),
    re.IGNORECASE)
TOPICS = {name: r"\b(?:" + words + lexicon.alternatives("epistemic_topics", name) + ")" for name, words in {
    "cause": r"caus\w*|mechanism\w*",
    "predation": r"predat\w*",
    "competition": r"competit\w*",
    "mutation": r"mutat\w*|genetic\w*",
    "reproduction": r"reproduct\w*",
    "ancestry": r"ancestr\w*|parent\w*|descend\w*|origin\w*",
}.items()}
EVIDENCE_SUBJECT = re.compile(r"\b(?:observations?|evidence|reference|account|annotation\w*|notes?"
    + lexicon.alternatives("evidence_subjects") + ")", re.IGNORECASE)
INFERENCE = re.compile(r"\b(?:do|does|did)\s+not\s+establish\b"
    r"|\b(?:doesn't|don't|didn't|cannot|can't)\s+establish\b", re.IGNORECASE)


def topics(text):
    return {name for name, pattern in TOPICS.items() if re.search(pattern, text, re.IGNORECASE)}


def epistemic_support_issues(claim, data):
    """Return format/evidence defects only; the caller owns its repair loop."""
    text = claim["claim"].replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    subject = topics(text)
    if claim["label"] != "supported" or not UNCERTAINTY.search(text):
        return []
    # An explicit inference distinction ('color alone does not establish a
    # species replacement') is not a claim that the observations leave that
    # replacement unknown. Preserve ordinary source-based comparison rules.
    if INFERENCE.search(text) and not EVIDENCE_SUBJECT.search(text):
        return []
    witness = claim.get("reference_quote", "")
    # Rubric facts still support ordinary claims; epistemic claims need an
    # explicit uncertainty statement in the account, notes or uncertain claims.
    sources = [data.get("reference", ""), *data.get("interpretation_notes", []),
               *data.get("uncertain_claims", [])]
    if (not witness or not any(witness in source for source in sources)
            or not UNCERTAINTY.search(witness)):
        return ["Supported epistemic claim requires a verbatim positive uncertainty witness from reference/notes/uncertain_claims"]
    witness_topics = topics(witness)
    # A broad causal uncertainty may contain a named causal mechanism; an
    # ancestry-only or measurement-only uncertainty cannot cover general cause.
    # Full subject, actor and phase entailment remain model responsibilities.
    if subject and not subject <= witness_topics and "cause" not in witness_topics:
        return ["Epistemic witness lacks the claim's topic or a containing causal class; preserve source scope"]
    return []
