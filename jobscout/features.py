"""Canonical feature vocabulary for gated assessment.

Stage 1 (jobscout/assess.py) asks the LLM to extract exactly these features from a
listing — each as {value, confidence, evidence}. Structured [[gates]] in config.toml
may only reference names listed here; anything the schema can't express goes through
a gate's natural-language `rule` instead. The vocabulary is generic (no user-specific
policy): descriptions below are the extraction contract handed to the model.
"""
from __future__ import annotations

# role_substance values the extractor must choose from (or null when unclear).
ROLE_SUBSTANCES = ["leadership", "founding", "ic", "management"]

# Only the features some [[gate]] or the tiering actually consumes are extracted —
# role_substance (tiering) plus the four structured-gate inputs below. Five formerly
# extracted-but-unused fields (title, function, remote_policy, company_eng_headcount,
# company_stage) were dropped: they cost ~25% of the response for no policy effect.
# Add a feature back here the moment a gate or tier needs it.
FEATURES: dict[str, str] = {
    "role_substance": (
        "What the role ACTUALLY is, regardless of its title label: 'leadership' = leads an "
        "engineering function/org (Head/Director/VP substance); 'management' = manages one team "
        "of engineers; 'founding' = hands-on 0-to-1 / new-product / early-stage ownership role "
        "built close to the founders; 'ic' = pure individual contributor with no team, org, or "
        "0-to-1 remit. One of: " + ", ".join(ROLE_SUBSTANCES) + "."
    ),
    "office_days_per_week": (
        "Required in-office days per week as a number, but ONLY when the listing EXPLICITLY "
        "states an office/remote policy or a day count: 0 for an explicit 'fully remote', 5 for "
        "an explicit 'fully on-site'/'5 days in office', the stated number for hybrid ('minimum "
        "N days' = N). Otherwise return null — do NOT guess or infer a number. In particular, do "
        "NOT derive a day count from soft co-location signals: 'based in <city>', a named HQ, "
        "'work alongside the founders', 'in-person/in-office culture', proximity to the team, or "
        "a scraped 'In-Office'/'Hybrid' Location card tag are NOT a stated policy and MUST yield "
        "null, not a number. Founding / early-stage roles rarely state a formal day count — if "
        "none is stated, return null. Only use confidence >= 0.7 when an explicit day count or "
        "policy sentence is present in the description; otherwise null."
    ),
    "salary_gbp": (
        "Stated annual salary in GBP as a number — use the TOP of a stated range. null when "
        "unstated, non-annual, or in a currency you cannot confidently convert."
    ),
    "primary_backend_language": (
        "The primary required backend language (lowercase, e.g. python, java, go, typescript); "
        "null when none is required or several are equally acceptable."
    ),
    "industry": (
        "The company's industry, lowercase (e.g. fintech, music, crypto, gambling, healthcare, "
        "devtools); null when unclear."
    ),
}

# Operators a structured gate may use against a feature value.
GATE_OPS = ["<=", ">=", "==", "!=", "in", "not_in", "contains"]
