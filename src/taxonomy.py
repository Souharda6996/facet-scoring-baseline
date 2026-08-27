"""
Deterministic, rule-based taxonomy classification for facet rows.

Design intent (see DECISIONS.md): classification here is regex/keyword-based,
not LLM-based, so that Part 1 output is 100% reproducible byte-for-byte across
runs and requires no model/API calls. LLM calls are reserved for Part 2
(per-conversation scoring), where non-determinism is expected and evidence is
grounded in actual text.
"""
import re

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_LEADING_ID_RE = re.compile(r"^\s*(\d+)\.\s*")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_WS_RE = re.compile(r"\s+")


def strip_leading_id(raw: str):
    """Strip a leading 'NNN. ' scrape-artifact prefix. Returns (stripped, id_or_None)."""
    m = _LEADING_ID_RE.match(raw)
    if m:
        return raw[m.end():], m.group(1)
    return raw, None


def split_camel_case(s: str) -> str:
    """Insert a space at lower->Upper boundaries only, so acronyms (IQ, FSH, HEXACO)
    stay intact while 'SelfEsteem' -> 'Self Esteem'."""
    return _CAMEL_BOUNDARY_RE.sub(" ", s)


def normalize_facet(raw: str) -> dict:
    """Produce a normalized display form + metadata about transformations applied."""
    original = raw
    transforms = []

    s = raw.strip()

    s2, leading_id = strip_leading_id(s)
    if leading_id is not None:
        s = s2.strip()
        transforms.append("stripped_leading_numeric_id")

    if s.endswith(":"):
        s = s[:-1].strip()
        transforms.append("stripped_trailing_colon")

    camel_split = split_camel_case(s)
    if camel_split != s:
        s = camel_split
        transforms.append("split_camel_case")

    s = _WS_RE.sub(" ", s).strip()

    return {
        "normalized_value": s,
        "leading_id": leading_id,
        "transforms": transforms,
        "original_had_trailing_colon": original.strip().endswith(":"),
    }


# ---------------------------------------------------------------------------
# Malformed / header-like detection
# ---------------------------------------------------------------------------

def detect_malformed(raw: str) -> dict:
    """Returns {'is_malformed': bool, 'malformed_reason': str|None}.
    Deterministic heuristics only -- no manual row-by-row labeling.
    """
    s = raw.strip()

    if s == "":
        return {"is_malformed": True, "malformed_reason": "empty_value"}

    # Trailing colon is the dominant, highly reliable signal in this dataset for
    # "this row is a section/category header that was scraped in as a facet row"
    # e.g. "Democratic Leadership:", "HEXACO Personality Inventory Facets:",
    # "Numerical Reasoning Subcomponents:", "Judging (J):"
    if s.endswith(":"):
        return {"is_malformed": True, "malformed_reason": "header_like_trailing_colon"}

    # Defensive extra check: known category/aggregator nouns with no real single
    # scorable meaning, even without a trailing colon.
    header_nouns = (
        "subcomponents", "components", "end points", "additional common parameters",
    )
    low = s.lower()
    if any(h in low for h in header_nouns):
        return {"is_malformed": True, "malformed_reason": "header_like_aggregator_phrase"}

    return {"is_malformed": False, "malformed_reason": None}


# ---------------------------------------------------------------------------
# facet_type classification (keyword/regex, priority-ordered)
# ---------------------------------------------------------------------------

SPIRITUAL_KEYWORDS = [
    "sufi", "hindu", "buddhist", "islamic", "jewish", "sikh", "kabbalah",
    "i ching", "hexagram", "astrology", "meditation", "scripture", "quran",
    "sacred text", "spiritual", "religious", "seerah", "dhikr", "khatam",
    "sephira", "zohar", "bhagavad", "baháí", "ridván", "shabbat",
    "sukkot", "reiki", "energy-healing", "kirtan", "gnostic", "archon",
    "new-age", "channeling", "pilgrimage", "vrata", "yoga discipline",
]

MEDICAL_BIO_KEYWORDS = [
    "hormone", "fsh level", "parathyroid", "gene", "genetic", "chromatin",
    "immune-response", "metabolic rate", "serotonin transporter",
    "polygenic risk", "basophil", "macronutrient", "caffeine sensitivity gene",
    "microbiome", "biomarker",
]

CLINICAL_KEYWORDS = [
    "diagnosis", "disorder", "apnea", "(hy)", "(ma)", "(dep)", "(pd)", "(pa)",
    "(sc)", "(si)", "(mf)", "(hs)", "hysteria", "hypomania",
    "chronic pain presence", "burnout symptoms", "depression symptoms",
    "depression:", "sleep-disorder",
]

COGNITIVE_KEYWORDS = [
    "reasoning", "iq", "intelligence quotient", "psychomotor", "spatial perception",
    "memory for sounds", "auditory memory", "working memory", "mental arithmetic",
    "spelling accuracy", "divided attention", "sequential memory", "comparing alphanumeric",
    "estimating calculations", "comprehension of spoken information", "sentence structure",
    "analogies", "numeric filing", "alphabetical filing", "understanding mathematical",
    "understanding mechanical",
]

BEHAVIORAL_COUNT_PATTERN = re.compile(
    r"(/\s*(day|week|year|month)\b|\bcount\b|\bhours?\b|\bsessions?\b|\byears?\b|"
    r"\bfrequency\b|%\b|\bmg/day\b|\bkm/week\b|\btime outdoors\b)",
    re.IGNORECASE,
)

SENSITIVE_BIOGRAPHICAL_EXACT = {
    "nationality", "drug-use history", "physical-violence exposure",
    "kink-interest diversity", "data-sharing consent level",
    "home-security-system presence", "sleep-environment temperature",
}


def _keyword_pattern(keyword: str) -> re.Pattern:
    # Short/generic single-token keywords (e.g. "gene", "iq") need word
    # boundaries or they false-positive as substrings of unrelated words
    # (e.g. "gene" inside "General"). Multi-word phrases are specific enough
    # that a plain substring match is safe and simpler.
    # \b only works cleanly around plain alphanumeric tokens -- keywords with
    # parentheses/punctuation (e.g. "(dep)") sit between two non-word chars
    # where \b never fires, so leave those as plain substring matches.
    if keyword.isalpha() and len(keyword) <= 6:
        return re.compile(r"\b" + re.escape(keyword) + r"\b")
    return re.compile(re.escape(keyword))


_KEYWORD_PATTERN_CACHE = {}


def _matches_any(low: str, keywords) -> bool:
    for k in keywords:
        pat = _KEYWORD_PATTERN_CACHE.get(k)
        if pat is None:
            pat = _keyword_pattern(k)
            _KEYWORD_PATTERN_CACHE[k] = pat
        if pat.search(low):
            return True
    return False


def classify_facet_type(normalized_value: str, raw_value: str, is_malformed: bool) -> str:
    if is_malformed:
        return "header_artifact"

    low = normalized_value.lower()
    raw_low = raw_value.lower()

    if raw_low.strip() in SENSITIVE_BIOGRAPHICAL_EXACT or low in SENSITIVE_BIOGRAPHICAL_EXACT:
        return "sensitive_biographical"

    # Numbered spiritual-practice entries are the clearest spiritual signal;
    # also catch spiritual keywords without a numeric prefix.
    if _matches_any(low, SPIRITUAL_KEYWORDS):
        return "spiritual_religious_practice"

    if _matches_any(low, CLINICAL_KEYWORDS):
        return "clinical_psychological_scale"

    if _matches_any(low, MEDICAL_BIO_KEYWORDS):
        return "medical_biological"

    if _matches_any(low, COGNITIVE_KEYWORDS):
        return "cognitive_psychometric"

    if BEHAVIORAL_COUNT_PATTERN.search(low):
        return "behavioral_frequency_count"

    return "personality_trait_or_disposition"


# ---------------------------------------------------------------------------
# conversation_observable + sensitivity + abstention_reason, derived from facet_type
# ---------------------------------------------------------------------------

_TYPE_RULES = {
    "header_artifact": dict(
        conversation_observable=False, sensitivity="n/a",
        abstention_reason="not_a_scorable_facet_category_header",
        requires_explicit_disclosure=False,
    ),
    "medical_biological": dict(
        conversation_observable=False, sensitivity="high",
        abstention_reason="requires_medical_lab_or_genetic_evidence",
        requires_explicit_disclosure=False,
    ),
    "clinical_psychological_scale": dict(
        conversation_observable=False, sensitivity="high",
        abstention_reason="requires_validated_clinical_instrument_or_diagnosis",
        requires_explicit_disclosure=False,
    ),
    "cognitive_psychometric": dict(
        conversation_observable=False, sensitivity="medium",
        abstention_reason="requires_formal_psychometric_testing",
        requires_explicit_disclosure=False,
    ),
    "spiritual_religious_practice": dict(
        conversation_observable=True, sensitivity="medium",
        abstention_reason=None,
        requires_explicit_disclosure=True,
    ),
    "sensitive_biographical": dict(
        conversation_observable=True, sensitivity="high",
        abstention_reason=None,
        requires_explicit_disclosure=True,
    ),
    "behavioral_frequency_count": dict(
        conversation_observable=True, sensitivity="low",
        abstention_reason=None,
        requires_explicit_disclosure=True,
    ),
    "personality_trait_or_disposition": dict(
        conversation_observable=True, sensitivity="low",
        abstention_reason=None,
        requires_explicit_disclosure=False,
    ),
}


def derive_scoring_metadata(facet_type: str) -> dict:
    return dict(_TYPE_RULES[facet_type])


def build_scoring_anchors(normalized_value: str) -> dict:
    """Generic, templated 5-level ordinal anchor set. Facet-specific wording is
    substituted into a fixed template so anchors are reproducible for all rows
    without hand-authoring 399 individual rubrics."""
    f = normalized_value
    return {
        "1": f"Conversation shows clear, explicit evidence of low/absent '{f}' "
             f"(or strong evidence of the opposite).",
        "2": f"Conversation shows weak or partial evidence leaning toward low '{f}'.",
        "3": f"Evidence is mixed, neutral, or too thin to lean either direction on '{f}'.",
        "4": f"Conversation shows weak-to-moderate evidence leaning toward high '{f}'.",
        "5": f"Conversation shows clear, explicit, repeated evidence of high '{f}'.",
    }
