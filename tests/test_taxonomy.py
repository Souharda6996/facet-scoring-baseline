import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taxonomy import (
    normalize_facet, detect_malformed, classify_facet_type, split_camel_case,
)


def test_header_like_colon_detected():
    r = detect_malformed("Democratic Leadership:")
    assert r["is_malformed"] is True
    assert r["malformed_reason"] == "header_like_trailing_colon"


def test_plain_facet_not_malformed():
    r = detect_malformed("Risktaking")
    assert r["is_malformed"] is False


def test_leading_numeric_id_stripped():
    n = normalize_facet("800. Sufi practice: Sufi retreat attendance count")
    assert n["leading_id"] == "800"
    assert n["normalized_value"].startswith("Sufi practice")


def test_camel_case_split_preserves_acronyms():
    assert split_camel_case("SelfEsteem") == "Self Esteem"
    assert split_camel_case("HEXACO") == "HEXACO"
    assert split_camel_case("IQ") == "IQ"


def test_no_false_positive_gene_substring_in_general():
    # Regression test: "gene" as a bare substring previously matched inside
    # "General", misrouting "General Mood and Attitude" to medical_biological.
    t = classify_facet_type("General Mood and Attitude", "General Mood and Attitude", False)
    assert t == "personality_trait_or_disposition"


def test_parenthesized_clinical_code_matches_despite_word_boundary_fix():
    # Regression test: \b wrapping broke matching for "(dep)"/"(hy)"/"(ma)"
    # because \b never fires between two non-word characters like "(" and a
    # preceding space. classify_facet_type must still route these correctly.
    assert classify_facet_type("Depression (DEP)", "Depression (DEP)", False) == "clinical_psychological_scale"
    assert classify_facet_type("Hypomania (Ma)", "Hypomania (Ma)", False) == "clinical_psychological_scale"
    assert classify_facet_type("Hysteria (Hy)", "Hysteria (Hy)", False) == "clinical_psychological_scale"


def test_medical_lab_value_classified_and_gated():
    from taxonomy import derive_scoring_metadata
    t = classify_facet_type("FSH level", "FSH level", False)
    assert t == "medical_biological"
    meta = derive_scoring_metadata(t)
    assert meta["conversation_observable"] is False
    assert meta["abstention_reason"] == "requires_medical_lab_or_genetic_evidence"


def test_numbered_spiritual_entry_classified():
    t = classify_facet_type(
        "Astrology: Rising sign is Scorpio", "692. Astrology: Rising sign is Scorpio", False
    )
    assert t == "spiritual_religious_practice"


def test_behavioral_count_pattern():
    t = classify_facet_type("Caffeine intake (mg/day)", "Caffeine intake (mg/day)", False)
    assert t == "behavioral_frequency_count"
