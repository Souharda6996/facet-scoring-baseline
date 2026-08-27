import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from score import parse_batch_response, _coerce_result_item  # noqa: E402

FACET_BATCH = [
    {"facet_id": "F0000", "raw_value": "Risktaking", "normalized_value": "Risktaking",
     "facet_type": "personality_trait_or_disposition"},
    {"facet_id": "F0001", "raw_value": "Naivety", "normalized_value": "Naivety",
     "facet_type": "personality_trait_or_disposition"},
]


def test_well_formed_response_parses_cleanly():
    raw = """{"results": [
        {"facet_id": "F0000", "status": "scored", "score": 4, "confidence": 0.8,
         "evidence": "said they love bungee jumping", "reason": "explicit risk-seeking statement"},
        {"facet_id": "F0001", "status": "insufficient_evidence", "score": null, "confidence": 0.2,
         "evidence": null, "reason": "no relevant evidence"}
    ]}"""
    results = parse_batch_response(raw, FACET_BATCH)
    assert len(results) == 2
    assert results[0]["status"] == "scored" and results[0]["score"] == 4
    assert results[1]["status"] == "insufficient_evidence" and results[1]["score"] is None


def test_markdown_fenced_json_is_stripped_and_parsed():
    raw = """```json
    {"results": [
        {"facet_id": "F0000", "status": "scored", "score": 2, "confidence": 0.6,
         "evidence": "seemed cautious", "reason": "hesitant language"}
    ]}
    ```"""
    results = parse_batch_response(raw, [FACET_BATCH[0]])
    assert results[0]["status"] == "scored" and results[0]["score"] == 2


def test_completely_broken_json_falls_back_to_insufficient_evidence_not_crash():
    raw = "I'm not able to provide a JSON response right now, sorry!"
    results = parse_batch_response(raw, FACET_BATCH)
    assert len(results) == 2
    assert all(r["status"] == "insufficient_evidence" for r in results)
    assert all(r["confidence"] == 0.0 for r in results)


def test_out_of_range_score_is_downgraded_not_trusted():
    raw = '{"results": [{"facet_id": "F0000", "status": "scored", "score": 9, "confidence": 0.9, "evidence": "x", "reason": "y"}]}'
    results = parse_batch_response(raw, [FACET_BATCH[0]])
    assert results[0]["status"] == "insufficient_evidence"
    assert results[0]["score"] is None
    assert "invalid_score_value" in results[0]["reason"]


def test_invalid_status_value_downgraded():
    item = {"facet_id": "F0000", "status": "definitely_yes", "score": 5, "confidence": 0.9,
             "evidence": "x", "reason": "y"}
    coerced = _coerce_result_item(item, {"F0000"})
    assert coerced["status"] == "insufficient_evidence"


def test_missing_facet_in_llm_response_still_produces_a_row():
    # Model only answers for F0000, omits F0001 entirely.
    raw = '{"results": [{"facet_id": "F0000", "status": "scored", "score": 3, "confidence": 0.5, "evidence": "x", "reason": "y"}]}'
    results = parse_batch_response(raw, FACET_BATCH)
    ids = {r["facet_id"] for r in results}
    assert ids == {"F0000", "F0001"}
    f1 = next(r for r in results if r["facet_id"] == "F0001")
    assert f1["status"] == "insufficient_evidence"
    assert f1["reason"] == "model_omitted_this_facet_from_batch_response"


def test_unknown_facet_id_in_response_is_ignored_not_injected():
    raw = '{"results": [{"facet_id": "F9999", "status": "scored", "score": 5, "confidence": 0.9, "evidence": "x", "reason": "y"}]}'
    results = parse_batch_response(raw, [FACET_BATCH[0]])
    ids = {r["facet_id"] for r in results}
    assert ids == {"F0000"}  # F9999 never requested, must not appear
    assert results[0]["status"] == "insufficient_evidence"  # F0000 was never actually answered


def test_confidence_out_of_range_is_clamped():
    item = {"facet_id": "F0000", "status": "scored", "score": 3, "confidence": 5.0,
             "evidence": "x", "reason": "y"}
    coerced = _coerce_result_item(item, {"F0000"})
    assert coerced["confidence"] == 1.0
