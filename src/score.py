"""
Part 2: compact-batch structured scoring of retrieved, taxonomy-eligible facets.

Design:
- Facets are scored in small batches (default 8) via a single LLM call per
  batch, never one call per facet and never all facets in one call (both
  explicitly disallowed by the brief).
- Output is requested as strict JSON and then independently re-validated in
  Python -- the LLM's own "valid JSON" claim is not trusted. Any facet the
  model fails to return, or returns malformed, degrades to an explicit
  insufficient_evidence status rather than crashing the batch or the pipeline.
- The model is instructed to ground every "scored" verdict in a short
  evidence field quoting/paraphrasing the conversation; this is a soft
  guardrail (the hard guardrail is the taxonomy gate in retrieve.py, which
  never lets non-observable facets reach this module at all).
"""
import json
import re
from dataclasses import dataclass, field

from llm_client import call_llm, LLMCallError

ALLOWED_STATUS = {"scored", "not_observable", "insufficient_evidence"}

SYSTEM_PROMPT = """You are a careful conversation-analysis annotator. You will be given \
a short conversation and a small batch of psychological/behavioral facets to evaluate \
strictly from that conversation text.

Rules you must follow exactly:
1. Base every judgment ONLY on the given conversation text. Never use outside knowledge, \
stereotypes, or assumptions about the speaker not stated in the text.
2. If the conversation does not contain clear evidence for a facet, you MUST return \
status "insufficient_evidence" rather than guessing. Do not invent a score to be helpful.
3. Never infer medical, biological, or lab-test facts, diagnoses, or precise numeric \
external facts that are not explicitly and unambiguously stated in the conversation.
4. Score on a 5-level integer ordinal scale (1=low/opposite, 3=neutral/mixed, 5=high/clear) \
only when status is "scored".
5. For every facet, return a short "evidence" string: either a short quote/paraphrase from \
the conversation (if scored) or a brief note on why evidence is missing (if abstaining).
6. Return a "confidence" float between 0 and 1 reflecting how certain you are.
7. Output ONLY a JSON object of the exact shape described in the user message. No prose, \
no markdown fences, no commentary outside the JSON object.
"""

USER_TEMPLATE = """CONVERSATION:
\"\"\"
{conversation}
\"\"\"

FACETS TO EVALUATE (evaluate every one of these {n} facets, using each facet_id exactly once):
{facet_block}

Respond with a single JSON object of this exact shape:
{{
  "results": [
    {{
      "facet_id": "<facet_id>",
      "status": "scored" | "not_observable" | "insufficient_evidence",
      "score": <integer 1-5, or null if not "scored">,
      "confidence": <float 0.0-1.0>,
      "evidence": "<short quote or paraphrase from the conversation, or brief reason if abstaining>",
      "reason": "<one short sentence justifying the verdict>"
    }}
  ]
}}
"""


def _format_facet_block(facet_batch: list[dict]) -> str:
    lines = []
    for f in facet_batch:
        anchors = f.get("scoring_anchors") or {}
        lo = anchors.get("1", "")
        hi = anchors.get("5", "")
        lines.append(
            f"- facet_id={f['facet_id']} | name=\"{f['normalized_value']}\" | "
            f"1={lo} | 5={hi}"
        )
    return "\n".join(lines)


def build_prompt(conversation: str, facet_batch: list[dict]) -> tuple[str, str]:
    user_prompt = USER_TEMPLATE.format(
        conversation=conversation.strip(),
        n=len(facet_batch),
        facet_block=_format_facet_block(facet_batch),
    )
    return SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Robust parsing / validation -- never let a malformed LLM response propagate
# or crash the pipeline. Every facet in the batch always gets a result row.
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw_text: str) -> dict:
    cleaned = _CODE_FENCE_RE.sub("", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(cleaned)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object found in LLM output")


def _coerce_result_item(item: dict, valid_facet_ids: set) -> dict | None:
    if not isinstance(item, dict):
        return None
    facet_id = item.get("facet_id")
    if facet_id not in valid_facet_ids:
        return None

    status = item.get("status")
    if status not in ALLOWED_STATUS:
        status = "insufficient_evidence"
        note = f"invalid_status_value:{item.get('status')!r}"
    else:
        note = None

    score = item.get("score")
    if status == "scored":
        try:
            score = int(score)
            if not (1 <= score <= 5):
                raise ValueError
        except (TypeError, ValueError):
            status = "insufficient_evidence"
            note = f"invalid_score_value:{item.get('score')!r}"
            score = None
    else:
        score = None

    try:
        confidence = float(item.get("confidence"))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.3

    evidence = item.get("evidence")
    evidence = str(evidence).strip()[:500] if evidence else None
    model_reason = item.get("reason")
    model_reason = str(model_reason).strip()[:300] if model_reason else ""
    # A coercion note (invalid status/score) is a diagnostic fact about the
    # response and must survive even when the model also supplied its own
    # "reason" text -- otherwise the downgrade becomes invisible to callers.
    reason = f"{note} (model said: {model_reason})" if note and model_reason else (note or model_reason)

    return {
        "facet_id": facet_id,
        "status": status,
        "score": score,
        "confidence": confidence,
        "evidence": evidence,
        "reason": reason,
        "parse_note": note,
    }


def parse_batch_response(raw_text: str, facet_batch: list[dict]) -> list[dict]:
    valid_ids = {f["facet_id"] for f in facet_batch}
    results_by_id = {}

    try:
        obj = _extract_json_object(raw_text)
        raw_results = obj.get("results") if isinstance(obj, dict) else None
        if raw_results is None and isinstance(obj, list):
            raw_results = obj
        if not isinstance(raw_results, list):
            raise ValueError("'results' is not a list")

        for item in raw_results:
            coerced = _coerce_result_item(item, valid_ids)
            if coerced is not None:
                results_by_id[coerced["facet_id"]] = coerced
    except Exception as e:  # noqa: BLE001 -- malformed output must degrade, not crash
        # Whole-batch parse failure: every facet in this batch falls back below.
        parse_failure_reason = f"batch_parse_failure:{type(e).__name__}:{e}"
        for f in facet_batch:
            results_by_id.setdefault(f["facet_id"], {
                "facet_id": f["facet_id"], "status": "insufficient_evidence",
                "score": None, "confidence": 0.0, "evidence": None,
                "reason": parse_failure_reason, "parse_note": parse_failure_reason,
            })

    # Guarantee every requested facet has a row, even if the model omitted it.
    for f in facet_batch:
        results_by_id.setdefault(f["facet_id"], {
            "facet_id": f["facet_id"], "status": "insufficient_evidence",
            "score": None, "confidence": 0.0, "evidence": None,
            "reason": "model_omitted_this_facet_from_batch_response",
            "parse_note": "omitted",
        })

    return [results_by_id[f["facet_id"]] for f in facet_batch]


def score_batch(conversation: str, facet_batch: list[dict], model: str = None) -> list[dict]:
    """Scores one batch of (already taxonomy-eligible) facets against one
    conversation via a single LLM call. Never raises -- on any LLM failure,
    every facet in the batch gets an insufficient_evidence fallback row."""
    system_prompt, user_prompt = build_prompt(conversation, facet_batch)
    kwargs = {"model": model} if model else {}
    try:
        raw_text = call_llm(system_prompt, user_prompt, **kwargs)
    except LLMCallError as e:
        raw_text = ""
        results = parse_batch_response("", facet_batch)
        for r in results:
            r["reason"] = f"llm_call_failed:{e}"
            r["parse_note"] = "llm_call_failed"
        return _attach_facet_metadata(results, facet_batch)

    results = parse_batch_response(raw_text, facet_batch)
    return _attach_facet_metadata(results, facet_batch)


def _attach_facet_metadata(results: list[dict], facet_batch: list[dict]) -> list[dict]:
    meta_by_id = {f["facet_id"]: f for f in facet_batch}
    out = []
    for r in results:
        meta = meta_by_id[r["facet_id"]]
        merged = {
            "facet_id": r["facet_id"],
            "raw_value": meta["raw_value"],
            "normalized_value": meta["normalized_value"],
            "facet_type": meta["facet_type"],
            "source": "llm",
            **r,
        }
        out.append(merged)
    return out


def chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]
