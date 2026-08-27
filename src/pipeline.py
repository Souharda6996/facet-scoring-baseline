"""
End-to-end: conversation -> retrieve candidates -> taxonomy gate -> batch score.

This is the single entry point both the benchmark (eval/run_eval.py) and the
hallucination demo call into, so all three exercise the identical code path.
"""
from pathlib import Path

from retrieve import FacetRetriever
from score import score_batch, chunk

DEFAULT_TOP_K = 20
DEFAULT_BATCH_SIZE = 8


def evaluate_conversation(
    conversation: str,
    retriever: FacetRetriever,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_facet_ids: list[str] | None = None,
    model: str | None = None,
) -> dict:
    """Returns a dict with:
      - taxonomy_blocked: facets retrieved by embedding similarity but never
        sent to the LLM because the taxonomy marks them non-observable
      - llm_scored: facets sent to the LLM scorer, with their verdicts
      - all_results: the two lists merged into one flat list (uniform schema)
    """
    candidates = retriever.retrieve(conversation, top_k=top_k)

    if force_facet_ids:
        existing_ids = {c["facet_id"] for c in candidates}
        forced = retriever.retrieve_by_facet_ids(
            [fid for fid in force_facet_ids if fid not in existing_ids]
        )
        candidates = candidates + forced

    llm_eligible, taxonomy_blocked = retriever.split_by_taxonomy_gate(candidates)

    taxonomy_blocked_results = [
        {
            "facet_id": c["facet_id"],
            "raw_value": c["raw_value"],
            "normalized_value": c["normalized_value"],
            "facet_type": c["facet_type"],
            "source": "taxonomy_gate",
            "status": "not_observable",
            "score": None,
            "confidence": None,
            "evidence": None,
            "reason": c["taxonomy_abstention_reason"] or "facet_type_not_conversation_observable",
            "similarity": c["similarity"],
        }
        for c in taxonomy_blocked
    ]

    llm_results = []
    for batch in chunk(llm_eligible, batch_size):
        batch_results = score_batch(conversation, batch, model=model)
        sim_by_id = {c["facet_id"]: c["similarity"] for c in batch}
        for r in batch_results:
            r["similarity"] = sim_by_id.get(r["facet_id"])
        llm_results.extend(batch_results)

    return {
        "taxonomy_blocked": taxonomy_blocked_results,
        "llm_scored": llm_results,
        "all_results": taxonomy_blocked_results + llm_results,
    }
