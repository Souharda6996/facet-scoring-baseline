"""
Retrieval/routing layer: given a conversation, select a relevant subset of
facets to score instead of evaluating all ~370 (or, at scale, 5000+).

Two-stage design (see DECISIONS.md "taxonomy gate vs retrieval filter"):
  1. Embedding retrieval over ALL indexed facets (including non-observable
     ones like medical/clinical/cognitive facets) -> top_k by cosine sim.
  2. Taxonomy gate splits the retrieved candidates into:
       - llm_eligible: conversation_observable == True -> sent to the LLM scorer
       - taxonomy_blocked: conversation_observable == False -> auto-abstained
         with the taxonomy's abstention_reason, WITHOUT ever calling the LLM.

Keeping stage 1 un-filtered is deliberate: it lets us show, concretely, that
embeddings *would* surface something like "FSH level" for a fertility-related
conversation, and that it is the taxonomy gate -- not the LLM's own judgment --
that stops it from being scored. This is the core anti-hallucination guarantee
demonstrated in hallucination_demo/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ENRICHED_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "facets_enriched.csv"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "facet_index.npz"


class FacetRetriever:
    def __init__(self, enriched_csv=ENRICHED_CSV, index_path=INDEX_PATH):
        self.df = pd.read_csv(enriched_csv, encoding="utf-8").set_index("facet_id", drop=False)

        data = np.load(index_path, allow_pickle=True)
        self.embeddings = data["embeddings"]  # (N, D), L2-normalized
        self.facet_ids = data["facet_ids"]
        model_name = str(data["model_name"])
        self.model = SentenceTransformer(model_name)

    def _facet_row_to_candidate(self, facet_id: str, similarity: float) -> dict:
        row = self.df.loc[facet_id]
        anchors = None
        if isinstance(row["scoring_anchors_json"], str) and row["scoring_anchors_json"]:
            anchors = json.loads(row["scoring_anchors_json"])
        return {
            "facet_id": facet_id,
            "raw_value": row["raw_value"],
            "normalized_value": row["normalized_value"],
            "facet_type": row["facet_type"],
            "conversation_observable": bool(row["conversation_observable"]),
            "sensitivity": row["sensitivity"],
            "requires_explicit_disclosure": bool(row["requires_explicit_disclosure"]),
            "taxonomy_abstention_reason": row["abstention_reason"] if pd.notna(row["abstention_reason"]) else None,
            "scoring_anchors": anchors,
            "similarity": float(similarity),
        }

    def retrieve(self, conversation_text: str, top_k: int = 20) -> list[dict]:
        query_emb = self.model.encode([conversation_text], normalize_embeddings=True)[0]
        sims = self.embeddings @ query_emb  # cosine sim (both L2-normalized)
        top_idx = np.argsort(-sims)[:top_k]
        return [
            self._facet_row_to_candidate(self.facet_ids[i], sims[i])
            for i in top_idx
        ]

    @staticmethod
    def split_by_taxonomy_gate(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
        llm_eligible = [c for c in candidates if c["conversation_observable"]]
        taxonomy_blocked = [c for c in candidates if not c["conversation_observable"]]
        return llm_eligible, taxonomy_blocked

    def retrieve_by_facet_ids(self, facet_ids: list[str]) -> list[dict]:
        """Force-include specific facets regardless of embedding similarity
        (used by the benchmark to guarantee the 20 representative facets are
        evaluated for every test conversation, plus by the hallucination demo
        to guarantee a specific medical facet is on the candidate list)."""
        out = []
        id_to_idx = {fid: i for i, fid in enumerate(self.facet_ids)}
        for fid in facet_ids:
            if fid in id_to_idx:
                out.append(self._facet_row_to_candidate(fid, similarity=float("nan")))
        return out
