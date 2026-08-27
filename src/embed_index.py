"""
Build a local embedding index over the enriched facet catalogue.

Design (see DECISIONS.md "retrieval index"):
- Header-artifact rows (facet_type == "header_artifact") are excluded entirely
  -- they are not facets at all, so they should never be retrievable.
- Everything else (including medical/clinical/cognitive facets that are NOT
  conversation_observable) IS embedded and indexed. This is deliberate: the
  taxonomy gate that blocks non-observable facets from ever reaching the LLM
  lives in the retrieval/routing layer (src/retrieve.py), not in the index
  itself. Keeping them in the index lets us demonstrate, concretely, that
  embeddings alone would surface a facet like "FSH level" for a
  fertility-related conversation -- and that the taxonomy gate is the thing
  actually stopping it from being scored. See hallucination_demo/.
- At this catalogue size (~370 scorable rows) a flat numpy cosine-similarity
  search is simplest and exactly as correct as an ANN index. See
  DECISIONS.md / README "scale to 5000 facets" for why this would become a
  FAISS/HNSW index at that scale instead of a code change in kind.

Usage:
    python src/embed_index.py
Writes data/processed/facet_index.npz (embeddings + facet_id order) and
uses the local sentence-transformers cache (downloads once if not present).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ENRICHED_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "facets_enriched.csv"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "facet_index.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _embedding_text(row: pd.Series) -> str:
    ftype = str(row["facet_type"]).replace("_", " ")
    return f"{row['normalized_value']} ({ftype})"


def build_index(model_name: str = MODEL_NAME):
    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8")
    indexable = df[df["facet_type"] != "header_artifact"].reset_index(drop=True)

    texts = indexable.apply(_embedding_text, axis=1).tolist()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    facet_ids = indexable["facet_id"].to_numpy()

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        embeddings=embeddings,
        facet_ids=facet_ids,
        model_name=np.array(model_name),
    )
    print(f"Indexed {len(facet_ids)} facets (excluded {len(df) - len(indexable)} header artifacts)")
    print(f"Wrote -> {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
