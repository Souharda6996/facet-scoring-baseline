"""
Optional brownie point: "a simple comparison/ablation between two retrieval/
scoring choices" (per the brief).

Ablates one real design decision from src/embed_index.py: facets are embedded
as "{normalized_value} ({facet_type})" rather than the bare facet name alone
(see `_embedding_text()`). This script checks whether that choice actually
helps retrieval, using the 20 reference (conversation, facet) pairs in
eval/reference_labels.csv as ground truth for "this facet should be
retrievable for this conversation" -- comparing rank/recall under both
embedding formats.

Pure embedding computation, no LLM calls -- fast, deterministic, no Ollama
dependency.

Usage: python eval/retrieval_ablation.py
Writes eval/retrieval_ablation_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EVAL_DIR = Path(__file__).resolve().parent
ENRICHED_CSV = EVAL_DIR.parent / "data" / "processed" / "facets_enriched.csv"
CONVERSATIONS_PATH = EVAL_DIR / "conversations.json"
REFERENCE_PATH = EVAL_DIR / "reference_labels.csv"
REPORT_PATH = EVAL_DIR / "retrieval_ablation_report.md"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FORMATS = {
    "current (name + facet_type context)": lambda row: f"{row['normalized_value']} ({str(row['facet_type']).replace('_', ' ')})",
    "ablated (bare facet name only)": lambda row: str(row["normalized_value"]),
}


def build_embeddings(df: pd.DataFrame, model: SentenceTransformer, fmt_fn):
    texts = df.apply(fmt_fn, axis=1).tolist()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def rank_of(facet_id: str, sims: np.ndarray, facet_ids: np.ndarray) -> int:
    """1-indexed rank of facet_id by similarity (1 = most similar)."""
    order = np.argsort(-sims)
    ranked_ids = facet_ids[order]
    return int(np.where(ranked_ids == facet_id)[0][0]) + 1


def run():
    df = pd.read_csv(ENRICHED_CSV, encoding="utf-8")
    indexable = df[df["facet_type"] != "header_artifact"].reset_index(drop=True)
    facet_ids = indexable["facet_id"].to_numpy()

    conversations = {c["id"]: c["text"] for c in json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))}
    ref = pd.read_csv(REFERENCE_PATH, encoding="utf-8")

    model = SentenceTransformer(MODEL_NAME)

    results = {}
    for fmt_name, fmt_fn in FORMATS.items():
        embeddings = build_embeddings(indexable, model, fmt_fn)
        rows = []
        for _, r in ref.iterrows():
            conv_text = conversations[r["conversation_id"]]
            query_emb = model.encode([conv_text], normalize_embeddings=True)[0]
            sims = embeddings @ query_emb
            rank = rank_of(r["facet_id"], sims, facet_ids)
            rows.append({
                "conversation_id": r["conversation_id"], "facet_id": r["facet_id"],
                "facet_name": r["facet_name"], "rank": rank,
            })
        results[fmt_name] = pd.DataFrame(rows)

    lines = ["# Retrieval Ablation: does embedding facet_type context help?\n"]
    lines.append(
        "Ablates `src/embed_index.py`'s `_embedding_text()` choice to embed facets as "
        "`\"{name} ({facet_type})\"` rather than the bare name. Ground truth: the 20 "
        "(conversation, facet) pairs in `eval/reference_labels.csv` -- for each, what rank "
        "does the correct facet get by cosine similarity to the conversation, under each "
        "embedding format? Lower rank is better (rank 1 = most similar facet in the whole catalogue).\n"
    )

    for fmt_name, rdf in results.items():
        lines.append(f"## {fmt_name}\n")
        lines.append(f"- Mean rank: **{rdf['rank'].mean():.1f}**")
        lines.append(f"- Median rank: **{rdf['rank'].median():.1f}**")
        lines.append(f"- Recall@8 (rank <= 8): **{(rdf['rank'] <= 8).mean():.0%}**")
        lines.append(f"- Recall@20 (rank <= 20): **{(rdf['rank'] <= 20).mean():.0%}**")
        lines.append("")

    fmt_names = list(results.keys())
    merged = results[fmt_names[0]][["conversation_id", "facet_name", "rank"]].merge(
        results[fmt_names[1]][["conversation_id", "facet_name", "rank"]],
        on=["conversation_id", "facet_name"], suffixes=("_current", "_ablated"),
    )
    merged["delta"] = merged["rank_ablated"] - merged["rank_current"]  # positive = current format is better (lower rank)

    lines.append("## Per-facet comparison\n")
    lines.append("Positive delta = the current (name + facet_type) format ranks the correct facet higher (better).\n")
    lines.append("| conversation | facet | rank (current) | rank (ablated) | delta |")
    lines.append("|---|---|---|---|---|")
    for _, r in merged.sort_values("delta", ascending=False).iterrows():
        lines.append(f"| {r['conversation_id']} | {r['facet_name']} | {r['rank_current']} | {r['rank_ablated']} | {r['delta']:+d} |")
    lines.append("")

    n_current_better = int((merged["delta"] > 0).sum())
    n_ablated_better = int((merged["delta"] < 0).sum())
    n_tied = int((merged["delta"] == 0).sum())
    lines.append(f"\n**Summary: current format better on {n_current_better}/20, ablated format better on "
                 f"{n_ablated_better}/20, tied on {n_tied}/20.**\n")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote -> {REPORT_PATH}")
    print(f"current mean rank: {results[fmt_names[0]]['rank'].mean():.1f} | "
          f"ablated mean rank: {results[fmt_names[1]]['rank'].mean():.1f}")
    print(f"current better: {n_current_better}, ablated better: {n_ablated_better}, tied: {n_tied}")


if __name__ == "__main__":
    run()
