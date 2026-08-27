"""
Part 3: small benchmark runner.

For each test conversation, runs the real pipeline (retrieval -> taxonomy gate
-> batch LLM scoring) -- force-including the specific facets that have a human
reference label for that conversation, on top of normal embedding retrieval,
so every reference row is guaranteed a system verdict to compare against.

Usage:
    python eval/run_eval.py
Writes eval/results_raw.csv (full system output) and eval/report.md
(comparison against reference_labels.csv + failure-mode summary).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieve import FacetRetriever  # noqa: E402
from pipeline import evaluate_conversation  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CONVERSATIONS_PATH = EVAL_DIR / "conversations.json"
REFERENCE_PATH = EVAL_DIR / "reference_labels.csv"
RESULTS_RAW_PATH = EVAL_DIR / "results_raw.csv"
REPORT_PATH = EVAL_DIR / "report.md"

BAND_RANGES = {
    "low": (1, 2),
    "mid": (2, 4),
    "mid-high": (3, 5),
    "high": (4, 5),
}


def score_in_band(score, band: str) -> bool:
    if score is None or band not in BAND_RANGES:
        return False
    lo, hi = BAND_RANGES[band]
    return lo <= score <= hi


def run():
    conversations = json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))
    ref = pd.read_csv(REFERENCE_PATH, encoding="utf-8")

    retriever = FacetRetriever()

    all_rows = []
    for conv in conversations:
        conv_id = conv["id"]
        needed_facet_ids = ref.loc[ref["conversation_id"] == conv_id, "facet_id"].tolist()

        # top_k kept modest so the full 12-conversation benchmark finishes in a
        # reasonable time on this machine's CPU/GPU-split local inference (see
        # DEBUGGING.md #4); reference-label coverage is guaranteed regardless
        # via force_facet_ids, independent of top_k.
        result = evaluate_conversation(
            conv["text"], retriever, top_k=8,
            force_facet_ids=needed_facet_ids,
        )
        for r in result["all_results"]:
            row = dict(r)
            row["conversation_id"] = conv_id
            row["conversation_category"] = conv["category"]
            all_rows.append(row)
        print(f"[{conv_id}] {conv['category']}: {len(result['llm_scored'])} llm-scored, "
              f"{len(result['taxonomy_blocked'])} taxonomy-blocked")

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(RESULTS_RAW_PATH, index=False, encoding="utf-8")
    print(f"Wrote raw system output -> {RESULTS_RAW_PATH}")

    write_report(raw_df, ref)


def write_report(raw_df: pd.DataFrame, ref: pd.DataFrame):
    merged = ref.merge(
        raw_df, on=["conversation_id", "facet_id"], how="left", suffixes=("_expected", "_system")
    )

    def evaluate_row(row):
        expected_status = row["expected_status"]
        system_status = row.get("status")
        if pd.isna(system_status):
            return "MISSING", "system produced no row for this facet (not retrieved at all)"

        if expected_status == "scored":
            if system_status != "scored":
                return "WRONG_ABSTAIN", f"expected scored({row['expected_score_band']}) but system returned {system_status}"
            if score_in_band(row.get("score"), row["expected_score_band"]):
                return "MATCH", "score within expected band"
            return "WRONG_SCORE", f"expected band {row['expected_score_band']}, system scored {row.get('score')}"
        else:
            # expected abstention (not_observable or insufficient_evidence)
            if system_status == "scored":
                return "HALLUCINATION", f"expected abstain ({expected_status}) but system invented score {row.get('score')}"
            if system_status == expected_status:
                return "MATCH", "exact abstention-type match"
            return "MATCH_LOOSE", f"both abstained but via different status ({system_status} vs {expected_status})"

    outcomes = merged.apply(evaluate_row, axis=1, result_type="expand")
    merged["outcome"], merged["outcome_detail"] = outcomes[0], outcomes[1]

    n = len(merged)
    counts = merged["outcome"].value_counts().to_dict()
    n_match = counts.get("MATCH", 0) + counts.get("MATCH_LOOSE", 0)
    n_hallucination = counts.get("HALLUCINATION", 0)

    lines = []
    lines.append("# Benchmark Report\n")
    lines.append(f"Reference rows evaluated: **{n}**\n")
    lines.append("## Outcome counts\n")
    for k in ["MATCH", "MATCH_LOOSE", "WRONG_SCORE", "WRONG_ABSTAIN", "HALLUCINATION", "MISSING"]:
        lines.append(f"- `{k}`: {counts.get(k, 0)}")
    lines.append("")
    lines.append(f"**Agreement rate (MATCH + MATCH_LOOSE): {n_match}/{n} = {n_match/n:.0%}**\n")
    lines.append(f"**Hallucinations (system scored something that must abstain): {n_hallucination}/{n}**\n")
    lines.append("This second number is the one that matters most for this assignment: it is 0 only if every "
                  "medical/clinical/cognitive facet in the reference set was correctly blocked from scoring.\n")

    lines.append("## Row-by-row detail\n")
    lines.append("| conv | facet | category | expected | system status | system score | outcome | detail |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in merged.iterrows():
        lines.append(
            f"| {r['conversation_id']} | {r['facet_name']} | {r['conversation_category']} | "
            f"{r['expected_status']}"
            f"{'('+str(r['expected_score_band'])+')' if r['expected_status']=='scored' else ''} | "
            f"{r.get('status')} | {r.get('score')} | **{r['outcome']}** | {r['outcome_detail']} |"
        )
    lines.append("")

    lines.append("## Failure modes observed\n")
    failures = merged[~merged["outcome"].isin(["MATCH", "MATCH_LOOSE"])]
    if len(failures) == 0:
        lines.append("None -- every reference row matched.\n")
    else:
        for outcome_type in failures["outcome"].unique():
            sub = failures[failures["outcome"] == outcome_type]
            lines.append(f"**{outcome_type}** ({len(sub)}):")
            for _, r in sub.iterrows():
                lines.append(f"- {r['conversation_id']}/{r['facet_name']}: {r['outcome_detail']}")
                if isinstance(r.get("evidence"), str) and r["evidence"]:
                    lines.append(f"  - model's cited evidence: \"{r['evidence']}\"")
                if isinstance(r.get("reason"), str) and r["reason"]:
                    lines.append(f"  - model's stated reason: \"{r['reason']}\"")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report -> {REPORT_PATH}")
    print(f"Agreement: {n_match}/{n} = {n_match/n:.0%} | Hallucinations: {n_hallucination}/{n}")


def regenerate_report_only():
    """Rebuilds report.md from the already-saved results_raw.csv, without
    re-running the (slow, local-LLM-backed) pipeline. Useful when only the
    report-formatting logic changed."""
    raw_df = pd.read_csv(RESULTS_RAW_PATH, encoding="utf-8")
    ref = pd.read_csv(REFERENCE_PATH, encoding="utf-8")
    write_report(raw_df, ref)


if __name__ == "__main__":
    import sys as _sys
    if "--report-only" in _sys.argv:
        regenerate_report_only()
    else:
        run()
