"""
Part 1: audit and enrich the facet catalogue.

Usage:
    python src/preprocess.py

Reads data/raw/Facets Assignment.csv, writes:
    data/processed/facets_enriched.csv
    data/processed/AUDIT_SUMMARY.md

This script is the single source of truth for the enriched facet table --
never hand-edit the CSV. Re-running it is idempotent and deterministic.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import (
    normalize_facet, detect_malformed, classify_facet_type,
    derive_scoring_metadata, build_scoring_anchors,
)

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Facets Assignment.csv"
OUT_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "facets_enriched.csv"
OUT_AUDIT = Path(__file__).resolve().parent.parent / "data" / "processed" / "AUDIT_SUMMARY.md"


def build_enriched_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    col = raw_df.columns[0]
    rows = []
    normalized_seen = {}

    for idx, raw_value in enumerate(raw_df[col].astype(str).tolist()):
        facet_id = f"F{idx:04d}"

        malformed = detect_malformed(raw_value)
        norm = normalize_facet(raw_value)
        normalized_value = norm["normalized_value"]

        facet_type = classify_facet_type(normalized_value, raw_value, malformed["is_malformed"])
        meta = derive_scoring_metadata(facet_type)

        dup_of = normalized_seen.get(normalized_value.lower())
        is_duplicate = dup_of is not None
        if not is_duplicate:
            normalized_seen[normalized_value.lower()] = facet_id

        abstention_reason = meta["abstention_reason"]
        if is_duplicate and abstention_reason is None:
            abstention_reason = None  # duplicates still individually scorable; flagged separately

        anchors = build_scoring_anchors(normalized_value) if not malformed["is_malformed"] else None

        rows.append({
            "facet_id": facet_id,
            "raw_value": raw_value,
            "normalized_value": normalized_value,
            "leading_id_prefix": norm["leading_id"],
            "normalization_transforms": ";".join(norm["transforms"]) if norm["transforms"] else "",
            "is_malformed": malformed["is_malformed"],
            "malformed_reason": malformed["malformed_reason"],
            "facet_type": facet_type,
            "conversation_observable": meta["conversation_observable"] and not malformed["is_malformed"],
            "sensitivity": meta["sensitivity"],
            "requires_explicit_disclosure": meta["requires_explicit_disclosure"],
            "abstention_reason": abstention_reason,
            "is_duplicate_normalized": is_duplicate,
            "duplicate_of_facet_id": dup_of,
            "scoring_scale": "1-5 ordinal" if not malformed["is_malformed"] else None,
            "scoring_anchors_json": json.dumps(anchors, ensure_ascii=False) if anchors else None,
        })

    return pd.DataFrame(rows)


def write_audit_summary(df: pd.DataFrame, path: Path):
    n = len(df)
    n_malformed = int(df["is_malformed"].sum())
    n_dupes = int(df["is_duplicate_normalized"].sum())
    n_observable = int(df["conversation_observable"].sum())
    n_not_observable = n - n_observable
    n_disclosure_required = int(df["requires_explicit_disclosure"].sum())

    type_counts = Counter(df["facet_type"])
    malformed_reason_counts = Counter(df.loc[df["is_malformed"], "malformed_reason"])
    abstention_reason_counts = Counter(df.loc[df["abstention_reason"].notna(), "abstention_reason"])
    sensitivity_counts = Counter(df["sensitivity"])
    transform_counts = Counter()
    for t in df["normalization_transforms"]:
        if t:
            for part in t.split(";"):
                transform_counts[part] += 1

    lines = []
    lines.append("# Facet Audit Summary\n")
    lines.append(f"Total raw rows: **{n}**\n")
    lines.append("## Facet type distribution\n")
    for k, v in type_counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Malformed / header-artifact rows\n")
    lines.append(f"Flagged malformed: **{n_malformed}** ({n_malformed/n:.1%})\n")
    for k, v in malformed_reason_counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Normalization transforms applied\n")
    for k, v in transform_counts.most_common():
        lines.append(f"- `{k}`: {v} rows")
    lines.append("")
    lines.append("## Duplicate normalized values\n")
    lines.append(f"Rows whose normalized form duplicates an earlier row: **{n_dupes}**\n")
    dupe_rows = df[df["is_duplicate_normalized"]][["facet_id", "raw_value", "duplicate_of_facet_id"]]
    for _, r in dupe_rows.iterrows():
        orig = df.loc[df["facet_id"] == r["duplicate_of_facet_id"], "raw_value"].values
        orig_val = orig[0] if len(orig) else "?"
        lines.append(f"- `{r['facet_id']}` \"{r['raw_value']}\" duplicates `{r['duplicate_of_facet_id']}` \"{orig_val}\"")
    lines.append("")
    lines.append("## Conversation-observable vs not\n")
    lines.append(f"- conversation_observable=True: **{n_observable}** ({n_observable/n:.1%})")
    lines.append(f"- conversation_observable=False: **{n_not_observable}** ({n_not_observable/n:.1%})")
    lines.append(f"- of the observable set, requires_explicit_disclosure=True (self-report only, not inferable from tone/behavior): **{n_disclosure_required}**")
    lines.append("")
    lines.append("## Abstention reasons (facets excluded from scoring by taxonomy gate)\n")
    for k, v in abstention_reason_counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Sensitivity distribution\n")
    for k, v in sensitivity_counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Sample of each facet_type\n")
    for t in type_counts:
        sample = df[df["facet_type"] == t]["raw_value"].head(4).tolist()
        lines.append(f"**{t}**: " + "; ".join(f'"{s}"' for s in sample))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    raw_df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    enriched = build_enriched_table(raw_df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUT_CSV, index=False, encoding="utf-8")
    write_audit_summary(enriched, OUT_AUDIT)
    print(f"Wrote {len(enriched)} rows -> {OUT_CSV}")
    print(f"Wrote audit summary -> {OUT_AUDIT}")


if __name__ == "__main__":
    main()
