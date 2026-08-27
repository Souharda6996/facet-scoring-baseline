# Debugging Log

Real issues hit during development, not hypothetical ones. Each entry: symptom -> diagnosis -> root cause -> fix -> verification.

---

## 1. Keyword-based facet classifier mis-routed "General Mood and Attitude" into `medical_biological`

**Symptom:** After the first run of `src/preprocess.py`, spot-checking the `medical_biological` bucket in `data/processed/facets_enriched.csv` showed `"General Mood and Attitude"` sitting next to `"FSH level"` and `"Parathyroid-hormone level"` — clearly wrong; it's a plain personality/mood facet, not a lab value.

**Diagnosis:** Traced `classify_facet_type()` in `src/taxonomy.py`. `MEDICAL_BIO_KEYWORDS` contains the token `"gene"` (meant to catch "Caffeine sensitivity gene", "genetic risk", etc.), and the original `_matches_any()` did a naive `k in low` substring check.

**Root cause:** `"gene"` is a substring of `"General"`. `"general mood and attitude".lower()` contains `"gene"` literally (Ge**ne**ral), so the naive substring match fired on an unrelated word.

**Fix:** Rewrote `_matches_any` to compile each keyword into a regex, wrapping short alphanumeric-only keywords (<=6 chars, no spaces/punctuation) in `\b...\b` word boundaries, while leaving multi-word phrases and punctuation-containing keywords as plain substring matches (see next entry for why that split was necessary).

**Verification:** Added `tests/test_taxonomy.py::test_no_false_positive_gene_substring_in_general`, which asserts `"General Mood and Attitude"` classifies as `personality_trait_or_disposition`. Re-ran `preprocess.py` and confirmed the row moved buckets in the output CSV.

---

## 2. The word-boundary fix for #1 silently broke parenthesized clinical codes

**Symptom:** After deploying the `\b`-wrapping fix above, `"Depression (DEP)"` was *still* misclassified — it fell all the way through to `personality_trait_or_disposition` instead of `clinical_psychological_scale`, even though `"(dep)"` is explicitly listed in `CLINICAL_KEYWORDS`. Curiously, `"Hypomania (Ma)"` and `"Hysteria (Hy)"` classified correctly, which masked the bug on first glance since two of the three parenthesized-code rows still worked.

**Diagnosis:** Interactively tested the compiled patterns: `re.compile(r"\b\(dep\)\b").search("depression (dep)")` returned `None`. `"Hypomania (Ma)"` only worked by accident — it also matched the plain word `"hypomania"` elsewhere in `CLINICAL_KEYWORDS`, not the `"(ma)"` code.

**Root cause:** `\b` is a transition between a `\w` and a `\W` character. `"("` and `")"` are both non-word characters, and the character immediately before `"("` here is a space (also non-word). So at both edges of `"(dep)"` the assertion sits between two non-word characters and never fires — `\b` mechanically cannot match around a token that starts and ends with punctuation.

**Fix:** Changed the keyword-pattern rule to only apply `\b`-wrapping when `keyword.isalpha()` (pure letters, no punctuation) — punctuation-containing keywords like `"(dep)"`, `"(hy)"` fall back to plain substring matching, which is safe for them because parenthesized short-codes are distinctive enough not to collide with unrelated words.

**Verification:** Added `tests/test_taxonomy.py::test_parenthesized_clinical_code_matches_despite_word_boundary_fix`, asserting all three of `"Depression (DEP)"`, `"Hypomania (Ma)"`, `"Hysteria (Hy)"` classify as `clinical_psychological_scale`. All 9 tests in the suite pass; re-ran full `preprocess.py` and confirmed via `data/processed/AUDIT_SUMMARY.md` sample listing.

---
