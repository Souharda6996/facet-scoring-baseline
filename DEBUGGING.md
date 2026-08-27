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

## 3. Score-downgrade diagnostic note silently discarded when the model also supplied its own `reason` text

**Symptom:** Writing `tests/test_score_parsing.py` before the LLM was even online (pure unit tests against `parse_batch_response`), a test asserting that an out-of-range score (`"score": 9`, outside the 1-5 scale) produces a `reason` mentioning `invalid_score_value` failed: the actual `reason` field just contained `"y"` -- the model's own (fabricated, in this synthetic test) reason string, not the diagnostic explaining why the score was rejected.

**Diagnosis:** In `_coerce_result_item()` (`src/score.py`), when a score/status is downgraded, a local `note` variable is set to something like `"invalid_score_value:9"`. But the final `reason` assignment was `reason = item.get("reason") if item.get("reason") else (note or "")` -- i.e. it only fell back to `note` when the model's own `reason` field was empty. Since the model almost always fills in *some* `reason` text, the coercion note was getting silently overwritten in the common case, not just the edge case.

**Root cause:** The fallback logic had the priority backwards for a downgrade scenario: it treated the model's self-reported justification as authoritative over the pipeline's own record of *why it stopped trusting the model's structured fields*. That's exactly backwards -- the whole point of the coercion layer is to expose when the model's output couldn't be trusted as-is.

**Fix:** Changed the logic so that when a coercion `note` exists, it always appears in the final `reason` (prefixed, with the model's original text appended in parentheses for context), instead of being conditionally overwritten.

**Verification:** `tests/test_score_parsing.py::test_out_of_range_score_is_downgraded_not_trusted` now passes; full suite is 17/17 passing (`pytest tests/`).

---

## 4. First live end-to-end run silently discarded an entire 8-facet batch

**Symptom:** Ran `evaluate_conversation()` live (not a unit test) on a clearly risk-taking conversation, forcing `Risktaking` (F0000) into the candidate set. The whole call took 296.5s. Every one of the first 8 facets came back `status="insufficient_evidence"`, `confidence=0.0`, `evidence=None` -- including `Risktaking`, despite the conversation being about as unambiguous a risk-taking statement as could be written ("I never think twice before saying yes to something crazy"). Only the second batch (3 facets) returned real, well-reasoned verdicts.

**Diagnosis:** `confidence=0.0` + `evidence=None` across an *entire* batch is exactly the fallback pattern `parse_batch_response()` produces on total call/parse failure (see the "Guarantee every requested facet has a row" and `LLMCallError` branches in `src/score.py`) -- not what a real LLM response looks like even when it abstains (real abstentions from the model carry a `confidence` and an `evidence`/`reason` string, as seen in the second batch: e.g. `"No explicit encouragement of others to participate."`). This pointed at the whole first `call_llm()` invocation failing, not at the model choosing to abstain on 8 facets in a row.

**Root cause:** Reproduced directly by calling `call_llm()` on the exact same 8-facet prompt with a stretched-out 300s timeout and 0 retries: generation actually took **~144 seconds**. The default timeout at the time was `120s` with `retries=1` (so the first attempt timed out at 120s, the retry then also ran into the same ~144s generation time and timed out again) -- on this machine's RTX 3050 (4GB VRAM), Qwen2.5-7B runs a 52%/48% CPU/GPU split (confirmed via `ollama ps`), so an 8-facet batch's full JSON generation routinely exceeds 120s. The pipeline's own retry-then-abstain safety net (correctly) caught the failure and avoided a crash -- but the *default* timeout was tuned for faster inference than this hardware actually delivers, so real, well-reasoned model output was being thrown away as a false negative.

**Fix:** Raised `call_llm()`'s default `timeout` from 120s to 240s (`src/llm_client.py`), and lowered `pipeline.py`'s default `DEFAULT_BATCH_SIZE` from 8 to 5 -- smaller batches generate less JSON per call, keeping generation time comfortably under the new timeout on this hardware without going back to a call-per-facet design (which the brief explicitly disallows). Both are documented as hardware-fit tuning knobs, not architectural constants -- a faster GPU or a hosted endpoint could safely raise the batch size back up.

**Verification:** Re-ran the same 8-facet prompt directly against `call_llm()` with `timeout=300, retries=0` and confirmed it now returns well-formed, fully reasoned JSON for all 8 facets in ~144s (comfortably under the new 240s default) -- e.g. `Adventure-Seeking Behavior` correctly scored 5/5 with evidence quoting the bungee/skydiving line, and topically irrelevant facets like `Buddhist practice` correctly returned `not_observable` with a real reason, not the blanket-failure fallback pattern.

---
