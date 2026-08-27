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

## 5. `eval/run_eval.py` crashed immediately: `reference_labels.csv` had unquoted commas inside rationale text

**Symptom:** First full run of `eval/run_eval.py` failed instantly with `pandas.errors.ParserError: Error tokenizing data. C error: Expected 6 fields in line 7, saw 7`.

**Diagnosis:** `eval/reference_labels.csv` was hand-written as plain comma-joined lines, not generated through a CSV-aware writer. Line 7 (the C04 "quoted" conversation's rationale) reads: `"...describes themself as a heavy planner -- the trait belongs to a third party, not the speaker"` -- the comma before "not the speaker" is prose punctuation, not a field separator, but the raw file had no quoting to say so, so the CSV parser split it into an extra 7th field.

**Root cause:** Authoring a CSV by hand (string-joining values with commas) instead of through `csv.writer`, for a file where free-text rationale fields were always going to contain commas. This is the same category of mistake `src/preprocess.py` was built specifically to avoid for the facet catalogue itself ("reproducible preprocessing rather than manually editing the file") -- it just resurfaced in a benchmark fixture that wasn't generated through code.

**Fix:** Rewrote `eval/reference_labels.csv` via Python's `csv.writer`, which automatically quotes any field containing a comma (visible in the resulting file: the C04, C05, C09, and C10 rationale fields are now wrapped in `"..."`).

**Verification:** `pd.read_csv('eval/reference_labels.csv')` now parses cleanly (20/20 rows); re-launched `eval/run_eval.py`.

---

## 6. Model resolves a self-contradictory statement by trusting whichever half came last, instead of abstaining

**Symptom:** First full benchmark run (`eval/report.md`) flagged one `HALLUCINATION` outcome: `C03`, a conversation reading "I trust people completely, I always give everyone the benefit of the doubt. Actually no, forget that -- I don't trust a single person anymore, everyone lies eventually." Reference expectation was `insufficient_evidence` (the statement is directly self-contradictory with no resolution). The system instead returned `status=scored, score=1, confidence=0.95`, with its own logged reason: *"The speaker explicitly states a lack of trust in others"* -- reasoning entirely from the second half of the statement and silently discarding the first half's direct contradiction.

**Diagnosis:** This is not a taxonomy-gate failure (`Trust in others` is correctly `conversation_observable=True` -- a contradictory personality statement is exactly the kind of thing the LLM, not a deterministic rule, has to judge). It's a prompt-following gap: nothing in `SYSTEM_PROMPT` (`src/score.py`) told the model that a *live, unresolved* contradiction should itself be read as ambiguity rather than resolved by recency.

**Root cause:** Qwen2.5-7B defaulted to a recency heuristic ("the last thing said is the real stance") instead of treating direct self-contradiction as a distinct signal calling for abstention -- unsurprising, since the original prompt never named that case at all.

**Fix attempt #1 (verified insufficient):** Added an explicit rule to `SYSTEM_PROMPT`: "if the conversation makes directly contradictory statements about the same facet ... do NOT resolve this by trusting whichever statement came last ... return insufficient_evidence." Re-ran *only* this facet/conversation pair directly against `score_batch()` (not the full 12-conversation benchmark, to avoid a ~40min re-run for a one-line prompt change) to check the fix before deciding whether to roll it out. Result: **identical output** -- still `scored, 1, 0.95`, same reasoning. An abstract rule alone did not change the model's behavior. This negative result is reported here rather than discarded, because "I tried a fix and verified it didn't work" is itself a real finding about this model's instruction-following depth on abstract multi-step rules.

**Fix attempt #2 (verified working):** Replaced the abstract rule with a concrete worked example appended to `SYSTEM_PROMPT`: the exact same conversation/facet pair, showing the wrong answer (recency-biased scoring) explicitly labeled wrong with a one-line explanation, followed by the correct answer (`insufficient_evidence`, with reasoning citing the contradiction itself). Re-ran the identical single-facet call: now returns `status=insufficient_evidence, confidence=0.9`, with reason *"self-contradictory, no reliable direction can be read from this conversation"* -- the model correctly named the actual problem instead of picking a side.

**Why fix #2 worked where #1 didn't:** Consistent with `DECISIONS.md` #3's trade-off (a 7B model was chosen for hardware fit, at a known cost in reasoning depth vs. a larger model) -- abstract multi-step instructions ("detect X, then when X is true, override your default Y") are exactly the kind of instruction-following a smaller model tends to struggle with, while a concrete right-vs-wrong example for the specific pattern is something even a smaller model can pattern-match against directly. This is a real, observed manifestation of that documented trade-off, not a hypothetical one.

**Verification:** Direct `score_batch()` call on the exact `C03`/`Trust in others` pair now returns `insufficient_evidence` as shown above. The full benchmark was re-run end-to-end after this fix (see updated `eval/report.md`) to confirm the fix doesn't regress any of the other 19 reference rows and to keep the shipped report consistent with the current code.

---

## 7. Adding a new report column broke the markdown table's pipe separators, silently merging two columns

**Symptom:** While adding the "retrieved organically?" column to `eval/report.md`'s row-by-row table (see `DECISIONS.md` #6), the freshly regenerated table rendered with a broken header: `"...outcome | detail  retrieved organically? |"` -- two spaces and no `|` between `detail` and the new column, meaning the two columns would render as one merged cell instead of two separate ones.

**Diagnosis:** The header/row construction used `header[:-1] + " retrieved organically? |"` -- i.e. "strip the last character of the existing string, then append the new column text." The last character of `"...| detail |"` is `|`, so stripping it left `"...| detail "` (trailing space, no pipe), and appending `" retrieved organically? |"` produced `"...| detail  retrieved organically? |"` with the separating pipe simply gone. The same off-by-one string-slicing mistake was made in three places: the header, the separator row (`|---|---|...`), and every data row.

**Root cause:** Treating "add a column to a markdown table" as a string-suffix-editing problem (`[:-1] + new_suffix`) instead of a structural one. String slicing on `"...| X |"` to insert a new trailing column needs to insert *before* the final `|`, not remove it and hope the replacement re-supplies an equivalent one -- the replacement text didn't start with `|`, so no separator existed between the old last column and the new one.

**Fix:** Rewrote the table-building code in `eval/run_eval.py`'s `write_report()` to construct each row from an explicit list of cell values joined with `" | "` and wrapped in `"| ... |"` (`"| " + " | ".join(cells) + " |"`), for both the header and every data row, instead of string-slicing an already-formatted line. This is correct by construction regardless of how many columns exist, rather than being correct only for the exact column count it was originally written for.

**Verification:** Regenerated `eval/report.md` via `python eval/run_eval.py --report-only` and inspected the actual output: the header now reads `"| conv | facet | ... | detail | retrieved organically? |"` with a proper `|` between every column, and every data row has exactly 9 pipe-separated cells matching the 9-column header.

---

## 8. `docker build` had never actually been run -- it pulled ~16GB of unnecessary CUDA libraries and stalled on layer export

**Symptom:** The Dockerfile existed and looked reasonable but had never been executed against a real Docker daemon. Running `docker build -t facet-scoring-baseline .` for the first time: the `pip install -r requirements.txt` step took over 500 seconds and pulled a long list of `nvidia-cu*` packages (`nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-cufft`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-cusparse`, `nvidia-nccl-cu13`, and more), and the final `exporting layers` step then stalled indefinitely while the host's free disk space dropped from ~23GB to ~15GB during the build.

**Diagnosis:** `docker system df` showed the build had produced ~16GB of images and ~10GB of build cache. `docker images` and the pip install log made the cause obvious: `sentence-transformers` depends on `torch`, and `pip install torch` on a plain `python:3.11-slim` image defaults to the full CUDA-enabled wheel (multiple GB of NVIDIA driver libraries), regardless of whether the container will ever have GPU access.

**Root cause:** This container was never going to use a GPU -- the only ML inference it performs is encoding short facet/conversation strings with `all-MiniLM-L6-v2` (a 22M-parameter model that runs trivially fast on CPU), and the actual LLM scoring happens in a separate Ollama process outside the image entirely (`OLLAMA_URL`, already documented in the Dockerfile). Nothing in `requirements.txt` or the Dockerfile constrained `pip` to a CPU-only torch build, so it silently took the much larger default.

**Fix:** Added a `RUN pip install torch --index-url https://download.pytorch.org/whl/cpu` step in the Dockerfile *before* `pip install -r requirements.txt`, using PyTorch's own CPU-only wheel index. When `requirements.txt` is installed afterward, `torch` is already satisfied and `sentence-transformers` doesn't trigger a second, CUDA-enabled install.

**Verification:** Stopped the original stalled build, ran `docker builder prune -af` to reclaim the ~6GB of build cache it had produced (confirmed via `docker system df` before/after, and confirmed via `docker images` that no partial images were left behind -- the only images present were the host machine's pre-existing, unrelated containers), then re-ran `docker build` with the CPU-only torch step added. The rebuilt image finished in full (`facet-scoring-baseline:latest`, 2.25GB -- down from a build that was on track for several times that) and, critically, actually **runs**: `docker run --rm facet-scoring-baseline` executes the image's default command (`pytest tests/ -v`) inside the container and all 17 tests pass, confirming the image isn't just buildable but functional end to end.

---
