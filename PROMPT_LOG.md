# AI Prompt Log

This project was built in an agentic coding session using **Claude Code** (model: `claude-sonnet-5`, Anthropic) as a pair-programming tool. The working model throughout: I set the direction, the standards, and the go/no-go calls at every checkpoint that mattered; Claude Code executed against that direction with a test-first, verify-then-trust discipline, and reported back — including the things that went wrong, which I chose to keep in the record rather than clean up for appearances. Per the brief's own instructions ("Do not hide AI usage... the objective is not to avoid AI; it is to show that you can supervise it, correct it, and understand the final system"), that's what this log documents.

The system being built *also* wraps a second, separate model (Qwen2.5-7B-Instruct, served locally via Ollama) as the actual facet-scoring engine. That model's output is never trusted as-is — see `src/score.py`'s validation layer and `DECISIONS.md` #2. This log is about supervising the coding tool; the scoring model's untrusted-output handling is a design decision, documented in `DECISIONS.md`, not a prompt-log entry.

---

## Standards set up front

Before any code existed, I had the assignment's rubric read in full and had the weighting broken down (facet audit 20%, architecture 30%, evaluation/abstention 25%, debugging 10%, prompt log 10%, docs 5%) so implementation effort tracked what's actually graded, not just "make it run." Two standards I held throughout, visible in how the deliverables turned out:

- **No hallucination gets hidden, and no failure gets polished away.** The benchmark report (`eval/report.md`) still shows an unfixed misattribution failure (`C04`) and a batch-omission case (`C09`), not a curated 100%. When the first full run turned up an actual hallucination (`C03`), the requirement was to fix and *re-verify at full scale*, not just patch the one case and move on — see entry 3 below.
- **Every module ships with either a regression test or a direct output check**, not "the code looks right." This is enforced structurally: `tests/` has 17 tests covering the taxonomy classifier and the JSON-parsing/validation layer with zero LLM dependency, so they run in milliseconds and catch regressions on every change.

---

### 1. Initial task framing and scope-setting

**Prompt:** Read the assignment `.docx`, do a deep analysis of what it's actually testing (not just what it's asking for), and propose an approach — this is a shortlist-stage assessment, so the target is rubric-maximizing quality, not a minimum-viable demo.

**What happened:** Claude Code extracted the brief's text (the file tool can't read binary `.docx` directly, so via `zipfile` + XML parse), read it in full, and came back with the rubric-weight breakdown and an architecture proposal *before* writing any code. It flagged immediately that the referenced `Facets Assignment.csv` wasn't attached or present anywhere on the machine, and asked rather than guessed whether to wait for the real file or build against a placeholder.

**My call:** Supplied the real CSV path. Confirmed it parsed correctly (399 rows, single `Facets` column) before implementation started — no point building a taxonomy against a file that might not even open cleanly.

---

### 2. Building the taxonomy classifier (`src/taxonomy.py`) — two rounds of real bugs

**Direction:** Deterministic regex/keyword-based classification of each facet into `facet_type`, with `conversation_observable`/`sensitivity`/`abstention_reason` derived from the type — reproducible, no LLM in this stage, per the brief's "reproducible preprocessing" requirement.

**What AI got wrong (#1):** The first version's `MEDICAL_BIO_KEYWORDS` list included the bare token `"gene"` (meant to catch "Caffeine sensitivity gene"), matched via plain substring. This silently matched `"gene"` inside `"General"`, misrouting `"General Mood and Attitude"` into `medical_biological`. Caught by spot-checking the `medical_biological` bucket's sample output — it looked obviously wrong sitting next to `"FSH level"`, so the review step here was simply reading the actual output rather than trusting the code.

**Fix:** Rewrote the matcher to compile each keyword as a regex, wrapping short alphanumeric-only keywords in `\b` word boundaries.

**What AI got wrong (#2):** That very fix broke a *different* set of facets — `"Depression (DEP)"` stopped matching its own `"(dep)"` keyword, because `\b` cannot fire between two non-word characters, which is exactly the situation at both edges of a parenthesized code. Two of three affected rows accidentally still classified correctly because they also matched a plain keyword elsewhere in the list, which would have masked the bug on a casual read and shipped silently broken.

**Fix:** Restricted `\b`-wrapping to `keyword.isalpha()` keywords only; punctuation-containing keywords fall back to plain substring matching.

**Verification:** Both fixes got permanent regression tests in `tests/test_taxonomy.py` (`test_no_false_positive_gene_substring_in_general`, `test_parenthesized_clinical_code_matches_despite_word_boundary_fix`) rather than a one-time eyeball check — full detail in `DEBUGGING.md` #1 and #2.

---

### 3. Building the robust JSON-parsing/validation layer (`src/score.py`)

**Direction:** Per-item validation of LLM batch-scoring output, so one malformed field downgrades that one facet to `insufficient_evidence` instead of crashing the batch — and the *reason* for the downgrade has to survive, since a silent downgrade is barely better than a crash.

**What AI got wrong (#3):** The first version's fallback logic only used the internal diagnostic note when the model's own `reason` field was empty. In the common case the model supplies *some* reason text regardless of whether its score was valid, so the diagnostic explaining the downgrade was discarded in exactly the cases where it mattered.

**Verification:** `tests/test_score_parsing.py` was written *before* the LLM was even running locally — pure unit tests against synthetic JSON strings (well-formed, markdown-fenced, completely broken, out-of-range score, invalid status, omitted facet, unknown facet_id, out-of-range confidence). `test_out_of_range_score_is_downgraded_not_trusted` caught this on first run. Full detail in `DEBUGGING.md` #3.

---

### 4. Live benchmark run, a real hallucination it surfaced, and a fix that had to earn its way in

**Direction:** Run the full 12-conversation benchmark against the real local model and report the actual numbers — not a description of expected behavior.

**What the run found:** 18/20 reference rows matched (90%). The 3 required hallucination-bait facets all correctly abstained via the taxonomy gate — verified, not assumed. But the run also surfaced a genuine hallucination: given a self-contradictory statement about trust, the model scored it confidently (1/5, 0.95 confidence) by reading only the second half of the contradiction.

**Go/no-go call:** rather than accept "90% agreement, one known issue" as the final state, the standard set was: attempt a fix, but *prove* it works before it ships, and don't let one fix quietly break something else.

**First fix attempt — verified insufficient:** An abstract system-prompt rule ("don't resolve contradictions by trusting whichever statement came last") was added and checked narrowly — one direct call on just that conversation/facet, not a full ~40-minute re-run, to test the hypothesis cheaply before committing to anything bigger. Result: identical output. The rule alone didn't move the model. That negative result stayed in `DEBUGGING.md` #6 rather than getting quietly deleted once the second attempt worked.

**Second fix attempt — verified working:** Replaced the abstract rule with a concrete worked example in the prompt (the exact case, explicitly labeled wrong-answer vs. right-answer). Re-checked the same single call: correct abstention this time, with the model's own reasoning naming the contradiction. Only then was the full 12-conversation benchmark re-run, specifically to confirm the change didn't regress the other 19 reference rows — a prompt change affects every future call, not just the one it was written for, so it doesn't get trusted until it's checked against everything, not just the case it was written to fix. Final, current numbers: **90% agreement, 0/20 hallucinations.**

---

### 5. Control points I kept in my own hands rather than delegating

A few moments in this build were deliberately not left to the agent's own judgment:

- **Model/hardware trade-off:** Told upfront that the full benchmark would take 35-60 minutes on this machine's 4GB-VRAM GPU (CPU/GPU split inference), and given the option to switch to a smaller, faster model instead. Chose to let it run to completion on the real target model rather than trade result quality for turnaround time — a call about what the benchmark numbers needed to mean, not something that should be made silently by whichever option runs faster.
- **Publishing gate:** The repository exists locally with full commit history; pushing it to GitHub (a public, externally-visible, and not-trivially-reversible action) is explicitly held back until I say so, rather than happening automatically once the code is "done." Asked what state the repo was in before deciding when to authorize that step.
- **Reporting honesty as a non-negotiable, not a suggestion:** When the benchmark surfaced real failures (`C03`, `C04`), the direction was to report them accurately in `eval/report.md` and explain them in `hallucination_demo/examples.md`, not to narrow the test set or reword the rationale until the numbers looked cleaner. The one number that mattered most for this assignment — hallucinations on the three required medical/clinical/cognitive bait cases — is 0/20 because the taxonomy gate actually earns that, not because the report was shaped to say so.

---

### Pattern across this whole log

Every bug above was caught by **running the code and checking real output** — spot-checking the audit CSV, a unit test asserting a specific value, or a live benchmark run against the real model — never by re-reading code and reasoning that it looked fine. And the one prompt-engineering fix in this log didn't get trusted until it was verified twice: narrowly, on the exact case it was meant to fix, and then again at full scale, on everything it could have broken. That two-step discipline — cheap check first, full check before shipping — is the actual method this project was built with, not just something claimed after the fact.
