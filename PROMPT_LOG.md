# AI Prompt Log

This project was built in an agentic coding session using **Claude Code** (model: `claude-sonnet-5`, Anthropic). The assistant had shell, file-edit, and test-execution access and worked iteratively: implement -> run -> spot-check/test -> fix. This log records the material prompts/turns and, per the brief's requirement, is honest about what the AI got wrong along the way and how it was caught and corrected.

The system being built *also* wraps a second, separate model (Qwen2.5-7B-Instruct, served locally via Ollama) as the actual facet-scoring engine. That model's output is never trusted as-is — see `src/score.py`'s validation layer and `DECISIONS.md` #2. This log is about supervising the *coding* assistant (Claude Code); the scoring model's untrusted-output handling is a design decision, not a prompt-log entry.

---

### 1. Initial task framing

**Prompt (user):** "Read the assignment docx, do deep analysis, tell me what to do — this is an assessment for a company I'm shortlisted at, give your best output."

**What I did:** Extracted text from the `.docx` (python `zipfile` + XML parse, since the file tool can't read binary `.docx` directly), read the full brief, identified the grading rubric weights, and produced an architecture plan before writing any code. Flagged that the referenced `Facets Assignment.csv` wasn't attached/present anywhere on the machine, and used `AskUserQuestion` to check whether to wait for it or build against a synthetic placeholder, rather than guessing.

**Verification:** User supplied the real CSV path (`E:\Ahoum AIML\Facets Assignment.csv`) in the next turn; confirmed it parsed correctly (399 rows, single `Facets` column) before proceeding.

---

### 2. Building the taxonomy classifier (`src/taxonomy.py`)

**Prompt (self-directed, per the architecture plan):** Implement deterministic regex/keyword-based classification of each facet into `facet_type`, with `conversation_observable`/`sensitivity`/`abstention_reason` derived from the type.

**What AI got wrong (#1):** The first version's `MEDICAL_BIO_KEYWORDS` list included the bare token `"gene"` (intended to catch "Caffeine sensitivity gene"), matched via plain substring (`k in low`). This silently matched `"gene"` inside `"General"`, misrouting `"General Mood and Attitude"` into `medical_biological`. Caught by manually spot-checking the `medical_biological` bucket's sample output in the audit summary — it looked obviously wrong next to `"FSH level"`.

**What I corrected:** Rewrote the matcher to compile each keyword as a regex, wrapping short alphanumeric-only keywords in `\b` word boundaries.

**What AI got wrong (#2):** That very fix broke a *different* set of facets — `"Depression (DEP)"` stopped matching its own `"(dep)"` keyword, because `\b` cannot fire between two non-word characters (a space and `"("`), which is exactly the situation at both edges of a parenthesized code. Two of three affected rows (`Hypomania (Ma)`, `Hysteria (Hy)`) accidentally still classified correctly because they *also* matched a plain keyword (`"hypomania"`, `"hysteria"`) elsewhere in the list — which masked the bug on a casual read of the output and would have shipped silently broken.

**What I corrected:** Restricted `\b`-wrapping to `keyword.isalpha()` keywords only; punctuation-containing keywords fall back to plain substring matching.

**How I verified both:** Wrote `tests/test_taxonomy.py` with regression tests pinned to both exact failure cases (`test_no_false_positive_gene_substring_in_general`, `test_parenthesized_clinical_code_matches_despite_word_boundary_fix`) rather than just eyeballing the fix once. Full detail in `DEBUGGING.md` #1 and #2.

---

### 3. Building the robust JSON-parsing/validation layer (`src/score.py`)

**Prompt (self-directed):** Implement per-item validation of LLM batch-scoring output so a malformed field (bad status/score/confidence) downgrades that one facet to `insufficient_evidence` instead of crashing the batch, while preserving *why* it was downgraded.

**What AI got wrong (#3):** The first version's fallback logic for the `reason` field was `item.get("reason") if item.get("reason") else (note or "")` — i.e. it only used the internal diagnostic `note` (e.g. `"invalid_score_value:9"`) when the model's own `reason` field was *empty*. In the common case, the model supplies some `reason` text regardless of whether its score/status was valid, so the diagnostic explaining the downgrade was silently discarded in exactly the cases where it mattered most.

**What I corrected:** Changed the priority so the diagnostic note always appears when a coercion happened, with the model's own text appended for context rather than replacing it.

**How I verified:** Wrote `tests/test_score_parsing.py` *before* the LLM was even running locally (pure unit tests against synthetic JSON strings covering: well-formed, markdown-fenced, completely broken, out-of-range score, invalid status, omitted facet, unknown facet_id, out-of-range confidence). `test_out_of_range_score_is_downgraded_not_trusted` caught this bug on first run. Full detail in `DEBUGGING.md` #3.

---

### 4. Live benchmark run + iterating on a real failure it surfaced

**Prompt (self-directed):** Run the 12-conversation benchmark end to end (`eval/run_eval.py`) against the real local model and report actual numbers, rather than describing expected behavior.

**What the run found:** 18/20 reference rows matched (90%). The 3 required hallucination-bait facets (medical/clinical/cognitive) all correctly abstained via the taxonomy gate — verified, not assumed. But two genuine failures also surfaced, and I did not filter them out of the report: a misattribution error (`C04`, the system scored the *speaker's* Risktaking off a *quoted* sentence belonging to their sister — its own logged reason literally says "the sister's statement clearly indicates...") and one real `HALLUCINATION`-classified row (`C03`, a self-contradictory trust statement scored confidently instead of triggering abstention).

**What I did about `C03`, and what AI got wrong (#4):** Rather than just writing the failure into the report, I attempted a fix: added an explicit system-prompt rule telling the model not to resolve direct self-contradictions by trusting whichever half came last. I verified this narrowly (one direct `score_batch()` call on just that conversation/facet, not a full ~40-minute benchmark re-run) before deciding whether to roll it out — **and the fix did not work**. Identical output, same score, same confidence. An abstract instruction alone didn't change the model's behavior on this specific pattern.

**What I corrected:** Replaced the abstract rule with a concrete worked example in the system prompt (the exact conversation, explicitly labeled wrong-answer vs. right-answer). Re-verified the same single call: this time the model correctly abstained, citing the contradiction itself as the reason. Then re-ran the *full* 12-conversation benchmark (not just the one facet) to confirm the prompt change didn't regress any of the other 19 reference rows, since a prompt change affects every future call, not just the one it was written for.

**How I verified:** Direct, narrow verification before committing to a full expensive re-run (checking the hypothesis cheaply first), then a full re-run before shipping the change (checking it didn't break anything else). Both attempts and the reasoning for why the first didn't work are recorded in `DEBUGGING.md` #6, not just the version that worked.

---

### Pattern across all four corrections

In each case the issue was caught by **running the code and checking real output** (spot-checking the audit CSV, a unit test asserting a specific expected value, or a live benchmark run against the real model) — not by re-reading the code and reasoning that it looked right. That's the supervision method used throughout this project: every module has either a regression test or a direct output inspection step attached to the turn that wrote it, and the one prompt-engineering fix in this log was verified narrowly before being trusted, and re-verified at full scale before being shipped.
