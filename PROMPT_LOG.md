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

### Pattern across all three corrections

In each case the bug was caught by **running the code and checking real output** (spot-checking the audit CSV, or a unit test asserting a specific expected value) — not by re-reading the code and reasoning that it looked right. That's the supervision method used throughout this project: every module has either a regression test or a direct output inspection step attached to the turn that wrote it, before moving on to the next module.

*(This log is appended to as the remaining benchmark/report/README work completes.)*
