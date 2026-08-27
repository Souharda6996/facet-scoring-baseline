# Design Decisions

Non-trivial decisions made during this assignment: the ambiguity/problem, options considered, the choice made, and the trade-off accepted. Decisions forced directly by the brief (e.g. "use an open-weight model <=16B") are not listed here.

---

## 1. Facet taxonomy classification: deterministic rules vs. LLM-based classification

**Problem:** 399 raw facet strings need to be classified into a taxonomy (personality trait, medical/biological, cognitive-ability, spiritual practice, etc.) so downstream scoring knows what is safe to evaluate from conversation. The brief explicitly says "you are not expected to hand-label every row" and asks for "reproducible preprocessing."

**Options considered:**
- **LLM-based classification** (few-shot prompt per facet, or a batch prompt). Handles genuinely ambiguous cases with more nuance, e.g. distinguishing "Psychoticism" (an Eysenck clinical dimension) from "Aloofness" (a plain trait word) by reasoning about connotation rather than keyword matching.
- **Deterministic regex/keyword rules** (chosen). Every classification is traceable to an explicit rule in `src/taxonomy.py`; re-running `preprocess.py` on the same CSV always produces byte-identical output; zero inference cost or API dependency for a one-time audit step.

**Choice:** Deterministic rules for Part 1. The LLM is reserved for Part 2's per-conversation scoring, where non-determinism is expected, bounded (temperature=0, still can vary slightly across runs/hardware), and grounded in the actual evidence text rather than in metadata generation.

**Trade-off:** Some genuinely borderline facets get a coarser classification than an LLM might produce — e.g. `Psychoticism` and `Attachment style: Secure` land in `personality_trait_or_disposition` even though a stricter reading might put them closer to clinical constructs. This is a known limitation (see README "known limitations"), accepted because reproducibility and zero-cost auditability of the *taxonomy itself* matters more here than marginal classification nuance — the taxonomy is infrastructure that other decisions (the abstention gate) depend on being stable and inspectable.

---

## 2. Where to enforce the "don't score non-observable facets" guarantee

**Problem:** Medical/clinical/cognitive facets (e.g. `FSH level`, `Sleep Apnea`, `Intelligence Quotient (IQ)`) must never receive an invented score. Where should that guarantee actually live?

**Options considered:**
- **(a) Exclude non-observable facets from the embedding index entirely.** Simplest — they can never be retrieved, full stop.
- **(b) Prompt-only guardrail.** Include everything in retrieval, rely on the LLM's system-prompt instructions ("abstain if evidence is medical/lab-based") to self-police.
- **(c) Two-stage gate (chosen).** Index everything except header-artifact rows, retrieve normally by embedding similarity, then run a deterministic taxonomy check on every retrieved candidate *before* any LLM prompt is constructed. Non-observable candidates are routed straight to `status="not_observable"` with `source="taxonomy_gate"`, without ever reaching the LLM.

**Choice:** (c). Option (a) would have been simpler but throws away evidence: it can't demonstrate that embeddings *would* have surfaced a medical facet as topically relevant (e.g. "FSH level" for a fertility-related conversation) — which is exactly the scenario the assignment's "prove you are not scoring hallucinations" challenge is asking to be shown, not just asserted. Option (b) was rejected outright: an LLM is stochastic and prompt instructions are not a hard guarantee — that's the failure mode the challenge exists to catch, so it can't also be the safeguard against it.

**Trade-off:** (c) costs a small amount of extra embedding compute (some facets get indexed and then immediately gated out on most conversations) versus (a), in exchange for a guarantee that is (i) deterministic and non-bypassable by anything the LLM does, and (ii) directly demonstrable — every taxonomy-blocked result is visible in the output with its specific `abstention_reason`, which is what `hallucination_demo/` relies on to show the mechanism actually firing rather than just claiming it would.

---

## 3. Model choice: Qwen2.5-7B-Instruct via Ollama

**Problem:** Need an open-weight, <=16B model runnable within 24 hours on the available hardware (RTX 3050 Laptop, 4GB VRAM, ~23GB free disk).

**Options considered:**
- **Qwen2.5-1.5B/3B-Instruct** — fast and fits comfortably in 4GB VRAM, but noticeably weaker at the reasoning the benchmark specifically demands: distinguishing sarcasm from literal statements, resolving contradictory self-reports, correctly attributing a quoted trait to a third party rather than the speaker.
- **Qwen2.5-14B-Instruct** — still within the 16B limit and stronger reasoning, but 4-bit quantized is still ~8-9GB, which does not fit the 4GB card; it would run mostly on CPU, making iteration (rebuilding the benchmark repeatedly while debugging) too slow for a 24h window.
- **Qwen2.5-7B-Instruct via Ollama (chosen)** — Apache-2.0 licensed, 4-bit quantized weight is ~4.7GB (partial GPU offload + CPU fallback on the 4GB card), and Ollama provides a working local chat+JSON-mode HTTP endpoint out of the box.
- **Raw HuggingFace `transformers` + `bitsandbytes` 4-bit** — more low-level control, but `bitsandbytes` is historically unreliable on Windows, and hand-rolling batched generation + JSON-mode + retry logic on top of raw `transformers` would spend the 24h budget on inference-serving plumbing rather than on the retrieval/gating/scoring design the assignment is actually grading.

**Choice:** Qwen2.5-7B-Instruct via Ollama.

**Trade-off:** Meaningfully less raw reasoning capability than a 14B+ model, in exchange for a system that reliably runs end-to-end on the available hardware within the time budget, with a solved inference-serving layer (batching, JSON formatting, keep-alive) that let engineering time go into `src/taxonomy.py`, `src/retrieve.py`, and `src/score.py` instead.

---

## 4. Scoring-anchor authoring: fully generic template vs. hand-written per-facet rubrics

**Problem:** A five-level ordinal scale needs anchor definitions so the LLM (and a human reviewer) know what "3" vs "5" means for a given facet. Hand-authoring a rich rubric for 399 facets is not feasible in 24 hours, and the brief explicitly does not require hand-labeling every row.

**Options considered:**
- **Hand-write anchors for every facet.** Highest quality, infeasible at this scale/time budget, and directly contradicts the brief's "not expected to hand-label every row."
- **Fully generic templated anchors (chosen).** `taxonomy.build_scoring_anchors()` substitutes the facet's normalized name into one fixed 5-level template ("clear evidence of low/absent `{facet}`" ... "clear evidence of high `{facet}`"), applied identically and reproducibly to every non-malformed row.

**Choice:** Generic template for all facets now; hand-authored anchors only where it matters, prioritized by retrieval frequency, would be the first upgrade with more time (see README "what I'd improve with another day").

**Trade-off:** Generic anchors are noticeably weaker for facets where "high" and "low" aren't self-evident opposites from the name alone (e.g. `Neuroticism`, `Discreteness`) compared to a facet like `Talkativeness` where the template reads naturally. Accepted in exchange for 100% reproducible, zero-manual-effort coverage of the full catalogue — and because this is exactly the kind of prioritization problem ("which facets deserve hand-authored rubrics?") that the retrieval-frequency signal from real usage should drive, rather than guessing upfront which of 399 facets matter most.

---
