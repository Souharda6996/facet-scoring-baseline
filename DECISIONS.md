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

## 5. Embedding text format: name + facet_type context, tested against bare name — result was inconclusive, and honestly reported as such

**Problem:** `src/embed_index.py` embeds each facet as `"{normalized_value} ({facet_type})"` rather than just the bare name, on the theory that the extra context word (e.g. "(medical biological)") helps the embedding model place topically related facets closer together. This was an assumption made at design time, not something validated before shipping.

**Options considered:**
- **Bare facet name only.** Simpler, shorter, no assumption baked in.
- **Name + facet_type context (originally shipped as the default).** More semantically explicit, on the (untested) theory that it helps.

**What I did:** Rather than leave the assumption unchecked, I ran `eval/retrieval_ablation.py` — a same-day ablation (pure embedding computation, no LLM calls) comparing both formats' retrieval rank against the 20 ground-truth (conversation, facet) pairs in `eval/reference_labels.csv`. Full results in `eval/retrieval_ablation_report.md`.

**Result (reported honestly, not spun):** The bare-name format actually ranked slightly *better* on this reference set — mean rank 48.9 vs. 59.2, better on 9/20 pairs vs. 6/20 for the name+type format (5 tied). The name+type format did noticeably worse specifically on the low-evidence conversation (`C07`) — plausibly because appending the same generic type-context phrase to every facet of a given type (e.g. "(personality trait or disposition)" on hundreds of rows) makes those facets look more similar to each other and to generic small-talk text, diluting the name's own discriminating signal exactly when there's no strong topical anchor to rely on instead.

**Choice:** Kept the name+facet_type format as the shipped default rather than switching on this result. With n=20 and a 20-catalogue-wide reference set, this difference is directional evidence, not a statistically decisive result — switching a production default on a single small ablation would be overreacting to noise. But the assumption is no longer *unvalidated* — it's now a flagged, data-informed open question rather than a silent guess.

**Trade-off:** Choosing not to act immediately on suggestive-but-inconclusive data, in exchange for not thrashing a default based on 20 data points. The correct next step (noted in README "what I'd improve with another day") is a larger reference set before deciding either way with real confidence -- this ablation's value is in having *asked the question* and recorded a real answer, not in having produced a final verdict.

---
