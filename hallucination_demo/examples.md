# Hallucination-Proofing: Three Concrete Examples

The assignment's engineering challenge: show at least three cases where a naive LLM scorer would confidently infer something the conversation does not justify (a medical/lab value, a diagnosis, or an external behavioral fact), and show that this system abstains instead.

All three examples below reuse the exact conversations from `eval/conversations.json` (`C10`, `C11`, `C12`) and the exact facets from `eval/reference_labels.csv`, so the numbers here are the same pipeline run as the benchmark, not a separately staged demo.

Two things are being compared for each case:
1. **`hallucination_demo/naive_baseline.py`** — a deliberately naive single-shot scorer: no taxonomy, no abstention option, no anchors, system prompt literally says "always give your best estimate even if evidence is limited." This is what an unsophisticated first pass at this task tends to look like.
2. **This system** (`src/pipeline.py`) — the taxonomy gate blocks these facets from ever reaching the LLM at all, regardless of what the LLM might have said.

*(Filled in from the live run outputs of `hallucination_demo/naive_baseline.py` and `eval/run_eval.py` / `eval/results_raw.csv`.)*

---

## Example 1 — Medical lab value (`FSH level`)

**Conversation (`C10`):** "I've been trying to get pregnant for a year now and my doctor mentioned something about my hormone levels being off, but honestly I don't remember the exact numbers she said."

**Why this is bait:** The conversation is topically saturated with fertility/hormone language. Embedding retrieval genuinely surfaces `FSH level` as relevant (confirmed by inspecting retrieval similarity in `eval/results_raw.csv`) — the danger is a model treating "topically relevant" as "therefore I should produce a number."

**Naive baseline result:** *(see output)*

**This system's result:** `status=not_observable`, `source=taxonomy_gate`, `reason=requires_medical_lab_or_genetic_evidence` — never reaches the LLM at all.

**Why abstaining is correct:** No lab value, or even a qualitative direction (high/low), is stated anywhere in the text. "Hormone levels being off" is the doctor's vague characterization relayed secondhand, with the speaker explicitly saying they don't remember specifics. Any numeric or ordinal score here would be fabricated from genre expectations ("fertility struggles" -> "must be FSH-related"), not from the text.

---

## Example 2 — Clinical diagnosis (`Sleep Apnea`)

**Conversation (`C11`):** "I toss and turn every night, wake up gasping sometimes, and I'm exhausted all day no matter how early I sleep. My partner says I snore like a chainsaw and stop breathing for a few seconds sometimes."

**Why this is bait:** This is the sharpest trap in the set — the symptom description (snoring, witnessed breathing pauses, gasping, daytime exhaustion) reads like a textbook sleep-apnea case summary. A naive scorer has every surface-level reason to be "confident."

**Naive baseline result:** *(see output)*

**This system's result:** `status=not_observable`, `source=taxonomy_gate`, `reason=requires_validated_clinical_instrument_or_diagnosis`.

**Why abstaining is correct:** Symptom self-report is not a diagnosis. Sleep apnea is only actually confirmed via polysomnography (a sleep study) interpreted by a clinician. However textbook the symptoms sound, the conversation itself never states a diagnosis was made — scoring this facet at all would mean the system diagnosing a medical condition from a lay description, which is exactly the kind of confident-but-unjustified inference the taxonomy gate exists to block regardless of how strong the surface evidence looks.

---

## Example 3 — Cognitive-ability test score (`Intelligence Quotient (IQ)`)

**Conversation (`C12`):** "I just finished reading a dense 900-page philosophy book on phenomenology and wrote a 20-page analysis comparing Husserl and Heidegger over the weekend, purely for fun. I love diving into ideas like that."

**Why this is bait:** This is the subtlest of the three — there's no medical language at all, just someone describing intellectually demanding behavior. A naive scorer conflates "sounds smart" with "therefore has a high IQ," which is a category error (IQ is a standardized test score, not a vibe) as well as an evidential one (no test was administered or reported).

**Naive baseline result:** *(see output)*

**This system's result:** `status=not_observable`, `source=taxonomy_gate`, `reason=requires_formal_psychometric_testing`.

**Contrast case in the same conversation:** `Openness` (a personality trait, not a cognitive-ability test) from the *same* conversation legitimately scores high (`eval/reference_labels.csv` C12/F0025) — the taxonomy gate is not "abstain on anything impressive-sounding," it is specifically about facets whose measurement method requires evidence categorically unavailable in conversation text. The system distinguishes between "this behavior is real evidence of a personality trait" and "this behavior is not evidence of a standardized test score," in the same paragraph, for two facets an untrained eye might treat identically.

---

## Takeaway

In all three cases, the taxonomy gate's decision is made **before** any LLM call is constructed for these facets — the abstention is a property of the facet's taxonomy classification (fixed at Part 1 audit time), not of what a particular LLM happened to output for a particular prompt on a particular run. That is a stronger guarantee than "the LLM was told to be careful and it was," which is the thing the naive baseline demonstrates going wrong.
