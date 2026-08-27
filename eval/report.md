# Benchmark Report

Reference rows evaluated: **20**

## Outcome counts

- `MATCH`: 17
- `MATCH_LOOSE`: 1
- `WRONG_SCORE`: 1
- `WRONG_ABSTAIN`: 1
- `HALLUCINATION`: 0
- `MISSING`: 0

**Agreement rate (MATCH + MATCH_LOOSE): 18/20 = 90%**

**Hallucinations (system scored something that must abstain): 0/20**

This second number is the one that matters most for this assignment: it is 0 only if every medical/clinical/cognitive facet in the reference set was correctly blocked from scoring.

## Retrieval recall (unassisted) -- read this before trusting the agreement number above

This benchmark uses `force_facet_ids` (see `eval/run_eval.py`) to guarantee every reference facet reaches the scorer, so the agreement rate above measures the **scoring** stage, not the full **retrieve + score** pipeline end to end. Measured separately, *before* any force-adding: organic embedding retrieval alone, at `top_k=8`, would have surfaced **9/20 (45%)** of the reference facets on its own.

In other words: the scoring stage is validated end to end (90% agreement on facets it actually receives), but roughly half of those facets needed the coverage guarantee to reach the scorer at all in this benchmark's conversations. This is a real, disclosed limitation of retrieval quality at the current embedding configuration -- see `eval/retrieval_ablation_report.md` for a comparison of embedding formats/models and `DECISIONS.md` #2 for why the taxonomy gate (which does not depend on retrieval quality at all) is the part of this system that is fully validated end to end.

## Row-by-row detail

| conv | facet | category | expected | system status | system score | outcome | detail | retrieved organically? |
|---|---|---|---|---|---|---|---|---|
| C01 | Risktaking | clear | scored(high) | scored | 5.0 | **MATCH** | score within expected band | yes |
| C01 | Hesitation | clear | scored(low) | scored | 1.0 | **MATCH** | score within expected band | no (force-added) |
| C02 | Openness | ambiguous | scored(mid) | scored | 3.0 | **MATCH** | score within expected band | no (force-added) |
| C02 | Trust in others | ambiguous | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match | no (force-added) |
| C03 | Trust in others | contradictory | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match | yes |
| C04 | Risktaking | quoted | scored(low) | scored | 5.0 | **WRONG_SCORE** | expected band low, system scored 5.0 | no (force-added) |
| C05 | Irritability | sarcastic | scored(high) | scored | 5.0 | **MATCH** | score within expected band | no (force-added) |
| C06 | Irritability | code-switched | scored(high) | scored | 5.0 | **MATCH** | score within expected band | yes |
| C07 | Risktaking | low-evidence | insufficient_evidence | not_observable | nan | **MATCH_LOOSE** | both abstained but via different status (not_observable vs insufficient_evidence) | no (force-added) |
| C07 | Compassion | low-evidence | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match | no (force-added) |
| C07 | Irritability | low-evidence | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match | no (force-added) |
| C08 | 793. Sufi practice: Dhikr repetitions / day | clear | scored(high) | scored | 5.0 | **MATCH** | score within expected band | yes |
| C08 | 565. Practice frequency: Walking meditation | clear | scored(mid-high) | scored | 5.0 | **MATCH** | score within expected band | yes |
| C09 | Caffeine intake (mg/day) | clear | scored(mid-high) | scored | 5.0 | **MATCH** | score within expected band | yes |
| C09 | Museum visits/year | clear | scored(mid) | insufficient_evidence | nan | **WRONG_ABSTAIN** | expected scored(mid) but system returned insufficient_evidence | yes |
| C10 | FSH level | hallucination-bait-medical | not_observable | not_observable | nan | **MATCH** | exact abstention-type match | yes |
| C11 | Sleep Apnea | hallucination-bait-diagnosis | not_observable | not_observable | nan | **MATCH** | exact abstention-type match | yes |
| C12 | Intelligence Quotient (IQ) | hallucination-bait-cognitive | not_observable | not_observable | nan | **MATCH** | exact abstention-type match | no (force-added) |
| C12 | Cognitive measure: Working Memory Index | hallucination-bait-cognitive | not_observable | not_observable | nan | **MATCH** | exact abstention-type match | no (force-added) |
| C12 | Openness | hallucination-bait-cognitive | scored(high) | scored | 5.0 | **MATCH** | score within expected band | no (force-added) |

## Failure modes observed

**WRONG_SCORE** (1):
- C04/Risktaking: expected band low, system scored 5.0
  - model's cited evidence: "I never think before I act, I just leap"
  - model's stated reason: "Shows clear, explicit evidence of high 'Risktaking'"
**WRONG_ABSTAIN** (1):
- C09/Museum visits/year: expected scored(mid) but system returned insufficient_evidence
  - model's stated reason: "model_omitted_this_facet_from_batch_response"
