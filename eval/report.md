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

## Row-by-row detail

| conv | facet | category | expected | system status | system score | outcome | detail |
|---|---|---|---|---|---|---|---|
| C01 | Risktaking | clear | scored(high) | scored | 5.0 | **MATCH** | score within expected band |
| C01 | Hesitation | clear | scored(low) | scored | 1.0 | **MATCH** | score within expected band |
| C02 | Openness | ambiguous | scored(mid) | scored | 3.0 | **MATCH** | score within expected band |
| C02 | Trust in others | ambiguous | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match |
| C03 | Trust in others | contradictory | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match |
| C04 | Risktaking | quoted | scored(low) | scored | 5.0 | **WRONG_SCORE** | expected band low, system scored 5.0 |
| C05 | Irritability | sarcastic | scored(high) | scored | 5.0 | **MATCH** | score within expected band |
| C06 | Irritability | code-switched | scored(high) | scored | 5.0 | **MATCH** | score within expected band |
| C07 | Risktaking | low-evidence | insufficient_evidence | not_observable | nan | **MATCH_LOOSE** | both abstained but via different status (not_observable vs insufficient_evidence) |
| C07 | Compassion | low-evidence | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match |
| C07 | Irritability | low-evidence | insufficient_evidence | insufficient_evidence | nan | **MATCH** | exact abstention-type match |
| C08 | 793. Sufi practice: Dhikr repetitions / day | clear | scored(high) | scored | 5.0 | **MATCH** | score within expected band |
| C08 | 565. Practice frequency: Walking meditation | clear | scored(mid-high) | scored | 5.0 | **MATCH** | score within expected band |
| C09 | Caffeine intake (mg/day) | clear | scored(mid-high) | scored | 5.0 | **MATCH** | score within expected band |
| C09 | Museum visits/year | clear | scored(mid) | insufficient_evidence | nan | **WRONG_ABSTAIN** | expected scored(mid) but system returned insufficient_evidence |
| C10 | FSH level | hallucination-bait-medical | not_observable | not_observable | nan | **MATCH** | exact abstention-type match |
| C11 | Sleep Apnea | hallucination-bait-diagnosis | not_observable | not_observable | nan | **MATCH** | exact abstention-type match |
| C12 | Intelligence Quotient (IQ) | hallucination-bait-cognitive | not_observable | not_observable | nan | **MATCH** | exact abstention-type match |
| C12 | Cognitive measure: Working Memory Index | hallucination-bait-cognitive | not_observable | not_observable | nan | **MATCH** | exact abstention-type match |
| C12 | Openness | hallucination-bait-cognitive | scored(high) | scored | 5.0 | **MATCH** | score within expected band |

## Failure modes observed

**WRONG_SCORE** (1):
- C04/Risktaking: expected band low, system scored 5.0
  - model's cited evidence: "I never think before I act, I just leap"
  - model's stated reason: "Shows clear, explicit evidence of high 'Risktaking'"
**WRONG_ABSTAIN** (1):
- C09/Museum visits/year: expected scored(mid) but system returned insufficient_evidence
  - model's stated reason: "model_omitted_this_facet_from_batch_response"
