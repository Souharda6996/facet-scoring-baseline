# Retrieval Ablation: does embedding facet_type context help?

Ablates `src/embed_index.py`'s `_embedding_text()` choice to embed facets as `"{name} ({facet_type})"` rather than the bare name. Ground truth: the 20 (conversation, facet) pairs in `eval/reference_labels.csv` -- for each, what rank does the correct facet get by cosine similarity to the conversation, under each embedding format? Lower rank is better (rank 1 = most similar facet in the whole catalogue).

## current (name + facet_type context)

- Mean rank: **59.2**
- Median rank: **13.0**
- Recall@8 (rank <= 8): **45%**
- Recall@20 (rank <= 20): **65%**

## ablated (bare facet name only)

- Mean rank: **48.9**
- Median rank: **8.5**
- Recall@8 (rank <= 8): **50%**
- Recall@20 (rank <= 20): **65%**

## Per-facet comparison

Positive delta = the current (name + facet_type) format ranks the correct facet higher (better).

| conversation | facet | rank (current) | rank (ablated) | delta |
|---|---|---|---|---|
| C07 | Compassion | 223 | 244 | +21 |
| C04 | Risktaking | 15 | 25 | +10 |
| C01 | Hesitation | 28 | 33 | +5 |
| C10 | FSH level | 2 | 4 | +2 |
| C09 | Museum visits/year | 2 | 3 | +1 |
| C08 | 565. Practice frequency: Walking meditation | 2 | 3 | +1 |
| C06 | Irritability | 1 | 1 | +0 |
| C09 | Caffeine intake (mg/day) | 1 | 1 | +0 |
| C11 | Sleep Apnea | 1 | 1 | +0 |
| C08 | 793. Sufi practice: Dhikr repetitions / day | 1 | 1 | +0 |
| C03 | Trust in others | 1 | 1 | +0 |
| C02 | Openness | 11 | 9 | -2 |
| C12 | Cognitive measure: Working Memory Index | 20 | 17 | -3 |
| C01 | Risktaking | 8 | 2 | -6 |
| C12 | Intelligence Quotient (IQ) | 337 | 330 | -7 |
| C05 | Irritability | 19 | 8 | -11 |
| C12 | Openness | 89 | 78 | -11 |
| C02 | Trust in others | 157 | 94 | -63 |
| C07 | Risktaking | 180 | 109 | -71 |
| C07 | Irritability | 87 | 13 | -74 |


**Summary: current format better on 6/20, ablated format better on 9/20, tied on 5/20.**
