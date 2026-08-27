# Retrieval Ablation: embedding format and model choice

Three configurations tested against the same 20 ground-truth (conversation, facet) pairs in `eval/reference_labels.csv`. Lower rank is better (rank 1 = most similar facet in the whole 369-facet catalogue). This script only measures retrieval quality -- it does not change the production index; see the module docstring for why.

## current (shipped) -- MiniLM, name + facet_type

- Mean rank: **59.2**
- Median rank: **13.0**
- Recall@8 (rank <= 8): **45%**
- Recall@20 (rank <= 20): **65%**

## ablated -- MiniLM, bare name

- Mean rank: **48.9**
- Median rank: **8.5**
- Recall@8 (rank <= 8): **50%**
- Recall@20 (rank <= 20): **65%**

## stronger model -- BGE-small-en-v1.5, bare name

- Mean rank: **35.5**
- Median rank: **13.5**
- Recall@8 (rank <= 8): **45%**
- Recall@20 (rank <= 20): **60%**

## Per-facet comparison

| conversation | facet | rank (current) | rank (ablated) | rank (BGE) |
|---|---|---|---|---|
| C06 | Irritability | 1 | 1 | 1 |
| C03 | Trust in others | 1 | 1 | 1 |
| C09 | Caffeine intake (mg/day) | 1 | 1 | 1 |
| C08 | 793. Sufi practice: Dhikr repetitions / day | 1 | 1 | 1 |
| C11 | Sleep Apnea | 1 | 1 | 1 |
| C09 | Museum visits/year | 2 | 3 | 20 |
| C10 | FSH level | 2 | 4 | 2 |
| C08 | 565. Practice frequency: Walking meditation | 2 | 3 | 3 |
| C01 | Risktaking | 8 | 2 | 10 |
| C02 | Openness | 11 | 9 | 21 |
| C04 | Risktaking | 15 | 25 | 41 |
| C05 | Irritability | 19 | 8 | 2 |
| C12 | Cognitive measure: Working Memory Index | 20 | 17 | 40 |
| C01 | Hesitation | 28 | 33 | 4 |
| C07 | Irritability | 87 | 13 | 17 |
| C12 | Openness | 89 | 78 | 55 |
| C02 | Trust in others | 157 | 94 | 54 |
| C07 | Risktaking | 180 | 109 | 86 |
| C07 | Compassion | 223 | 244 | 211 |
| C12 | Intelligence Quotient (IQ) | 337 | 330 | 139 |


**Summary -- a genuinely mixed result, reported as found, not smoothed over:**

BGE-small-en-v1.5 has by far the best **mean rank** (35.5 vs. 59.2 current / 48.9 ablated) -- it is much better at avoiding *catastrophically* bad ranks for hard cases (e.g. `Intelligence Quotient (IQ)`: rank 337/330 under MiniLM vs. rank 139 under BGE). But at the specific cutoffs this system actually operates at, it does **not** win: Recall@8 is 45% for BGE vs. 50% for the ablated MiniLM config (BGE ties the *current* shipped config's 45%, it does not beat it), and Recall@20 is actually slightly lower for BGE (60% vs. 65% ablated / 65% current).

**Practical read:** on this 20-pair reference set, at the top_k this system actually uses, **"ablated -- MiniLM, bare name" remains the best-supported choice of the three** -- switching to BGE would trade a large improvement in worst-case rank for no improvement (or a small regression) at the recall cutoff that actually matters for retrieval. This is exactly why the production index was not swapped on the strength of this ablation alone (see DECISIONS.md #5): a bigger reference set is needed before any of these differences can be trusted as more than sampling noise at n=20.
