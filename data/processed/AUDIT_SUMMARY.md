# Facet Audit Summary

Total raw rows: **399**

## Facet type distribution

- `personality_trait_or_disposition`: 260
- `spiritual_religious_practice`: 34
- `header_artifact`: 30
- `behavioral_frequency_count`: 26
- `cognitive_psychometric`: 22
- `medical_biological`: 11
- `clinical_psychological_scale`: 9
- `sensitive_biographical`: 7

## Malformed / header-artifact rows

Flagged malformed: **30** (7.5%)

- `header_like_trailing_colon`: 30

## Normalization transforms applied

- `stripped_leading_numeric_id`: 31 rows
- `stripped_trailing_colon`: 30 rows
- `split_camel_case`: 3 rows

## Duplicate normalized values

Rows whose normalized form duplicates an earlier row: **0**


## Conversation-observable vs not

- conversation_observable=True: **327** (82.0%)
- conversation_observable=False: **72** (18.0%)
- of the observable set, requires_explicit_disclosure=True (self-report only, not inferable from tone/behavior): **67**

## Abstention reasons (facets excluded from scoring by taxonomy gate)

- `not_a_scorable_facet_category_header`: 30
- `requires_formal_psychometric_testing`: 22
- `requires_medical_lab_or_genetic_evidence`: 11
- `requires_validated_clinical_instrument_or_diagnosis`: 9

## Sensitivity distribution

- `low`: 286
- `medium`: 56
- `n/a`: 30
- `high`: 27

## Sample of each facet_type

**personality_trait_or_disposition**: "Risktaking"; "Naivety"; "Acidity"; "Common-sense"
**header_artifact**: "Democratic Leadership:"; "HonestyHumility:"; "Relationship Building Themes:"; "Numerical Reasoning Subcomponents:"
**cognitive_psychometric**: "Statistical Reasoning"; "Comparing alphanumeric data"; "Spatial perception"; "Alphabetical filing skills"
**medical_biological**: "FSH level"; "Parathyroid-hormone level"; "Chromatin-accessibility score"; "Serotonin transporter availability"
**spiritual_religious_practice**: "Presence of Spiritual Pain"; "Role of Spirituality in Community Involvement"; "Pilgrimage participation count"; "800. Sufi practice: Sufi retreat attendance count"
**behavioral_frequency_count**: "Negative Affect Frequency"; "Feedback-giving frequency"; "Dance-cardio sessions"; "Subscription count"
**clinical_psychological_scale**: "Depression Symptoms"; "Depression: Feelings of sadness and hopelessness"; "Sleep-disorder diagnosis"; "Burnout Symptoms"
**sensitive_biographical**: "Data-sharing consent level"; "Physical-violence exposure"; "Sleep-environment temperature"; "Kink-interest diversity"
