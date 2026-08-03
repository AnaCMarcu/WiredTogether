# Annotation rubrics

Annotate each batch file `out/samples/<dim>/batch_NNN.jsonl` into `out/annotations/<dim>/batch_NNN.jsonl`: one JSON per line, schema `{"id": <item id>, ...rubric fields}`. Repass files are annotated the same way into `annotations/<dim>/repass.jsonl` in a SEPARATE session without looking at the first-pass labels.

## messages — MessageAnnotation
- `categories`: list from [inform, request, commit, acknowledge, question, noise] (multi-label)
- `coordination_function`: one of none|inform|request|commit|acknowledge — the PRIMARY function
- `is_grounded`: bool — consistent with the context window (actions/chamber), no invented objects
- `specificity`: 0 (vague) | 1 (some referents) | 2 (actionable, names object AND location/direction or addressee task)
- `notes`: short free text (optional)

## beliefs — BeliefAnnotation
- `hallucination`: bool — belief asserts something impossible/contradicted by the ground-truth line
- `stale`: bool — plausibly true earlier but not now
- `verifiable_claims`: int — claims checkable against ground truth
- `correct_claims`: int — of those, how many are correct
- `notes`: short free text (optional)

## social — SocialAnnotation
- `mentions_bond_values`: bool — reasoning cites bond numbers/directions
- `values_match_table`: yes|no|approx|absent — cited values vs referenced_bonds
- `decision_follows_bonds`: yes|no|partial — ask/respond choice consistent with bond ranking
- `explanation_quality`: 0 (generic) | 1 (references state) | 2 (bond-specific causal reasoning)
- `notes`: short free text (optional)

## failures — FailureAdjudication
- `is_failure`: bool — the window shows genuinely unproductive/erroneous behavior
- `matches_detector`: bool — the failure is of the flagged type (false for controls unless a real failure coincides)
- `severity`: 0 (cosmetic) | 1 (wastes steps) | 2 (blocks progression)
- `cause_hypothesis`: short free text
- `notes`: short free text (optional)
