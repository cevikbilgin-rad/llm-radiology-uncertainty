# Version Notes: Path-2 Anonymized Reproducibility Repository

This repository version was revised for the Path-2 feasibility/methodology resubmission of `DIR-2026-192`.

## Major Path-2 changes

1. Reframed the repository from a model benchmark to a REFINE-aligned feasibility/methodology workflow.
2. Added an explicit OpenI contamination notice and upper-bound interpretation.
3. Removed hybrid agreement labels from manuscript-facing outputs.
4. Applied conservative Landis-Koch labels using the lower bound of the 95% CI for model-to-consensus and lexicon-to-consensus rows.
5. Clarified that `uncertainty_level` is the primary 0-3 analysis variable and `uncertainty_score` is a secondary descriptive 0-10 output.
6. Rewrote clinical-domain analysis as exploratory descriptive analysis only, with no inferential testing or significance claims.
7. Reframed fabricated-evidence checks as verbatim phrase-level checks, not proof of semantic safety.
8. Included reviewer-requested sensitivity analyses for the dual-role PGY-2/operator reader and complete-disagreement cases.
9. Added an interpretation note that the PGY-2/operator-excluded senior-attending reference is equivalent by construction to attending-reader agreement because the attending was the tie-breaker.
10. Added LLM-vs-lexicon comparison output to support the contamination/memorization discussion.

## Double-blind/anonymization changes

1. Removed support for working-file column aliases containing author names.
2. Removed local Turkish working-file names from the expected public inputs.
3. Converted comments, docstrings, and terminal messages to English.
4. Added neutral reader labels (`reader1`, `reader2`, `reader3`) throughout.
5. Added a reviewer-facing `analysis_ready_validation_100.csv` file containing only public report IDs and derived ordinal scores.

## Reproducibility changes

1. Added `analysis_ready_validation_100.csv` so reviewers can reproduce numerical outputs without raw report text or API keys.
2. Added `memorization_probe_manual_review.csv` with derived manual-review flags used for the lexicon-error discussion.
3. Regenerated the `outputs/` folder from the updated `06_statistics.py` script.
4. Added `outputs/llm_lexicon_probe_summary.csv` and `outputs/sensitivity_interpretation_note.txt`.
