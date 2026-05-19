# Benchmarking LLMs for Ordinal Uncertainty Grading in Chest Radiograph Reports

**A REFINE-Compliant Multi-Reader Validation Study**

> Authors [details withheld for double-blind peer review] — *Diagnostic and Interventional Radiology*, 2026 (manuscript in preparation)

---

## Overview

Complete, reproducible analysis pipeline for a REFINE-compliant validation study benchmarking
large language models (LLMs) for **ordinal grading** (0–3 scale) of diagnostic uncertainty
(hedging language) in chest radiograph reports against a three-reader radiologist consensus.

All results reported in this README have been **independently verified against real pipeline data**.

---

## Key Results

### LLM Performance vs. Multi-Reader Consensus

| Method | κ | 95% CI | Agreement Level |
|---|---|---|---|
| Gemini 2.5 Pro | 0.814 | 0.739–0.872 | Substantial-to-almost-perfect |
| Claude Opus 4.5 | 0.807 | 0.726–0.869 | Substantial-to-almost-perfect |
| Lexicon baseline | 0.622 | 0.516–0.713 | Substantial |
| Inter-model | 0.857 | 0.801–0.901 | Almost perfect |

> **Note:** Agreement level labelled as "substantial-to-almost-perfect" because point estimates
> exceed κ=0.80 but lower CI bounds (0.726, 0.739) fall within the substantial range.
> See Landis & Koch (1977) for interpretation criteria.

### Inter-Rater Reliability

| Comparison | Cohen's κ | 95% CI | Gwet's AC1 |
|---|---|---|---|
| Resident1 vs Attending | 0.659 | 0.516–0.776 | — |
| Resident2 vs Attending | 0.497 | 0.342–0.633 | — |
| Resident1 vs Resident2 | 0.421 | 0.252–0.569 | 0.649 (0.531–0.766) |
| 3-rater overall (Fleiss' κ) | 0.273 | 0.186–0.364 | 0.694 (0.605–0.783) |
| Attending vs Consensus | 0.902 | 0.827–0.961 | 0.932 (0.880–0.984) |

> **Gwet's AC1 interpretation:**
> - Pairwise Resident1 vs Resident2 AC1 = **0.649** (substantial)
> - 3-rater overall AC1 = **0.694** (substantial) — this is the prevalence-adjusted
>   estimate for the full annotation cohort.
> - AC1 > κ confirms the **prevalence paradox**: low Fleiss' κ (0.273) is driven
>   by the 40% "none" class, not by true observer discordance.

### Additional Metrics

| Metric | Value |
|---|---|
| Phrase-level hallucination (Claude) | 0/184 phrases (0%) — 70 reports |
| Phrase-level hallucination (Gemini) | 0/134 phrases fabricated (0%); 3 partial (XXXX redaction) |
| Output stochasticity (Claude, 3×30) | 29/30 = 96.7% identical |
| Prompt sensitivity (κ, 3 variants) | Original vs Short: 0.899; Original vs CoT: 0.881; Short vs CoT: 0.818 |

---

## Dataset

**OpenI Indiana University Chest X-ray Collection**
- Source: <https://openi.nlm.nih.gov/>
- 3,955 raw reports → 28 excluded (missing Findings/Impression) → **3,927 final**
- Validation subset: 100 reports (stratified, 25 per hedge stratum: 0, 1, 2, ≥3)
- Discovery cohort: 3,827 reports (descriptive domain analysis)

---

## Repository Structure

```
├── 01_data_preparation.py      # OpenI download + lexicon scoring
├── 02_sampling.py              # Stratified random sampling (n=100)
├── 03_claude_api.py            # Claude Opus 4.5 API inference
├── 04_gemini_api.py            # Gemini 2.5 Pro API (JSON mode)
├── 05_consensus.py             # Majority vote + expert adjudication
├── 06_statistics.py            # κ, Fleiss, Gwet's AC1 + bootstrap CI
│                               #   + paired bootstrap κ-difference test
├── 07_hallucination_check.py   # Phrase-level verbatim matching (Claude + Gemini)
├── 08_clinical_domain_analysis.py  # Fisher's exact + Benjamini-Hochberg FDR
├── 09_stochasticity_test.py    # 3-run × 30-report output consistency
├── 10_prompt_sensitivity.py    # 3-variant × 50-report prompt robustness
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/[username]/llm-radiology-uncertainty
cd llm-radiology-uncertainty
pip install -r requirements.txt
```

---

## Usage

Run scripts in order:

```bash
# Step 1: Download OpenI data and compute lexicon scores
python 01_data_preparation.py

# Step 2: Create stratified validation subset (seed=42, 25 per stratum)
python 02_sampling.py

# Step 3: Run LLM inference (API keys required)
export ANTHROPIC_API_KEY="your-claude-key"
export GEMINI_API_KEY="your-gemini-key"
python 03_claude_api.py
python 04_gemini_api.py

# Step 4: Compute consensus from annotation file
python 05_consensus.py

# Step 5: Full statistical analysis (κ, Fleiss, Gwet's AC1, kappa-difference test)
python 06_statistics.py

# Step 6: Phrase-level hallucination check (Claude + Gemini)
python 07_hallucination_check.py

# Step 7: Clinical domain analysis (Fisher + FDR, discovery cohort)
python 08_clinical_domain_analysis.py

# Step 8: Stochasticity test (pre-computed data or --live API)
python 09_stochasticity_test.py           # pre-computed
python 09_stochasticity_test.py --live    # live API (ANTHROPIC_API_KEY required)

# Step 9: Prompt sensitivity analysis (pre-computed data or --live API)
python 10_prompt_sensitivity.py           # pre-computed
python 10_prompt_sensitivity.py --live    # live API
```

---

## Models

| Model | API Identifier | Provider | Inference Period |
|---|---|---|---|
| Claude Opus 4.5 | `claude-opus-4-5-20251101` | Anthropic | April 2026 |
| Gemini 2.5 Pro | `gemini-2.5-pro-preview-05-06` | Google AI Studio | April 2026 |

**Total API cost:** USD 27.17 (Claude) + TRY 60.04 ≈ USD 1.80 (Gemini)

---

## Required Data Files

Required input files (available upon reasonable request (contact information will be provided upon publication)):

| File | Description |
|---|---|
| `Master_Etiketleme_3Annotator_sonuç.xlsx` | 3-reader annotation scores (Reader1_Score, Reader2_Score, Reader3_Score, Konsensus) |
| `triple_validation_100.xlsx` | Claude + Gemini + lexicon scores (n=100) |
| `gemini_validation_100.xlsx` | Gemini phrases and reasoning (n=100) |
| `FINAL_3927_reports_with_claude.xlsx` | Discovery cohort with Claude scores (n=3,927) |
| `full_claude_results_FINAL.json` | Claude raw JSON output (3,927 reports) |
| `gemini_results_FINAL.json` | Gemini raw JSON output (100 reports) |
| `stochasticity_test_30reports.xlsx` | Pre-computed 3-run stochasticity data |
| `prompt_sensitivity_50reports.xlsx` | Pre-computed 3-variant prompt sensitivity data |

---

## Verbatim Prompt (Supplementary 2)

Both LLMs received identical zero-shot, single-turn prompts at temperature=0.
This is the `v3_final` (original) prompt:

```
You are a radiologist analyzing the linguistic certainty in a chest X-ray report.

Report:
{report_text}

Analyze the level of diagnostic uncertainty expressed in this report. Consider:
- Hedging language (e.g., "may represent", "possibly", "suspicious for")
- But do NOT count negations like "no evidence of pneumothorax" as uncertainty
- Consider context: "consistent with X" is more certain than "may represent X"

Respond ONLY with valid JSON in this exact format:
{
  "uncertainty_level": "none|low|moderate|high",
  "uncertainty_score": 0,
  "key_phrases": ["list of actual hedging phrases found"],
  "reasoning": "brief 1-sentence explanation"
}
```

Prompt iteration history:
- `v1`: Broad uncertainty query ("Rate the diagnostic uncertainty in this report 0–3")
- `v2`: Negation exclusion rule added
- `v3_final` (above): Semantic context weighting + JSON schema added

---

## Methodological Notes

| Revision | Rationale |
|---|---|
| **Gwet's AC1 (pairwise + 3-rater)** | Reported separately: pairwise Resident1 vs Resident2 AC1=0.649; 3-rater overall AC1=0.694. Both exceed Fleiss' κ (0.273), confirming prevalence paradox. |
| **Fisher's exact + Benjamini-Hochberg FDR** | Chi-square assumption violated (n<5 in Other/Technical domain, n=6). Bonferroni replaced by FDR (less conservative, more appropriate for 6 comparisons). |
| **6 clinical domains** | 30 MeSH categories → 6 radiologically meaningful domains. n≥5 threshold applied as prespecified quality filter. |
| **Paired bootstrap κ-difference test** | LLM superiority over lexicon tested via 1,000-iteration paired bootstrap (95% CI of Δκ excludes 0 for both models). |
| **Post-hoc power removed** | Circular reasoning (Hoenig & Heisey, 2001, *Am Stat*). Sample size justified by CI half-width target (≤0.08 κ units). |
| **Gemini JSON mode** | `response_mime_type="application/json"` (native JSON mode) used to ensure structured output. |
| **XXXX redaction** | OpenI dataset anonymizes identifiers with "XXXX". Gemini hallucination assessment accounts for this: 3 partial matches are XXXX-artifact, not fabrications. |

---

## Citation

```bibtex
@article{anon2026llm,
  title   = {Benchmarking Large Language Models for Ordinal Uncertainty Grading
             in Chest Radiograph Reports: A REFINE-Compliant
             Multi-Reader Validation Study},
  author  = {[Author details withheld for double-blind peer review]},
  journal = {Diagnostic and Interventional Radiology},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## REFINE Compliance

This study adheres to the REFINE reporting guideline (44-item checklist):

> Mese I, Akinci D'Antonoli T, Bluethgen C, et al. Reporting checklist for
> foundation and large language models in medical research (REFINE):
> an international consensus guideline. *Diagn Interv Radiol.* 2026.
> doi:10.4274/dir.2026.263812

Full REFINE checklist available as Supplementary 1 of the manuscript.

**Self-assessed compliance:** 24 Full / 11 Partial-No / 9 N/A
(Honest self-assessment; items marked No include 4.4 probable training-data
contamination, 4.10 dataset separation unconfirmed, 6.2 no clinical workflow integration.)

---

## License

MIT License.
The OpenI dataset is subject to its own terms: <https://openi.nlm.nih.gov/>

---

## Acknowledgements

The language of this repository documentation was refined with the assistance of
a large language model (Claude; Anthropic). All scientific content, study design,
data analysis, statistical interpretation, and clinical conclusions were performed
exclusively by the authors.
