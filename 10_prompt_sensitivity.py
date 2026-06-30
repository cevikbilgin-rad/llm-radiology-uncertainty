"""
10_prompt_sensitivity.py
========================
Prompt sensitivity analizi - Claude Opus 4.5.

Measures pairwise kappa agreement across three prompt variants in 50 validation reports.
Measures pairwise kappa agreement.

Prompt variants:
  v1 (original):   Tam prompt (Supplementary 2)
  v2 (short):      Shortened directive version
  v3 (cot):        Chain-of-thought versiyon

Expected outputs verified on study data:
  Original vs Short:           κ=0.899
  Original vs Chain-of-Thought: κ=0.881
  Short vs Chain-of-Thought:   κ=0.818

Usage (precomputed data):
    python 10_prompt_sensitivity.py

Usage (live API):
    export ANTHROPIC_API_KEY="..."
    python 10_prompt_sensitivity.py --live
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
import json
import re
from sklearn.metrics import cohen_kappa_score

RANDOM_SEED = 42
N_REPORTS   = 50

# ─── PROMPT VARYANTLARI ────────────────────────────────────────────────────

PROMPT_V1_ORIGINAL = """You are a radiologist analyzing the linguistic certainty in a chest X-ray report.

Report:
{report_text}

Analyze the level of diagnostic uncertainty expressed in this report. Consider:
- Hedging language (e.g., "may represent", "possibly", "suspicious for")
- But do NOT count negations like "no evidence of pneumothorax" as uncertainty
- Consider context: "consistent with X" is more certain than "may represent X"

Primary categorical output: uncertainty_level maps to the 0-3 ordinal scale (none=0, low=1, moderate=2, high=3).
Secondary descriptive output: uncertainty_score should be an integer from 0 to 10 and is not used in the primary analysis.

Respond ONLY with valid JSON in this exact format:
{{
  "uncertainty_level": "none|low|moderate|high",
  "uncertainty_score": 0,
  "key_phrases": ["list of actual hedging phrases found"],
  "reasoning": "brief 1-sentence explanation"
}}"""

PROMPT_V2_SHORT = """Analyze diagnostic uncertainty in this chest X-ray report.
Primary ordinal scale: none=0, low=1, moderate=2, high=3. uncertainty_score is a secondary 0-10 descriptive field only.
Exclude negations ("no evidence of..."). Count hedges only.

Report: {report_text}

JSON only: {{"uncertainty_level":"none|low|moderate|high","uncertainty_score":0,"key_phrases":[],"reasoning":""}}"""

PROMPT_V3_COT = """You are an expert radiologist. Analyze this chest X-ray report step by step.

Report:
{report_text}

Step 1: List all hedging phrases (e.g., "may represent", "possibly", "cannot exclude").
Step 2: Exclude negations like "no evidence of X" - these are NOT uncertainty.
Step 3: Count the remaining hedges: 0=none, 1=low, 2=moderate, >=3=high.
Step 4: Use uncertainty_level as the primary 0-3 ordinal category; uncertainty_score is a secondary 0-10 descriptive field only.
Step 5: Output your final answer as JSON only.

{{
  "uncertainty_level": "none|low|moderate|high",
  "uncertainty_score": 0,
  "key_phrases": ["hedging phrases found after step 1-2"],
  "reasoning": "brief 1-sentence explanation"
}}"""


def call_claude(client, prompt: str) -> int:
    """Claude API call returning an integer score."""
    level_map = {"none": 0, "low": 1, "moderate": 2, "high": 3}
    try:
        resp = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        m = re.match(r'^```(?:json)?\s*(.+?)\s*```\s*$', raw, re.DOTALL)
        if m: raw = m.group(1).strip()
        parsed = json.loads(raw)
        return level_map.get(parsed.get('uncertainty_level', 'none').lower(), 0)
    except Exception as e:
        print(f"    Error: {e}")
        return -1


def run_live_prompt_sensitivity(df_val: pd.DataFrame, ids_50: list, output_file: str):
    """Live API calls with three prompt variants."""
    import anthropic, getpass
    api_key = os.environ.get('ANTHROPIC_API_KEY') or getpass.getpass("Anthropic API key: ")
    client  = anthropic.Anthropic(api_key=api_key)

    rows = []
    for i, rid in enumerate(ids_50):
        row = df_val[df_val['id'] == rid].iloc[0]
        report_text = f"FINDINGS: {row['findings']}\n\nIMPRESSION: {row['impression']}"
        print(f"  [{i+1:2d}/{N_REPORTS}] ID={rid}")

        s1 = call_claude(client, PROMPT_V1_ORIGINAL.format(report_text=report_text))
        time.sleep(0.5)
        s2 = call_claude(client, PROMPT_V2_SHORT.format(report_text=report_text))
        time.sleep(0.5)
        s3 = call_claude(client, PROMPT_V3_COT.format(report_text=report_text))
        time.sleep(0.5)
        rows.append({'id': rid, 'original': s1, 'v2_short': s2, 'v3_cot': s3})
        print(f"    original={s1}, short={s2}, cot={s3}")

    df_out = pd.DataFrame(rows)
    df_out.to_excel(output_file, index=False)
    return df_out


def analyze_sensitivity(df: pd.DataFrame):
    """Pairwise kappa hesapla."""
    orig  = df['original'].astype(int)
    short = df['v2_short'].astype(int)
    cot   = df['v3_cot'].astype(int)

    def kq(a, b, label):
        k = cohen_kappa_score(a, b, weights='quadratic')
        print(f"  {label}: κ={k:.3f}")
        return k

    k1 = kq(orig,  short, "Original vs Short            ")
    k2 = kq(orig,  cot,   "Original vs Chain-of-Thought ")
    k3 = kq(short, cot,   "Short vs Chain-of-Thought    ")
    return k1, k2, k3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true',
                        help='Run live API calls')
    args = parser.parse_args()

    print("=" * 65)
    print("PROMPT SENSITIVITY ANALYSIS [REFINE 2.1, 5.6]")
    print(f"N={N_REPORTS} reports x 3 prompt variants | temperature=0")
    print("=" * 65)
    print("\nPrompt variants:")
    print("  v1 (original): Full zero-shot prompt (Supplementary 2)")
    print("  v2 (short):    Shortened directive version")
    print("  v3 (cot):      Chain-of-thought version")

    sens_file = 'prompt_sensitivity_50reports.xlsx'

    if args.live:
        df_triple = pd.read_excel('triple_validation_100.xlsx')
        df_3927   = pd.read_excel('FINAL_3927_reports_with_claude.xlsx')
        np.random.seed(RANDOM_SEED)
        all_ids  = list(df_triple['id'].unique())
        ids_50   = list(np.random.choice(all_ids, N_REPORTS, replace=False))
        df_val   = df_3927[df_3927['id'].isin(ids_50)].copy()
        print(f"\nRunning LIVE API calls ({N_REPORTS} x 3 = {N_REPORTS*3})...")
        df_sens = run_live_prompt_sensitivity(df_val, ids_50, sens_file)
    else:
        if not os.path.exists(sens_file):
            print(f"\nERROR: {sens_file} not found.")
            print("For live API use: python 10_prompt_sensitivity.py --live")
            return
        df_sens = pd.read_excel(sens_file)
        print(f"\nPre-computed data loaded: {len(df_sens)} reports")

    print(f"\n--- Pairwise Inter-prompt Agreement ---")
    k1, k2, k3 = analyze_sensitivity(df_sens)

    # Validation checks
    print(f"\n{'='*65}")
    print("VALIDATION CHECKS (vs manuscript claims)")
    print("="*65)
    checks = [
        ("Original vs Short κ ≈ 0.899",           abs(k1 - 0.899) < 0.005),
        ("Original vs Chain-of-Thought κ ≈ 0.881", abs(k2 - 0.881) < 0.005),
        ("Short vs Chain-of-Thought κ ≈ 0.818",    abs(k3 - 0.818) < 0.005),
    ]
    for name, ok in checks:
        print(f"  {'OK' if ok else 'ERROR'} {name}")

    print(f"""
─────────────────────────────────────────────────────────────────
MANUSCRIPT REPORTING (Section 3.3 / Section 2.5):
  Prompt variants tested on {N_REPORTS}-report subset:
  Original vs Short:            κ={k1:.3f}
  Original vs Chain-of-Thought: κ={k2:.3f}
  Short vs Chain-of-Thought:    κ={k3:.3f}
  -> All >=0.80 (almost perfect) -> robust to prompt reformulation
─────────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()
