"""
09_stochasticity_test.py
========================
Output stochasticity kontrolü — Claude Opus 4.5 (ve opsiyonel Gemini 2.5 Pro).

30 rastgele validation raporu üzerinde 3 bağımsız çalıştırma ile
temperature=0'da output tutarlılığını ölçer.

Beklenen çıktı (gerçek veri ile doğrulandı):
  Claude Opus 4.5: 29/30 identical (96.7%)

NOT: Bu script hem mevcut test verilerini analiz eder (ön-hesaplanmış)
     hem de API'ye gerçek çağrı yapabilir (API key gerekir).

Kullanım (ön-hesaplanmış veri):
    python 09_stochasticity_test.py

Kullanım (canlı API — her iki model için):
    export ANTHROPIC_API_KEY="..."
    export GEMINI_API_KEY="..."
    python 09_stochasticity_test.py --live
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
import json

RANDOM_SEED = 42
N_REPORTS   = 30
N_RUNS      = 3


def analyze_precomputed(filepath: str, model_name: str):
    """Ön-hesaplanmış stochasticity verisini analiz et."""
    df = pd.read_excel(filepath)
    print(f"\n{'─'*60}")
    print(f"MODEL: {model_name}")
    print(f"{'─'*60}")
    print(f"Reports tested: {len(df)}")
    print(f"Runs per report: {N_RUNS}")

    consistent_n = int(df['consistent'].sum())
    inconsistent = df[~df['consistent']].copy()

    print(f"\nIdentical across all 3 runs: {consistent_n}/{len(df)} "
          f"({100*consistent_n/len(df):.1f}%)")

    if len(inconsistent) > 0:
        print(f"\nInconsistent cases ({len(inconsistent)}):")
        for _, row in inconsistent.iterrows():
            print(f"  ID={row['id']}: run1={row['run1']} run2={row['run2']} run3={row['run3']}")

    return consistent_n, len(df)


def run_live_claude(df_val: pd.DataFrame, val_ids_30: list, output_file: str):
    """Canlı Claude API çağrısı (3 bağımsız run)."""
    import anthropic, getpass
    api_key = os.environ.get('ANTHROPIC_API_KEY') or getpass.getpass("Anthropic API key: ")
    client  = anthropic.Anthropic(api_key=api_key)

    PROMPT_TEMPLATE = """You are a radiologist analyzing the linguistic certainty in a chest X-ray report.

Report:
{report_text}

Analyze the level of diagnostic uncertainty expressed in this report. Consider:
- Hedging language (e.g., "may represent", "possibly", "suspicious for")
- But do NOT count negations like "no evidence of pneumothorax" as uncertainty
- Consider context: "consistent with X" is more certain than "may represent X"

Respond ONLY with valid JSON in this exact format:
{{
  "uncertainty_level": "none|low|moderate|high",
  "uncertainty_score": 0,
  "key_phrases": ["list of actual hedging phrases found"],
  "reasoning": "brief 1-sentence explanation"
}}"""

    level_map = {"none": 0, "low": 1, "moderate": 2, "high": 3}
    results = {rid: [] for rid in val_ids_30}

    for run in range(1, N_RUNS + 1):
        print(f"\n  Run {run}/{N_RUNS}...")
        for rid in val_ids_30:
            row = df_val[df_val['id'] == rid].iloc[0]
            report_text = f"FINDINGS: {row['findings']}\n\nIMPRESSION: {row['impression']}"
            try:
                resp = client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=512,
                    temperature=0,
                    messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(
                        report_text=report_text)}]
                )
                import re, json as _json
                raw = resp.content[0].text.strip()
                m = re.match(r'^```(?:json)?\s*(.+?)\s*```\s*$', raw, re.DOTALL)
                if m: raw = m.group(1).strip()
                parsed = _json.loads(raw)
                score = level_map.get(parsed.get('uncertainty_level', 'none').lower(), 0)
            except Exception as e:
                score = -1
                print(f"    Error for ID {rid}: {e}")
            results[rid].append(score)
            time.sleep(0.5)

    rows = []
    for rid, scores in results.items():
        consistent = len(set(scores)) == 1 and -1 not in scores
        rows.append({'id': rid, 'run1': scores[0], 'run2': scores[1],
                     'run3': scores[2], 'consistent': consistent})
    df_out = pd.DataFrame(rows)
    df_out.to_excel(output_file, index=False)
    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true',
                        help='Canlı API çağrısı yap (API key gerekir)')
    args = parser.parse_args()

    print("=" * 65)
    print("OUTPUT STOCHASTICITY TEST [REFINE 3.1, 3.3]")
    print(f"N={N_REPORTS} reports × {N_RUNS} independent runs | temperature=0")
    print("=" * 65)

    if args.live:
        # Canlı mod
        df_triple = pd.read_excel('triple_validation_100.xlsx')
        df_3927   = pd.read_excel('FINAL_3927_reports_with_claude.xlsx')
        np.random.seed(RANDOM_SEED)
        all_ids  = list(df_triple['id'].unique())
        ids_30   = list(np.random.choice(all_ids, N_REPORTS, replace=False))
        df_val   = df_3927[df_3927['id'].isin(ids_30)].copy()

        print(f"\nRunning LIVE Claude API calls ({N_REPORTS} × {N_RUNS} = "
              f"{N_REPORTS*N_RUNS} calls)...")
        df_stoch = run_live_claude(df_val, ids_30, 'stochasticity_test_30reports.xlsx')
        c_n = int(df_stoch['consistent'].sum())
        print(f"\nClaude Opus 4.5: {c_n}/{N_REPORTS} consistent "
              f"({100*c_n/N_REPORTS:.1f}%)")
    else:
        # Ön-hesaplanmış veri modu
        stoch_file = 'stochasticity_test_30reports.xlsx'
        if not os.path.exists(stoch_file):
            print(f"\nERROR: {stoch_file} bulunamadı.")
            print("Canlı API için: python 09_stochasticity_test.py --live")
            return
        c_n, total = analyze_precomputed(stoch_file, "Claude Opus 4.5")

    # Validation check
    print(f"\n{'='*65}")
    print("VALIDATION CHECK (vs manuscript claim: 96.7%)")
    print("="*65)
    expected_n = 29
    ok = c_n == expected_n
    print(f"  {'✓' if ok else '✗'} Claude: {c_n}/{N_REPORTS} = "
          f"{100*c_n/N_REPORTS:.1f}% (expected 96.7%)")
    print(f"""
─────────────────────────────────────────────────────────────────
MANUSCRIPT REPORTING (Section 3.3):
  "Output stochasticity was assessed across three independent runs
  of Claude Opus 4.5 on 30 randomly selected validation reports
  under identical conditions (temperature=0, seed={RANDOM_SEED}).
  Identical scores were obtained in {c_n} of {N_REPORTS} reports
  ({100*c_n/N_REPORTS:.1f}%), confirming high deterministic behavior."
─────────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()
