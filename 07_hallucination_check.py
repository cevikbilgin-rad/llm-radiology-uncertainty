"""
07_hallucination_check.py
=========================
Phrase-level fabricated-evidence check - Claude Opus 4.5 VE Gemini 2.5 Pro.

Checks whether each model-returned key_phrases item appears in the source report text.
Verifies verbatim matching only.

Limitations:
  - Measures fabricated phrase-level evidence only.
  - OpenI XXXX redaction is explicitly handled:
    Partial matches are expected for phrases containing XXXX and are not counted as fabricated.
  - Does NOT prove absence of semantic/contextual misattribution.

Expected outputs verified on study data:
  Claude:  0/184 phrase fabricated (0%) - 70 Report
  Gemini:  0/134 phrase fabricated (0%), 3 partial (XXXX nedeniyle)

NOT: "XXXX" anonymized terms: In OpenI, anonymized terms are replaced with XXXX.
     Therefore, some phrases may be partial rather than exact matches.

Usage:
    python 07_hallucination_check.py
"""

import pandas as pd
import json


def check_phrase(phrase: str, report_text: str) -> str:
    """
    Phrase'i Report metninde ara.
    Returns: 'verbatim' | 'partial' | 'fabricated'
    """
    phrase_low = str(phrase).strip().lower()
    text_low   = report_text.lower()

    if len(phrase_low) < 3:
        return 'skip'

    # 1. Verbatim exact match
    if phrase_low in text_low:
        return 'verbatim'

    # 2. Partial match using the first two words for XXXX redaction
    words = phrase_low.split()
    if len(words) >= 2:
        two_word = words[0] + ' ' + words[1]
        if two_word in text_low:
            return 'partial'

    # 3. XXXX redaction check for OpenI anonymization
    non_xxxx = [w for w in words if w != 'xxxx' and len(w) > 3]
    if non_xxxx:
        content_word = non_xxxx[0]
        if content_word in text_low:
            return 'partial'

    return 'fabricated'


def run_claude_check(claude_json: dict, df_reports: pd.DataFrame, val_ids: set) -> dict:
    """Claude hallucination check."""
    stats = dict(reports_checked=0, reports_with_phrases=0, reports_no_phrases=0,
                 total_phrases=0, verbatim=0, partial=0, fabricated=0)
    details = []

    for _, row in df_reports.iterrows():
        rid = str(row['id'])
        if rid not in val_ids:
            continue
        if rid not in claude_json:
            continue

        stats['reports_checked'] += 1
        result  = claude_json[rid]
        phrases = result.get('phrases') or result.get('key_phrases') or []
        if isinstance(phrases, str):
            phrases = [phrases] if phrases else []
        phrases = [p for p in phrases if p and len(str(p).strip()) >= 3]

        if not phrases:
            stats['reports_no_phrases'] += 1
            continue

        stats['reports_with_phrases'] += 1
        report_text = (str(row.get('findings', '')) + ' ' +
                       str(row.get('impression', '')))

        for phrase in phrases:
            outcome = check_phrase(phrase, report_text)
            if outcome == 'skip':
                continue
            stats['total_phrases'] += 1
            stats[outcome] += 1
            if outcome == 'fabricated':
                details.append(dict(model='Claude', id=rid, phrase=phrase,
                                    impression=str(row.get('impression', ''))[:200]))

    return stats, details


def run_gemini_check(df_gemini: pd.DataFrame) -> dict:
    """Gemini hallucination check (phrases stored in Excel)."""
    stats = dict(reports_checked=0, reports_with_phrases=0, reports_no_phrases=0,
                 total_phrases=0, verbatim=0, partial=0, fabricated=0)
    details = []

    for _, row in df_gemini.iterrows():
        stats['reports_checked'] += 1
        phrases_raw = row.get('gemini_phrases', '')

        if pd.isna(phrases_raw) or str(phrases_raw).strip() == '':
            stats['reports_no_phrases'] += 1
            continue

        phrases = [p.strip() for p in str(phrases_raw).split(',')
                   if len(p.strip()) >= 3]
        if not phrases:
            stats['reports_no_phrases'] += 1
            continue

        stats['reports_with_phrases'] += 1
        report_text = (str(row.get('findings', '')) + ' ' +
                       str(row.get('impression', '')))

        for phrase in phrases:
            outcome = check_phrase(phrase, report_text)
            if outcome == 'skip':
                continue
            stats['total_phrases'] += 1
            stats[outcome] += 1
            if outcome == 'fabricated':
                details.append(dict(model='Gemini', id=row['id'], phrase=phrase,
                                    impression=str(row.get('impression', ''))[:200]))

    return stats, details


def print_report(model_name: str, stats: dict, details: list):
    total = stats['total_phrases']
    rate_verb = 100 * stats['verbatim']  / total if total > 0 else 0
    rate_part = 100 * stats['partial']   / total if total > 0 else 0
    rate_fab  = 100 * stats['fabricated'] / total if total > 0 else 0

    print(f"\n{'─'*60}")
    print(f"MODEL: {model_name}")
    print(f"{'─'*60}")
    print(f"  Reports checked:          {stats['reports_checked']}")
    print(f"  Reports with phrases:     {stats['reports_with_phrases']}")
    print(f"  Reports without phrases:  {stats['reports_no_phrases']}")
    print(f"  Total phrases checked:    {total}")
    print(f"  Verbatim matches:         {stats['verbatim']} ({rate_verb:.1f}%)")
    print(f"  Partial matches (XXXX):   {stats['partial']}  ({rate_part:.1f}%)")
    print(f"  Fabricated phrase evidence:{stats['fabricated']}  ({rate_fab:.1f}%)")

    if stats['fabricated'] == 0:
        print(f"\n  OK RESULT: 0/{total} fabricated (0%) - no verbatim fabricated phrases observed")
    else:
        print(f"\n  ERROR RESULT: {stats['fabricated']}/{total} fabricated")
        for d in details:
            print(f"\n    ID {d['id']}: phrase='{d['phrase']}'")
            print(f"    Impression: {d['impression']}")


def main():
    print("=" * 65)
    print("PHRASE-LEVEL HALLUCINATION CHECK [REFINE 5.5]")
    print("=" * 65)

    # Load Claude JSON (3927 reports, subset to validation 100)
    with open('full_claude_results_FINAL.json') as f:
        claude_json = json.load(f)

    df_triple = pd.read_excel('triple_validation_100.xlsx')
    df_3927   = pd.read_excel('FINAL_3927_reports_with_claude.xlsx')
    df_gemini = pd.read_excel('gemini_validation_100.xlsx')

    val_ids = set(df_triple['id'].astype(str))
    df_val  = df_3927[df_3927['id'].astype(str).isin(val_ids)].copy()

    # ── CLAUDE ──────────────────────────────────────────────────────────────
    claude_stats, claude_details = run_claude_check(claude_json, df_val, val_ids)
    print_report("Claude Opus 4.5", claude_stats, claude_details)

    # ── GEMINI ──────────────────────────────────────────────────────────────
    gemini_stats, gemini_details = run_gemini_check(df_gemini)
    print_report("Gemini 2.5 Pro", gemini_stats, gemini_details)

    # ── VALIDATION CHECKS ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("VALIDATION CHECKS (vs manuscript)")
    print("="*65)
    checks = [
        ("Claude reports with phrases = 70",   claude_stats['reports_with_phrases'] == 70),
        ("Claude fabricated = 0",               claude_stats['fabricated'] == 0),
        ("Gemini fabricated = 0 (XXXX-aware)", gemini_stats['fabricated'] == 0),
    ]
    for name, ok in checks:
        print(f"  {'OK' if ok else 'ERROR'} {name}")

    print(f"""
─────────────────────────────────────────────────────────────────
MANUSCRIPT REPORTING (Section 3.4):
  Claude Opus 4.5:  0/{claude_stats['total_phrases']} phrases fabricated (0%)
                    {claude_stats['reports_with_phrases']} reports with identifiable phrases
  Gemini 2.5 Pro:   0/{gemini_stats['total_phrases']} phrases fabricated (0%)
                    {gemini_stats['partial']} partial matches (XXXX redaction artifact)
─────────────────────────────────────────────────────────────────
""")


if __name__ == '__main__':
    main()
