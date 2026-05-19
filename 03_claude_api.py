"""
03_claude_api.py
================
Claude Opus 4.5 ile 100 validation raporunu analiz et.
Model: claude-opus-4-5-20251101 (Anthropic, April 2026)

Çıktı: full_claude_results_FINAL.json

Gereksinimler:
    pip install anthropic>=0.39.0 pandas openpyxl

Kullanım:
    export ANTHROPIC_API_KEY="your-key-here"
    python 03_claude_api.py
"""

import anthropic
import pandas as pd
import json
import time
import os
import getpass
import re

# ─── MODEL KONFİGÜRASYONU ──────────────────────────────────────────────────
MODEL_ID    = "claude-opus-4-5-20251101"
TEMPERATURE = 0
MAX_RETRIES = 5
RETRY_DELAY = 2


def build_prompt(report_text: str) -> str:
    return f"""You are a radiologist analyzing the linguistic certainty in a chest X-ray report.

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


def parse_json_robust(raw: str) -> dict:
    """Markdown fence varsa kaldır, JSON parse et."""
    raw = raw.strip()
    # Markdown fence kaldır
    fence_match = re.match(r'^```(?:json)?\s*(.+?)\s*```\s*$', raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()
    return json.loads(raw)


def analyze_report(client: anthropic.Anthropic, report_text: str) -> dict:
    """Tek raporu analiz et, retry ile."""
    level_map = {"none": 0, "low": 1, "moderate": 2, "high": 3}

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=512,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": build_prompt(report_text)}]
            )
            raw    = response.content[0].text
            result = parse_json_robust(raw)
            result['claude_score'] = level_map.get(
                str(result.get('uncertainty_level', 'none')).lower(), 0
            )
            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {
                    'uncertainty_level': 'none',
                    'uncertainty_score': 0,
                    'key_phrases':       [],
                    'reasoning':         f'Parse error: {str(e)}',
                    'claude_score':      0
                }


def main():
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        api_key = getpass.getpass("Anthropic API key: ")

    client = anthropic.Anthropic(api_key=api_key)
    df = pd.read_excel('validation_set_100.xlsx')
    print(f"Toplam rapor: {len(df)}")

    # Checkpoint
    results = {}
    if os.path.exists('full_claude_results_FINAL.json'):
        with open('full_claude_results_FINAL.json') as f:
            results = json.load(f)
        print(f"Checkpoint: {len(results)} rapor zaten analiz edilmiş")

    for i, row in df.iterrows():
        report_id = str(row['id'])
        if report_id in results:
            continue

        report_text = f"FINDINGS: {row['findings']}\n\nIMPRESSION: {row['impression']}"
        result = analyze_report(client, report_text)
        results[report_id] = result

        print(f"[{i+1:3d}/100] ID {report_id}: score={result['claude_score']} | {result['uncertainty_level']}")

        if (i + 1) % 10 == 0:
            with open('full_claude_results_FINAL.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  → Checkpoint ({len(results)} rapor)")

        time.sleep(0.5)

    with open('full_claude_results_FINAL.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Tamamlandı: {len(results)} rapor → full_claude_results_FINAL.json")


if __name__ == '__main__':
    main()
