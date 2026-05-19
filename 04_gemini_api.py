"""
04_gemini_api.py
================
Gemini 2.5 Pro ile 100 validation raporunu analiz et.
Model: gemini-2.5-pro (Google AI Studio, April 2026)

Yapılandırılmış JSON çıktısı için response_mime_type="application/json"
parametresi kullanılır.

Çıktı: gemini_results_FINAL.json

Gereksinimler:
    pip install google-genai pandas openpyxl

Kullanım:
    export GEMINI_API_KEY="your-key-here"
    python 04_gemini_api.py
"""

from google import genai
from google.genai import types
import pandas as pd
import json
import time
import os
import getpass
import re

MODEL_ID    = "gemini-2.5-pro-preview-05-06"
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
    raw = raw.strip()
    fence_match = re.match(r'^```(?:json)?\s*(.+?)\s*```\s*$', raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()
    return json.loads(raw)


def analyze_report(client, report_text: str) -> dict:
    """Gemini structured output (JSON mode)."""
    level_map = {"none": 0, "low": 1, "moderate": 2, "high": 3}

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=build_prompt(report_text),
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    response_mime_type="application/json"
                )
            )
            result = parse_json_robust(response.text)
            result['gemini_score'] = level_map.get(
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
                    'reasoning':         f'Error: {str(e)}',
                    'gemini_score':      0
                }


def main():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        api_key = getpass.getpass("Gemini API key: ")

    client = genai.Client(api_key=api_key)
    df = pd.read_excel('validation_set_100.xlsx')
    print(f"Toplam rapor: {len(df)}")

    results = {}
    if os.path.exists('gemini_results_FINAL.json'):
        with open('gemini_results_FINAL.json') as f:
            results = json.load(f)
        print(f"Checkpoint: {len(results)} rapor zaten analiz edilmiş")

    for i, row in df.iterrows():
        report_id = str(row['id'])
        if report_id in results:
            continue

        report_text = f"FINDINGS: {row['findings']}\n\nIMPRESSION: {row['impression']}"
        result = analyze_report(client, report_text)
        results[report_id] = result

        print(f"[{i+1:3d}/100] ID {report_id}: score={result['gemini_score']} | {result['uncertainty_level']}")

        if (i + 1) % 10 == 0:
            with open('gemini_results_FINAL.json', 'w') as f:
                json.dump(results, f, indent=2)

        time.sleep(1.0)

    with open('gemini_results_FINAL.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Tamamlandı: {len(results)} rapor → gemini_results_FINAL.json")


if __name__ == '__main__':
    main()
