"""
03_claude_api.py
================
Claude Opus 4.5 inference for ordinal linguistic uncertainty grading.

Path-2 revision note:
This script supports the REFINE-aligned feasibility/methodology manuscript. The
primary output is uncertainty_level mapped to the 0-3 ordinal scale. The 0-10
uncertainty_score field is retained only as a secondary descriptive output.

Default use analyzes the 100-report validation cohort. Discovery-cohort inference
can be run by passing --input discovery_set_3827.xlsx --output claude_discovery_3827.json.

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python 03_claude_api.py
    python 03_claude_api.py --input discovery_set_3827.xlsx --output claude_discovery_3827.json
"""

import argparse
import getpass
import json
import os
import re
import time
from pathlib import Path

import anthropic
import pandas as pd

MODEL_ID = "claude-opus-4-5-20251101"
TEMPERATURE = 0
MAX_TOKENS = 512
MAX_RETRIES = 5
RETRY_DELAY = 2
LEVEL_MAP = {"none": 0, "low": 1, "moderate": 2, "high": 3}


def build_prompt(report_text: str) -> str:
    return f"""You are a radiologist analyzing the linguistic certainty in a chest X-ray report.

Report:
{report_text}

Analyze the level of diagnostic uncertainty expressed in this report. Consider:
- Hedging language (e.g., "may represent", "possibly", "suspicious for")
- But do NOT count negations like "no evidence of pneumothorax" as uncertainty
- Consider context: "consistent with X" is more certain than "may represent X"

Primary categorical output:
- uncertainty_level must be one of: none, low, moderate, high.
- The primary analysis maps uncertainty_level to the 0-3 ordinal scale: none=0, low=1, moderate=2, high=3.

Secondary descriptive output:
- uncertainty_score should be an integer from 0 to 10.
- uncertainty_score is collected only as a secondary descriptive output and is not used in the primary 0-3 analysis.

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


def analyze_report(client: anthropic.Anthropic, report_text: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": build_prompt(report_text)}],
            )
            raw = response.content[0].text
            result = parse_json_robust(raw)
            level = str(result.get("uncertainty_level", "none")).lower().strip()
            result["claude_score"] = LEVEL_MAP.get(level, 0)
            result["model_id"] = MODEL_ID
            return result
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return {
                    "uncertainty_level": "none",
                    "uncertainty_score": 0,
                    "key_phrases": [],
                    "reasoning": f"Parse/API error: {exc}",
                    "claude_score": 0,
                    "model_id": MODEL_ID,
                }


def combined_report_text(row: pd.Series) -> str:
    findings = row.get("findings", "")
    impression = row.get("impression", "")
    return f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="validation_set_100.xlsx", help="Input Excel with id/findings/impression columns")
    parser.add_argument("--output", default="claude_validation_100.json", help="Output JSON path")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or getpass.getpass("Anthropic API key: ")
    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_excel(args.input)
    output_path = Path(args.output)
    results = {}
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Checkpoint loaded: {len(results)} reports")

    for i, row in df.iterrows():
        report_id = str(row["id"])
        if report_id in results:
            continue
        result = analyze_report(client, combined_report_text(row))
        results[report_id] = result
        print(f"[{i+1:4d}/{len(df)}] ID={report_id} score={result['claude_score']} level={result['uncertainty_level']}")
        if (i + 1) % 10 == 0:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.5)

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
