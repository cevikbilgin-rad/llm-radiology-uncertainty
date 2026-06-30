"""
05_consensus.py
===============
Construct manual reference standards from three-reader annotations.

Primary reference standard:
  - Majority vote (>=2 of 3 readers)
  - Complete three-way disagreement -> attending radiologist decision

Sensitivity reference standard requested during review:
  - PGY-2/operator reader excluded
  - Senior resident + attending radiologist reference standard
  - Attending radiologist breaks senior-attending ties

Double-blind note:
This public/anonymized version uses neutral column names only. Working-file
column names containing author identifiers are intentionally not supported.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

CATEGORIES = [0, 1, 2, 3]
ANNOTATION_CANDIDATES = [
    "annotations_3readers.xlsx",
    "annotations_3readers.csv",
    "manual_annotations_3readers.xlsx",
    "manual_annotations_3readers.csv",
]
SHEET_CANDIDATES = ["Annotations", "Sheet1", 0]


NEUTRAL_COLUMN_ALIASES = {
    "id": "id",
    "report_id": "id",
    "Report_ID": "id",
    "Reader1_Score": "reader1",
    "reader1": "reader1",
    "Reader2_Score": "reader2",
    "reader2": "reader2",
    "Reader3_Score": "reader3",
    "reader3": "reader3",
    "Consensus": "consensus_raw",
    "consensus": "consensus_raw",
}


def find_file(candidates=ANNOTATION_CANDIDATES) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Annotation file not found. Expected one of: " + ", ".join(candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    last_error = None
    for sheet in SHEET_CANDIDATES:
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except Exception as exc:  # pragma: no cover - helpful for reviewer-side files
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {src: dst for src, dst in NEUTRAL_COLUMN_ALIASES.items() if src in df.columns}
    out = df.rename(columns=column_map).copy()
    required = ["id", "reader1", "reader2", "reader3"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required neutral annotation columns: {missing}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(int)
    if "consensus_raw" not in out.columns:
        out["consensus_raw"] = ""
    return out


def majority_or_attending(row: pd.Series) -> int:
    scores = [int(row["reader1"]), int(row["reader2"]), int(row["reader3"])]
    for score in CATEGORIES:
        if scores.count(score) >= 2:
            return score
    return int(row["reader3"])


def complete_disagreement(row: pd.Series) -> bool:
    return len({int(row["reader1"]), int(row["reader2"]), int(row["reader3"])}) == 3


def two_reader_senior_attending(row: pd.Series) -> int:
    """
    Editor-requested PGY-2/operator-excluded sensitivity reference.

    With an ordinal two-reader design, any senior-attending disagreement is a tie;
    the attending radiologist was the prespecified tie-breaker. Therefore this
    reference is, by construction, equivalent to the attending reader's score.
    This is intentional and is reported transparently in the manuscript.
    """
    if int(row["reader2"]) == int(row["reader3"]):
        return int(row["reader2"])
    return int(row["reader3"])


def main() -> None:
    annotation_file = find_file()
    df_raw = read_table(annotation_file)
    df = normalize_columns(df_raw)

    df["consensus_3reader"] = df.apply(majority_or_attending, axis=1)
    df["complete_disagreement"] = df.apply(complete_disagreement, axis=1)
    df["consensus_2reader_senior_attending"] = df.apply(two_reader_senior_attending, axis=1)

    out_cols = [
        "id", "reader1", "reader2", "reader3", "consensus_3reader",
        "complete_disagreement", "consensus_2reader_senior_attending",
    ]
    out = df[out_cols].copy()
    out.to_csv("manual_consensus_reference_standards.csv", index=False)

    print(f"Reports: {len(out)}")
    print(f"Complete three-way disagreement: {int(out['complete_disagreement'].sum())}")
    print("3-reader consensus distribution:")
    print(out["consensus_3reader"].value_counts().sort_index())
    print("Saved: manual_consensus_reference_standards.csv")


if __name__ == "__main__":
    main()
