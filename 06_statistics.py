"""
06_statistics.py
================
REFINE-aligned statistical analysis for the Path-2 feasibility/methodology manuscript.

This anonymized, reviewer-facing script can run in two modes:

1) Preferred reproducibility mode:
   - Uses analysis_ready_validation_100.csv
   - Contains only public report IDs and derived ordinal scores; no raw report text
   - Reproduces Table 2, Table 4, Supplementary 7 confusion matrices, and the
     LLM-vs-lexicon comparison used in the contamination/memorization discussion

2) Source-file mode:
   - Uses neutral manual annotation and LLM-score files if available
   - Does not support working-file columns containing author identifiers

Path-2 framing:
The OpenI validation cohort is probably not model-unseen. Therefore all kappa
estimates are upper-bound feasibility estimates, not generalization benchmark
performance.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import fleiss_kappa

warnings.filterwarnings("ignore")

BOOTSTRAP_N = 1000
SEED = 42
CATEGORIES = [0, 1, 2, 3]
OUTDIR = Path("outputs")


NEUTRAL_ANNOTATION_ALIASES = {
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


REQUIRED_ANALYSIS_COLUMNS = [
    "id",
    "reader1",
    "reader2",
    "reader3",
    "consensus_3reader",
    "complete_disagreement",
    "consensus_2reader_senior_attending",
    "claude_score",
    "gemini_score",
    "lexicon_score",
]


def landis_koch(kappa: float) -> str:
    """Landis and Koch categories; no hybrid labels are used."""
    if kappa <= 0.20:
        return "Slight"
    if kappa <= 0.40:
        return "Fair"
    if kappa <= 0.60:
        return "Moderate"
    if kappa <= 0.80:
        return "Substantial"
    return "Almost perfect"


def reported_agreement_label(name: str, kappa: float, ci_lower: float) -> str:
    """
    Conservative manuscript-facing agreement label.

    For model-to-consensus and lexicon-to-consensus rows, the label uses the
    lower bound of the 95% CI. Inter-model agreement is descriptive and uses the
    point estimate category.
    """
    if "Inter-model" in name:
        return landis_koch(float(kappa))
    return landis_koch(float(ci_lower))


def kappa_ci(y_true, y_pred, n_iter: int = BOOTSTRAP_N, seed: int = SEED):
    """Quadratic weighted Cohen kappa with percentile bootstrap CI."""
    np.random.seed(seed)  # kept to reproduce manuscript CIs exactly
    y1 = np.asarray(y_true, dtype=int)
    y2 = np.asarray(y_pred, dtype=int)
    orig = cohen_kappa_score(y1, y2, weights="quadratic")
    boots = []
    n = len(y1)
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        kappa = cohen_kappa_score(y1[idx], y2[idx], weights="quadratic")
        if not np.isnan(kappa):
            boots.append(kappa)
    return float(orig), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fleiss_ci(scores_df: pd.DataFrame, cols, n_cats: int = 4, n_iter: int = BOOTSTRAP_N, seed: int = SEED):
    """Fleiss kappa with bootstrap CI."""
    mat = np.zeros((len(scores_df), n_cats))
    for i, row in scores_df.reset_index(drop=True).iterrows():
        for col in cols:
            score = int(row[col])
            if 0 <= score < n_cats:
                mat[i, score] += 1
    fleiss_value = fleiss_kappa(mat, method="fleiss")
    np.random.seed(seed)
    boots = []
    n = len(scores_df)
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        kappa = fleiss_kappa(mat[idx], method="fleiss")
        if not np.isnan(kappa):
            boots.append(kappa)
    return float(fleiss_value), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def try_gwet_ac1(data: pd.DataFrame):
    """Gwet AC1 with quadratic weights, if irrCAC is installed."""
    try:
        from irrCAC.raw import CAC
        cac = CAC(data.astype(int), weights="quadratic")
        est = cac.gwet()["est"]
        return float(est["coefficient_value"]), float(est["confidence_interval"][0]), float(est["confidence_interval"][1])
    except Exception:
        return np.nan, np.nan, np.nan


def find_existing(candidates) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    for sheet in ["Annotations", "Sheet1", 0]:
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path}")


def normalize_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize neutral annotation columns only; author-name aliases are not supported."""
    column_map = {src: dst for src, dst in NEUTRAL_ANNOTATION_ALIASES.items() if src in df.columns}
    out = df.rename(columns=column_map).copy()
    required = ["id", "reader1", "reader2", "reader3"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing neutral annotation columns after normalization: {missing}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(int)
    if "consensus_raw" not in out.columns:
        out["consensus_raw"] = ""
    return out


def majority_or_attending(row) -> int:
    scores = [int(row["reader1"]), int(row["reader2"]), int(row["reader3"])]
    for score in CATEGORIES:
        if scores.count(score) >= 2:
            return score
    return int(row["reader3"])


def two_reader_senior_attending(row) -> int:
    """
    Editor-requested PGY-2/operator-excluded sensitivity reference.

    Because the attending radiologist is the tie-breaker for senior-attending
    disagreements, this reference is equivalent by construction to the attending
    radiologist's independent annotation. This is reported transparently as an
    LLM-versus-attending sensitivity analysis.
    """
    if int(row["reader2"]) == int(row["reader3"]):
        return int(row["reader2"])
    return int(row["reader3"])


def complete_disagreement(row) -> bool:
    return len({int(row["reader1"]), int(row["reader2"]), int(row["reader3"])}) == 3


def add_lexicon_score(df: pd.DataFrame) -> pd.Series:
    if "lexicon_score" in df.columns:
        return pd.to_numeric(df["lexicon_score"], errors="coerce").clip(0, 3).astype(int)
    if "lexicon_total" in df.columns:
        return pd.to_numeric(df["lexicon_total"], errors="coerce").clip(0, 3).astype(int)
    raise ValueError("LLM score file must include lexicon_score or lexicon_total")


def load_analysis_ready() -> pd.DataFrame | None:
    path = Path("analysis_ready_validation_100.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_ANALYSIS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"analysis_ready_validation_100.csv is missing columns: {missing}")
    for col in REQUIRED_ANALYSIS_COLUMNS:
        if col != "complete_disagreement":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    df["complete_disagreement"] = df["complete_disagreement"].astype(bool)
    return df


def load_from_source_files() -> pd.DataFrame:
    annotation_file = find_existing([
        "annotations_3readers.xlsx",
        "annotations_3readers.csv",
        "manual_annotations_3readers.xlsx",
        "manual_annotations_3readers.csv",
    ])
    llm_file = find_existing(["triple_validation_100.xlsx", "validation_scores_100.csv"])
    if annotation_file is None or llm_file is None:
        raise FileNotFoundError(
            "Could not find analysis_ready_validation_100.csv or the neutral source files. "
            "Provide analysis_ready_validation_100.csv for reviewer-side reproduction."
        )

    df_ann = normalize_annotation_columns(read_table(annotation_file))
    df_llm = read_table(llm_file).copy()
    df_llm["id"] = pd.to_numeric(df_llm["id"], errors="coerce").astype(int)
    df_llm["lexicon_score"] = add_lexicon_score(df_llm)

    df_ann["consensus_3reader"] = df_ann.apply(majority_or_attending, axis=1)
    df_ann["consensus_2reader_senior_attending"] = df_ann.apply(two_reader_senior_attending, axis=1)
    df_ann["complete_disagreement"] = df_ann.apply(complete_disagreement, axis=1)

    df = pd.merge(df_ann, df_llm, on="id", how="inner")
    for col in ["claude_score", "gemini_score", "lexicon_score"]:
        if col not in df.columns:
            raise ValueError(f"Missing required score column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    return df[REQUIRED_ANALYSIS_COLUMNS].copy()


def metric_row(name: str, y_ref, y_pred) -> dict:
    kappa, lo, hi = kappa_ci(y_ref, y_pred)
    return {
        "Model/Method": name,
        "Quadratic weighted kappa": round(kappa, 3),
        "95% CI lower": round(lo, 3),
        "95% CI upper": round(hi, 3),
        "Agreement level (reported; no hybrid labels)": reported_agreement_label(name, kappa, lo),
    }


def print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    print(df.to_string(index=False))


def write_confusion_matrix(y_ref, y_pred, name: str) -> None:
    mat = confusion_matrix(y_ref, y_pred, labels=CATEGORIES)
    mat_df = pd.DataFrame(
        mat,
        index=[f"Consensus {i}" for i in CATEGORIES],
        columns=[f"Pred {i}" for i in CATEGORIES],
    )
    mat_df.to_csv(OUTDIR / f"confusion_matrix_{name}.csv")


def run_llm_lexicon_probe(df: pd.DataFrame, y_ref: pd.Series) -> pd.DataFrame:
    """Reproduce the LLM-lexicon comparison used in the contamination discussion."""
    lex = df["lexicon_score"].astype(int)
    claude = df["claude_score"].astype(int)
    gemini = df["gemini_score"].astype(int)

    claude_lex_kappa = cohen_kappa_score(claude, lex, weights="quadratic")
    gemini_lex_kappa = cohen_kappa_score(gemini, lex, weights="quadratic")

    lex_wrong = lex != y_ref
    claude_correct_when_lex_wrong = int(((claude == y_ref) & lex_wrong).sum())
    gemini_correct_when_lex_wrong = int(((gemini == y_ref) & lex_wrong).sum())
    any_llm_correct_when_lex_wrong = int((((claude == y_ref) | (gemini == y_ref)) & lex_wrong).sum())
    both_llm_correct_when_lex_wrong = int((((claude == y_ref) & (gemini == y_ref)) & lex_wrong).sum())

    review_file = Path("memorization_probe_manual_review.csv")
    negation_cases = np.nan
    if review_file.exists():
        review = pd.read_csv(review_file)
        if "negation_exclusion_case" in review.columns:
            negation_cases = int(pd.to_numeric(review["negation_exclusion_case"], errors="coerce").fillna(0).sum())

    rows = [
        {"Metric": "Claude-Lexicon quadratic weighted kappa", "Value": round(float(claude_lex_kappa), 3)},
        {"Metric": "Gemini-Lexicon quadratic weighted kappa", "Value": round(float(gemini_lex_kappa), 3)},
        {"Metric": "Lexicon-wrong cases", "Value": int(lex_wrong.sum())},
        {"Metric": "Claude correct when lexicon was wrong", "Value": claude_correct_when_lex_wrong},
        {"Metric": "Gemini correct when lexicon was wrong", "Value": gemini_correct_when_lex_wrong},
        {"Metric": "Any LLM correct when lexicon was wrong", "Value": any_llm_correct_when_lex_wrong},
        {"Metric": "Both LLMs correct when lexicon was wrong", "Value": both_llm_correct_when_lex_wrong},
        {"Metric": "Manual review: negation-exclusion cases", "Value": negation_cases},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTDIR / "llm_lexicon_probe_summary.csv", index=False)
    return summary


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)

    df = load_analysis_ready()
    if df is None:
        df = load_from_source_files()

    y = df["consensus_3reader"].astype(int)

    primary = pd.DataFrame([
        metric_row("Gemini 2.5 Pro", y, df["gemini_score"]),
        metric_row("Claude Opus 4.5", y, df["claude_score"]),
        metric_row("Lexicon baseline", y, df["lexicon_score"]),
        metric_row("Inter-model agreement (Claude vs Gemini)", df["claude_score"], df["gemini_score"]),
    ])
    print_table("PRIMARY RESULTS - upper-bound feasibility estimates on OpenI", primary)
    primary.to_csv(OUTDIR / "table2_primary_performance.csv", index=False)

    inter_rows = []
    for a, b, name in [
        ("reader1", "reader3", "PGY-2/operator vs attending"),
        ("reader2", "reader3", "Senior resident vs attending"),
        ("reader1", "reader2", "PGY-2/operator vs senior resident"),
        ("reader3", "consensus_3reader", "Attending vs final consensus"),
    ]:
        inter_rows.append(metric_row(name, df[a], df[b]))
    fleiss_value, fleiss_lo, fleiss_hi = fleiss_ci(df, ["reader1", "reader2", "reader3"])
    inter_rows.append({
        "Model/Method": "Fleiss kappa (3-reader overall)",
        "Quadratic weighted kappa": round(fleiss_value, 3),
        "95% CI lower": round(fleiss_lo, 3),
        "95% CI upper": round(fleiss_hi, 3),
        "Agreement level (reported; no hybrid labels)": landis_koch(fleiss_value),
    })
    inter = pd.DataFrame(inter_rows)
    print_table("INTER-RATER RELIABILITY", inter)
    inter.to_csv(OUTDIR / "inter_rater_reliability.csv", index=False)

    ac_pair = try_gwet_ac1(df[["reader1", "reader2"]])
    ac_three = try_gwet_ac1(df[["reader1", "reader2", "reader3"]])
    with open(OUTDIR / "gwet_ac1_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Gwet AC1, pairwise PGY-2/operator vs senior resident: {ac_pair[0]:.3f} ({ac_pair[1]:.3f}-{ac_pair[2]:.3f})\n")
        f.write(f"Gwet AC1, 3-reader overall: {ac_three[0]:.3f} ({ac_three[1]:.3f}-{ac_three[2]:.3f})\n")

    y2 = df["consensus_2reader_senior_attending"].astype(int)
    sens_rows = [
        metric_row("Claude Opus 4.5 vs attending/two-reader sensitivity reference", y2, df["claude_score"]),
        metric_row("Gemini 2.5 Pro vs attending/two-reader sensitivity reference", y2, df["gemini_score"]),
    ]

    df_no_disagree = df[~df["complete_disagreement"]].copy()
    y_no = df_no_disagree["consensus_3reader"].astype(int)
    sens_rows += [
        metric_row("Claude Opus 4.5 excluding complete-disagreement cases", y_no, df_no_disagree["claude_score"]),
        metric_row("Gemini 2.5 Pro excluding complete-disagreement cases", y_no, df_no_disagree["gemini_score"]),
    ]

    y2_no = df_no_disagree["consensus_2reader_senior_attending"].astype(int)
    sens_rows += [
        metric_row("Claude Opus 4.5, attending/two-reader + no complete-disagreement cases", y2_no, df_no_disagree["claude_score"]),
        metric_row("Gemini 2.5 Pro, attending/two-reader + no complete-disagreement cases", y2_no, df_no_disagree["gemini_score"]),
    ]
    sensitivity = pd.DataFrame(sens_rows)
    print_table("SENSITIVITY ANALYSES", sensitivity)
    sensitivity.to_csv(OUTDIR / "table4_sensitivity_analyses.csv", index=False)

    note = (
        "The PGY-2/operator-excluded two-reader reference uses the attending radiologist "
        "as the tie-breaker for senior-attending disagreements; it is therefore equivalent "
        "by construction to agreement against the attending radiologist's independent scores.\n"
    )
    with open(OUTDIR / "sensitivity_interpretation_note.txt", "w", encoding="utf-8") as f:
        f.write(note)

    write_confusion_matrix(y, df["claude_score"], "claude")
    write_confusion_matrix(y, df["gemini_score"], "gemini")
    write_confusion_matrix(y, df["lexicon_score"], "lexicon")

    fail_rows = []
    for model_col, model_name in [("claude_score", "Claude Opus 4.5"), ("gemini_score", "Gemini 2.5 Pro"), ("lexicon_score", "Lexicon baseline")]:
        diff = df[model_col].astype(int) - y
        fail_rows.append({
            "Model": model_name,
            "Discordant cases": int((diff.abs() >= 1).sum()),
            "Minor deviations (1-point)": int((diff.abs() == 1).sum()),
            "Major deviations (>=2-point)": int((diff.abs() >= 2).sum()),
            "Over-hedging (model > consensus)": int((diff > 0).sum()),
            "Under-hedging (model < consensus)": int((diff < 0).sum()),
        })
    failure = pd.DataFrame(fail_rows)
    print_table("FAILURE ANALYSIS SUMMARY", failure)
    failure.to_csv(OUTDIR / "failure_analysis_summary.csv", index=False)

    probe = run_llm_lexicon_probe(df, y)
    print_table("LLM-LEXICON COMPARISON / MEMORIZATION-PROBE SUPPORT", probe)

    print(f"\nComplete-disagreement cases: {int(df['complete_disagreement'].sum())}/{len(df)}")
    print("Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
