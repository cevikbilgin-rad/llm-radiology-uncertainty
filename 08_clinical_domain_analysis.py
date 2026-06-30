"""
08_clinical_domain_analysis.py
==============================
Exploratory descriptive analysis of model-derived uncertainty distribution across
aggregated clinical domains in the discovery cohort.

Path-2 revision note:
This script intentionally performs NO p-value testing, NO FDR correction, and NO
claims of clinical validation. The discovery cohort lacks an independent human
reference standard; therefore outputs describe model behavior only.

Expected manuscript values after filtering/mapping:
  Parenchymal:      82.6% (n=213)
  Pleural:          70.7% (n=41)
  Cardiovascular:   49.0% (n=310)
  Skeletal/Other:   35.4% (n=48)
  Devices/Support:  30.2% (n=43)
  Other/Technical:  83.3% (n=6; flagged unstable, excluded from primary interpretation)

Usage:
    python 08_clinical_domain_analysis.py --input discovery_set_3827_with_claude.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOMAIN_MAPPING = {
    # Parenchymal
    "Opacity": "Parenchymal",
    "Pneumonia": "Parenchymal",
    "Atelectasis": "Parenchymal",
    "Infiltration": "Parenchymal",
    "Consolidation": "Parenchymal",
    "Lung Neoplasms": "Parenchymal",
    "Emphysema": "Parenchymal",
    "Tuberculosis": "Parenchymal",
    "Lung Diseases": "Parenchymal",
    # Pleural
    "Pleural Effusion": "Pleural",
    "Pneumothorax": "Pleural",
    "Pleural Diseases": "Pleural",
    # Cardiovascular
    "Cardiomegaly": "Cardiovascular",
    "Heart Enlargement": "Cardiovascular",
    "Aortic Diseases": "Cardiovascular",
    "Aorta": "Cardiovascular",
    "Atherosclerosis": "Cardiovascular",
    # Devices/Support
    "Catheters": "Devices/Support",
    "Pacemaker": "Devices/Support",
    "Tube": "Devices/Support",
    "Surgical Instruments": "Devices/Support",
    "Indwelling": "Devices/Support",
    # Skeletal/Other
    "Rib Fractures": "Skeletal/Other",
    "Scoliosis": "Skeletal/Other",
    "Bone": "Skeletal/Other",
    # Other/Technical
    "Technical Quality of Image Unsatisfactory": "Other/Technical",
}
DOMAIN_ORDER = ["Parenchymal", "Pleural", "Cardiovascular", "Skeletal/Other", "Devices/Support", "Other/Technical"]


def find_default_input():
    for name in [
        "discovery_set_3827_with_claude.xlsx",
        "FINAL_3927_reports_with_claude.xlsx",
        "openi_reports_3927_with_claude.xlsx",
    ]:
        if Path(name).exists():
            return name
    return None


def clinical_domains(mesh_terms) -> set[str]:
    if pd.isna(mesh_terms):
        return set()
    domains = set()
    for raw in str(mesh_terms).split("|"):
        term = raw.split("/")[0].strip()
        if term in DOMAIN_MAPPING:
            domains.add(DOMAIN_MAPPING[term])
    return domains


def run_domain_analysis(df: pd.DataFrame) -> pd.DataFrame:
    score_col = "claude_score" if "claude_score" in df.columns else "model_score"
    if score_col not in df.columns:
        raise ValueError("Input must contain 'claude_score' or 'model_score'.")
    df = df.copy()
    df["is_uncertain"] = (pd.to_numeric(df[score_col], errors="coerce").fillna(0).astype(int) > 0).astype(int)

    rows = []
    for _, row in df.iterrows():
        for dom in clinical_domains(row.get("mesh_terms", "")):
            rows.append({"domain": dom, "is_uncertain": int(row["is_uncertain"])})
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError("No mapped clinical domains found. Check mesh_terms column and DOMAIN_MAPPING.")

    stats = []
    for dom in DOMAIN_ORDER:
        sub = expanded[expanded["domain"] == dom]
        n = len(sub)
        u = int(sub["is_uncertain"].sum()) if n else 0
        stats.append({
            "Clinical Domain": dom,
            "Total N": n,
            "Uncertain N": u,
            "Rate (%)": round(100 * u / n, 1) if n else np.nan,
            "Interpretation": "flagged unstable; excluded from primary interpretation" if n < 10 else "descriptive model-derived estimate",
        })
    return pd.DataFrame(stats)


def plot_figure4(stats: pd.DataFrame, output_png: str):
    plot_df = stats.copy()
    plot_df["Rate (%)"] = pd.to_numeric(plot_df["Rate (%)"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 7.5))
    y_pos = np.arange(len(plot_df))
    colors = ["#c9312a", "#df5146", "#ec7c66", "#f2a18d", "#f6c2ad", "#f2f2f2"]
    bars = ax.barh(y_pos, plot_df["Rate (%)"], color=colors, edgecolor="#666666", linewidth=0.6)
    # Hatch small-n/unstable category
    bars[-1].set_hatch("////")
    bars[-1].set_edgecolor("#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Clinical Domain"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Model-derived uncertainty rate (%) - Claude Opus 4.5 score > 0")
    ax.set_title(
        "Figure 4. Exploratory descriptive analysis of model-derived uncertainty distribution\n"
        "across aggregated clinical domains (Discovery Cohort, N=3,827)",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for i, row in plot_df.iterrows():
        rate = row["Rate (%)"]
        n = int(row["Total N"])
        sample_label = f"n={n}" if n >= 10 else f"n={n} (caution: small n)"
        ax.text(min(rate + 1.5, 92), i, f"{rate:.1f}%", va="center", ha="left", fontweight="bold")
        ax.text(min(rate + 11, 96), i, sample_label, va="center", ha="left")

    foot = ("* Other/Technical (n=6): Sample size below stability threshold (n<10);\n"
            "rate flagged as unstable and excluded from primary interpretation.\n"
            "Hatched bar indicates outlier status.")
    ax.text(0.98, -0.16, foot, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff8e8", edgecolor="#888888"))
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Discovery cohort Excel file with mesh_terms and claude_score")
    parser.add_argument("--output-prefix", default="figure4_domain_uncertainty_descriptive")
    args = parser.parse_args()

    input_path = args.input or find_default_input()
    if input_path is None:
        raise FileNotFoundError("No discovery cohort input found. Use --input discovery_set_3827_with_claude.xlsx")
    df = pd.read_excel(input_path)
    stats = run_domain_analysis(df)
    stats.to_excel(f"{args.output_prefix}.xlsx", index=False)
    stats.to_csv(f"{args.output_prefix}.csv", index=False)
    plot_figure4(stats, f"{args.output_prefix}.png")
    print(stats.to_string(index=False))
    print("Descriptive outputs only: no p-values, no FDR, no significance claims.")


if __name__ == "__main__":
    main()
