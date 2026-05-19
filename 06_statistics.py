"""
06_statistics.py
================
REFINE-uyumlu istatistiksel analizler:
  - Pairwise Cohen's quadratic weighted kappa (95% bootstrap CI)
  - Fleiss' kappa (3 annotator, overall 3-reader agreement)
  - Gwet's AC1 with quadratic weights:
      * Pairwise (Resident1 vs Resident2): AC1=0.649
      * 3-rater overall: AC1=0.694
      * Attending vs Consensus: AC1=0.932
  - LLM vs consensus performance
  - Paired bootstrap kappa-difference test (LLM superiority over lexicon)
  - Landis & Koch interpretation categories

NOTLAR:
  - Post-hoc power analysis kaldırıldı (Hoenig & Heisey, 2001, Am Stat)
  - Gwet's AC1 pairwise (0.649) ≠ 3-rater overall AC1 (0.694)
    Makale Section 3.1'de her ikisi de ayrı ayrı raporlanır.

Beklenen çıktılar (gerçek veri ile doğrulandı):
  Claude vs Consensus:  κ=0.807 (95%CI 0.726–0.869) [Almost perfect]
  Gemini vs Consensus:  κ=0.814 (95%CI 0.739–0.872) [Almost perfect]
  Lexicon vs Consensus: κ=0.622 (95%CI 0.516–0.713) [Substantial]
  Inter-model:          κ=0.857 (95%CI 0.801–0.901) [Almost perfect]
  Fleiss κ:             0.273 [Fair]
  Resident1 vs Resident2 AC1=0.649
  3-rater overall AC1=0.694
  Attending vs Consensus AC1=0.932

Kullanım:
    python 06_statistics.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from irrCAC.raw import CAC
import warnings
warnings.filterwarnings('ignore')

BOOTSTRAP_N = 1000
SEED        = 42


# ─── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────


def landis_koch(k: float) -> str:
    """Landis & Koch (1977) kategorileri."""
    if k < 0.20: return "Slight"
    if k < 0.40: return "Fair"
    if k < 0.60: return "Moderate"
    if k < 0.80: return "Substantial"
    return "Almost perfect"


def kappa_ci(y_true, y_pred, n_iter=BOOTSTRAP_N, seed=SEED):
    """Quadratic weighted Cohen's kappa + 95% bootstrap CI."""
    np.random.seed(seed)
    y1 = np.array(y_true, dtype=int)
    y2 = np.array(y_pred, dtype=int)
    orig = cohen_kappa_score(y1, y2, weights='quadratic')
    boots, n = [], len(y1)
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        try:
            k = cohen_kappa_score(y1[idx], y2[idx], weights='quadratic')
            if not np.isnan(k):
                boots.append(k)
        except Exception:
            pass
    return orig, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def kappa_difference_ci(y_true, y_a, y_b, n_iter=BOOTSTRAP_N, seed=SEED):
    """
    Paired bootstrap kappa-difference test.
    H0: κ(A vs consensus) = κ(B vs consensus)
    Returns: delta_kappa, CI_lo, CI_hi
    If CI excludes 0 → statistically significant difference.
    """
    np.random.seed(seed)
    y1 = np.array(y_true, dtype=int)
    ya = np.array(y_a, dtype=int)
    yb = np.array(y_b, dtype=int)
    k_a = cohen_kappa_score(y1, ya, weights='quadratic')
    k_b = cohen_kappa_score(y1, yb, weights='quadratic')
    delta_obs = k_a - k_b
    boots, n = [], len(y1)
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        try:
            ka_b = cohen_kappa_score(y1[idx], ya[idx], weights='quadratic')
            kb_b = cohen_kappa_score(y1[idx], yb[idx], weights='quadratic')
            if not np.isnan(ka_b) and not np.isnan(kb_b):
                boots.append(ka_b - kb_b)
        except Exception:
            pass
    return delta_obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def fleiss_ci(scores_df, cols, n_cats=4, n_iter=BOOTSTRAP_N, seed=SEED):
    """Fleiss' kappa for 3+ raters + 95% bootstrap CI."""
    n = len(scores_df)
    mat = np.zeros((n, n_cats))
    for i, row in scores_df.reset_index(drop=True).iterrows():
        for c in cols:
            s = int(pd.to_numeric(row[c], errors='coerce'))
            if 0 <= s < n_cats:
                mat[i, s] += 1
    fk = fleiss_kappa(mat, method='fleiss')
    np.random.seed(seed)
    boots = []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        try:
            k = fleiss_kappa(mat[idx], method='fleiss')
            if not np.isnan(k):
                boots.append(k)
        except Exception:
            pass
    return fk, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def gwet_ac1_pairwise(s1, s2):
    """Gwet's AC1 quadratic weighted — pairwise two-rater."""
    data = pd.DataFrame({'r1': np.array(s1, dtype=int),
                         'r2': np.array(s2, dtype=int)}).dropna()
    cac = CAC(data, weights='quadratic')
    res = cac.gwet()['est']
    return (res['coefficient_value'],
            res['confidence_interval'][0],
            res['confidence_interval'][1])


def gwet_ac1_three_rater(s1, s2, s3):
    """Gwet's AC1 quadratic weighted — three-rater overall."""
    data = pd.DataFrame({'r1': np.array(s1, dtype=int),
                         'r2': np.array(s2, dtype=int),
                         'r3': np.array(s3, dtype=int)}).dropna()
    cac = CAC(data, weights='quadratic')
    res = cac.gwet()['est']
    return (res['coefficient_value'],
            res['confidence_interval'][0],
            res['confidence_interval'][1])


# ─── ANA FONKSİYON ────────────────────────────────────────────────────────


def main():
    # ── VERİ YÜKLEMESİ ─────────────────────────────────────────────────────
    df_ann = pd.read_excel(
        'Master_Etiketleme_3Annotator_sonuç.xlsx',
        sheet_name='Etiketlemeler'
    )
    df_llm   = pd.read_excel('triple_validation_100.xlsx')
    df_final = pd.read_excel('FINAL_3927_reports_with_claude.xlsx')

    # Konsensus hesapla (TARTIS → attending kararı)
    def final_consensus(row):
        val = str(row.get('Konsensus', '')).upper().strip()
        if val in ('TARTIS', 'NAN', ''):
            return int(row['Reader3_Score'])
        try:
            return int(float(val))
        except Exception:
            return int(row['Reader3_Score'])

    df_ann['Konsensus_Final'] = df_ann.apply(final_consensus, axis=1)
    tartis_n = (df_ann['Konsensus'].astype(str).str.upper().str.strip() == 'TARTIS').sum()

    df_ann['merge_id'] = df_ann['Rapor_ID'].astype(int)
    df_llm['merge_id'] = df_llm['id'].astype(int)
    df = pd.merge(df_ann, df_llm, on='merge_id', how='inner')

    # Lexicon skorlarını FINAL_3927'den çek
    df_fv = df_final[df_final['id'].isin(df['merge_id'])].copy()
    df_lex = pd.merge(
        df[['merge_id', 'Konsensus_Final']],
        df_fv[['id', 'lexicon_total']].rename(columns={'id': 'merge_id'}),
        on='merge_id', how='inner'
    )
    df_lex['lex_score'] = df_lex['lexicon_total'].clip(0, 3)

    y_cons  = df['Konsensus_Final'].astype(int)
    reader1 = df['Reader1_Score'].astype(int)
    reader2 = df['Reader2_Score'].astype(int)
    reader3 = df['Reader3_Score'].astype(int)
    claude  = df['claude_score'].astype(int)
    gemini  = df['gemini_score'].astype(int)
    lexicon = df_lex['lex_score'].astype(int)
    lex_cons = df_lex['Konsensus_Final'].astype(int)

    # ── SECTION 1: INTER-RATER AGREEMENT [REFINE 5.2] ──────────────────────
    print("=" * 65)
    print("REFINE 5.2 — INTER-RATER AGREEMENT")
    print("=" * 65)
    print(f"\nN = {len(df)} | TARTIS (complete disagreement) = {tartis_n}")
    print(f"\nConsensus distribution:")
    for v in sorted(y_cons.unique()):
        cnt = (y_cons == v).sum()
        print(f"  Score {v}: {cnt} ({100*cnt/len(y_cons):.0f}%)")

    print("\n--- Cohen's Quadratic Weighted Kappa (Pairwise) ---")
    pairs = [
        (reader1, reader3,    "Resident1 vs Attending"),
        (reader2, reader3,    "Resident2 vs Attending"),
        (reader1, reader2, "Resident1 vs Resident2"),
    ]
    for s1, s2, lbl in pairs:
        k, lo, hi = kappa_ci(s1, s2)
        print(f"  {lbl}: κ={k:.3f} (95%CI {lo:.3f}–{hi:.3f}) [{landis_koch(k)}]")

    fk, fk_lo, fk_hi = fleiss_ci(
        df[['Reader1_Score', 'Reader2_Score', 'Reader3_Score']],
        ['Reader1_Score', 'Reader2_Score', 'Reader3_Score']
    )
    print(f"\n--- Fleiss' Kappa (3 annotator overall) ---")
    print(f"  Fleiss' κ = {fk:.3f} (95%CI {fk_lo:.3f}–{fk_hi:.3f}) [{landis_koch(fk)}]")

    print(f"\n--- Gwet's AC1 (Quadratic Weighted) — Prevalence-robust ---")
    ac_r1r2, lo_r1r2, hi_r1r2 = gwet_ac1_pairwise(reader1, reader2)
    print(f"  Pairwise Resident1 vs Resident2: AC1={ac_r1r2:.3f} "
          f"(95%CI {lo_r1r2:.3f}–{hi_r1r2:.3f})")

    ac_3r, lo_3r, hi_3r = gwet_ac1_three_rater(reader1, reader2, reader3)
    print(f"  3-rater overall:                 AC1={ac_3r:.3f} "
          f"(95%CI {lo_3r:.3f}–{hi_3r:.3f})")

    ac_att, lo_att, hi_att = gwet_ac1_pairwise(reader3, y_cons)
    print(f"  Attending vs Consensus:          AC1={ac_att:.3f} "
          f"(95%CI {lo_att:.3f}–{hi_att:.3f})")

    print(f"\n--- Individual Annotator vs Consensus ---")
    for s, lbl in [(reader3, "Attending  "), (reader1, "Resident1  "), (reader2, "Resident2  ")]:
        k, lo, hi = kappa_ci(s, y_cons)
        print(f"  {lbl}: κ={k:.3f} (95%CI {lo:.3f}–{hi:.3f})")

    # ── SECTION 2: LLM PERFORMANCE [REFINE 5.1, 5.3] ───────────────────────
    print("\n" + "=" * 65)
    print("REFINE 5.1 & 5.3 — LLM vs CONSENSUS (PRIMARY RESULTS)")
    print("=" * 65)
    models = [
        (claude,  y_cons,   "Claude Opus 4.5  "),
        (gemini,  y_cons,   "Gemini 2.5 Pro   "),
        (lexicon, lex_cons, "Lexicon Baseline "),
    ]
    kappas = {}
    for scores, cons, lbl in models:
        k, lo, hi = kappa_ci(scores, cons)
        kappas[lbl.strip()] = k
        print(f"  {lbl}: κ={k:.3f} (95%CI {lo:.3f}–{hi:.3f}) [{landis_koch(k)}]")

    k_im, lo_im, hi_im = kappa_ci(claude, gemini)
    print(f"\n  Inter-model (Claude vs Gemini): κ={k_im:.3f} "
          f"(95%CI {lo_im:.3f}–{hi_im:.3f}) [{landis_koch(k_im)}]")

    # ── SECTION 3: PAIRED BOOTSTRAP KAPPA-DIFFERENCE TESTS ─────────────────
    print(f"\n--- Paired Bootstrap Kappa-Difference Tests (LLM vs Lexicon) ---")
    delta_c, dlo_c, dhi_c = kappa_difference_ci(y_cons, claude, lexicon[:len(y_cons)])
    print(f"  Δκ (Claude - Lexicon): {delta_c:+.3f} (95%CI {dlo_c:+.3f} to {dhi_c:+.3f})")
    print(f"  → {'CI excludes 0 → statistically significant' if dlo_c > 0 else 'CI includes 0'}")

    delta_g, dlo_g, dhi_g = kappa_difference_ci(y_cons, gemini, lexicon[:len(y_cons)])
    print(f"  Δκ (Gemini - Lexicon): {delta_g:+.3f} (95%CI {dlo_g:+.3f} to {dhi_g:+.3f})")
    print(f"  → {'CI excludes 0 → statistically significant' if dlo_g > 0 else 'CI includes 0'}")

    # ── SECTION 4: FAILURE ANALYSIS [REFINE 5.5] ────────────────────────────
    print("\n" + "=" * 65)
    print("REFINE 5.5 — FAILURE ANALYSIS (Claude Opus 4.5)")
    print("=" * 65)
    df['diff'] = (y_cons.values - claude.values)
    df['fark'] = df['diff'].abs()
    total = int((df['fark'] >= 1).sum())
    minor = int((df['fark'] == 1).sum())
    major = int((df['fark'] >= 2).sum())
    over  = int((claude.values > y_cons.values).sum())
    under = int((claude.values < y_cons.values).sum())

    print(f"  Total discordant: {total}/100 ({total:.0f}%)")
    print(f"  Minor (1-point):  {minor} ({100*minor/total:.1f}%)")
    print(f"  Major (≥2-point): {major} ({100*major/total:.1f}%)")
    print(f"  Over-hedging (Claude > Consensus): {over} ({100*over/total:.1f}%)")
    print(f"  Under-hedging (Claude < Consensus): {under} ({100*under/total:.1f}%)")
    print(f"\n  Major deviation cases (≥2-point difference):")
    for _, row in df[df['fark'] >= 2].iterrows():
        rid = int(row['merge_id'])
        cons_val = int(row['Konsensus_Final'])
        cl_val = int(row['claude_score'])
        print(f"    ID={rid}: Consensus={cons_val}, Claude={cl_val}")

    # ── VALIDATION CHECKS ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("VALIDATION CHECKS (vs manuscript claims)")
    print("=" * 65)
    ck, _, _ = kappa_ci(y_cons, claude)
    gk, _, _ = kappa_ci(y_cons, gemini)
    lk_val, _, _ = kappa_ci(lex_cons, lexicon)
    checks = [
        ("Claude κ ≈ 0.807",   abs(ck     - 0.807) < 0.005),
        ("Gemini κ ≈ 0.814",   abs(gk     - 0.814) < 0.005),
        ("Lexicon κ ≈ 0.622",  abs(lk_val - 0.622) < 0.005),
        ("Fleiss κ ≈ 0.273",   abs(fk     - 0.273) < 0.005),
        ("AC1 Res1vRes2 ≈ 0.649", abs(ac_r1r2 - 0.649) < 0.005),
        ("AC1 3-rater ≈ 0.694",   abs(ac_3r   - 0.694) < 0.005),
        ("AC1 Att vs Cons ≈ 0.932", abs(ac_att - 0.932) < 0.005),
        ("Discordant = 32",    total == 32),
        ("Major = 2",          major == 2),
        ("Over-hedging = 25",  over  == 25),
    ]
    all_pass = True
    for name, ok in checks:
        status = "✓" if ok else "✗"
        if not ok: all_pass = False
        print(f"  {status} {name}")
    print(f"\n{'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")


if __name__ == '__main__':
    main()
