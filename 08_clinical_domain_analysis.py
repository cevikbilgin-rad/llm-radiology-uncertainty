"""
08_clinical_domain_analysis.py
===============================
Discovery cohort (n=3,927) için klinik domain bazlı uncertainty analizi.

NOT (Eski 30 MeSH kategorisi → 6 ana klinik domain'e mapping):
30 MeSH kategorisi için ayrı ayrı chi-square yapılması istatistiksel
varsayım ihlali yarattı (n<5 hücreler, Tip I hata riski). Bu sürümde:
  - MeSH terimleri 6 ana klinik domain'e mapping edilir
  - Fisher's exact test kullanılır (chi-square varsayım ihlalini çözer)
  - Multiple comparison için Benjamini-Hochberg FDR uygulanır
    (Bonferroni'den daha modern, daha az muhafazakar)

Çıktı:
  - Figure3_Clinical_Domain_Uncertainty.png
  - clinical_domain_stats.xlsx

Beklenen sonuçlar:
  Parenchymal:     82.6% (n=213, p<0.001)
  Cardiovascular:  49.0% (n=310, p<0.001)
  Pleural:         70.7% (n=41,  p<0.001)
  Skeletal/Other:  35.4% (n=48,  NS)
  Devices/Support: 30.2% (n=43,  NS)
  Other/Technical: 83.3% (n=6,   p<0.05)

Kullanım:
    python 08_clinical_domain_analysis.py
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


# ─── KLİNİK DOMAIN MAPPING ─────────────────────────────────────────────────
# 30 MeSH terimi → 6 ana klinik radyolojik domain
DOMAIN_MAPPING = {
    # Parenchymal (Akciğer dokusu)
    'Opacity':       'Parenchymal',
    'Pneumonia':     'Parenchymal',
    'Atelectasis':   'Parenchymal',
    'Infiltration':  'Parenchymal',
    'Consolidation': 'Parenchymal',
    'Lung Neoplasms':'Parenchymal',
    'Emphysema':     'Parenchymal',
    'Tuberculosis':  'Parenchymal',
    'Lung Diseases': 'Parenchymal',

    # Pleural
    'Pleural Effusion': 'Pleural',
    'Pneumothorax':     'Pleural',
    'Pleural Diseases': 'Pleural',

    # Cardiovascular
    'Cardiomegaly':       'Cardiovascular',
    'Heart Enlargement':  'Cardiovascular',
    'Aortic Diseases':    'Cardiovascular',
    'Aorta':              'Cardiovascular',
    'Atherosclerosis':    'Cardiovascular',

    # Devices/Support
    'Catheters':            'Devices/Support',
    'Pacemaker':            'Devices/Support',
    'Tube':                 'Devices/Support',
    'Surgical Instruments': 'Devices/Support',
    'Indwelling':           'Devices/Support',

    # Skeletal/Other
    'Rib Fractures': 'Skeletal/Other',
    'Scoliosis':     'Skeletal/Other',
    'Bone':          'Skeletal/Other',

    # Other/Technical
    'Technical Quality of Image Unsatisfactory': 'Other/Technical',
}


def run_domain_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Fisher's exact + FDR per clinical domain."""
    df['is_uncertain'] = (df['claude_score'] > 0).astype(int)
    total_n = len(df)
    total_u = int(df['is_uncertain'].sum())

    # Her raporu domain'lerine genişlet
    rows = []
    for _, row in df.iterrows():
        if pd.isna(row['mesh_terms']):
            continue
        terms = [t.split('/')[0].strip() for t in str(row['mesh_terms']).split('|')]
        unique_domains = {DOMAIN_MAPPING.get(t) for t in terms if DOMAIN_MAPPING.get(t)}
        for dom in unique_domains:
            rows.append({'domain': dom, 'uncertain': int(row['is_uncertain'])})

    analysis_df = pd.DataFrame(rows)
    stats = []

    for domain in analysis_df['domain'].unique():
        subset = analysis_df[analysis_df['domain'] == domain]
        n_dom = len(subset)
        u_dom = int(subset['uncertain'].sum())

        # 2x2 tablo
        table = [
            [u_dom,           n_dom - u_dom],
            [total_u - u_dom, (total_n - total_u) - (n_dom - u_dom)]
        ]
        _, p_val = fisher_exact(table)

        stats.append({
            'Clinical Domain': domain,
            'Total N':         n_dom,
            'Uncertain N':     u_dom,
            'Rate (%)':        round(100 * u_dom / n_dom, 1),
            'p_raw':           p_val
        })

    res = pd.DataFrame(stats)

    # FDR (Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(res['p_raw'], method='fdr_bh')
    res['p_adj_FDR'] = p_adj
    res['Significant'] = res['p_adj_FDR'] < 0.05
    return res.sort_values('Rate (%)', ascending=False).reset_index(drop=True)


def plot_figure3(stats_df: pd.DataFrame):
    """Figure 3 grafiği."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    plot_df = stats_df.copy()
    ax = sns.barplot(
        x='Rate (%)', y='Clinical Domain', data=plot_df,
        hue='Clinical Domain', palette='Reds_r', legend=False
    )

    plt.title('Figure 3: Uncertainty Rate by Clinical Domain (Discovery Cohort)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Uncertainty Rate (%) - (Claude Score > 0)', fontsize=12)
    plt.xlim(0, 100)

    # n= etiketleri
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + 1, p.get_y() + p.get_height() / 2,
                f'n={int(plot_df.iloc[i]["Total N"])}',
                va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('Figure3_Clinical_Domain_Uncertainty.png', dpi=300, bbox_inches='tight')
    print("✓ Grafik kaydedildi: Figure3_Clinical_Domain_Uncertainty.png")


def main():
    df = pd.read_excel('FINAL_3927_reports_with_claude.xlsx')
    print(f"Toplam rapor: {len(df)}")

    stats_df = run_domain_analysis(df)

    print("\n" + "=" * 75)
    print("CLINICAL DOMAIN ANALYSIS (Fisher's Exact + Benjamini-Hochberg FDR)")
    print("=" * 75)
    print(stats_df[['Clinical Domain', 'Total N', 'Uncertain N',
                    'Rate (%)', 'p_adj_FDR', 'Significant']].to_string(index=False))

    plot_figure3(stats_df)
    stats_df.to_excel('clinical_domain_stats.xlsx', index=False)
    print("\n✓ Tablo kaydedildi: clinical_domain_stats.xlsx")


if __name__ == '__main__':
    main()
