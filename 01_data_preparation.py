"""
01_data_preparation.py
======================
OpenI Indiana University Chest X-ray dataset indirme ve lexicon analizi.

Exclusion: 28 rapor (eksik Findings veya Impression) → 3,927 final rapor.

Çıktı:
    FINAL_3927_reports_with_claude.xlsx  (lexicon skorları ile)

Kullanım:
    python 01_data_preparation.py

Referans: Demner-Fushman et al. (2016). J Am Med Inform Assoc. 23(2):304-310.
"""

import pandas as pd
import requests
import xml.etree.ElementTree as ET
import io
import tarfile

# ─── LEXICON TANIMI ────────────────────────────────────────────────────────
# Üç semantik kategori: epistemic, lexical (hedging), evidential
# Makale Methods 2.3 ile birebir eşleşir.
hedging_categories = {
    'epistemic': [
        'possible', 'possibly', 'probable', 'probably',
        'likely', 'unlikely', 'may', 'might', 'could',
        'perhaps', 'apparent', 'apparently', 'seems', 'suggest',
        'suggests', 'suggested', 'cannot exclude', 'cannot be excluded',
        'cannot rule out', 'suspicious', 'suspicious for',
        'worrisome', 'concerning', 'concern for', 'questionable',
        'equivocal', 'indeterminate', 'uncertain', 'uncertainty'
    ],
    'lexical': [
        'consistent with', 'consistent with possible',
        'suggestive of', 'compatible with',
        'may represent', 'could represent', 'might represent',
        'in keeping with', 'raises the possibility',
        'raises concern', 'differential includes',
        'differential diagnosis', 'vs', 'versus',
        'cannot differentiate', 'difficult to exclude'
    ],
    'evidential': [
        'appears', 'appears to', 'appear', 'appear to',
        'seems to', 'is seen', 'is noted', 'is identified',
        'is visualized', 'is demonstrated', 'is detected',
        'is suspected', 'is felt to', 'is thought to'
    ]
}


def count_hedges(text):
    """Hedge terim sayısını kategoriye göre döndür."""
    if pd.isna(text):
        return 0, 0, 0
    text_lower = str(text).lower()
    epistemic  = sum(1 for t in hedging_categories['epistemic']  if t in text_lower)
    lexical    = sum(1 for t in hedging_categories['lexical']    if t in text_lower)
    evidential = sum(1 for t in hedging_categories['evidential'] if t in text_lower)
    return epistemic, lexical, evidential


def hedge_to_score(total):
    """
    Ham hedge sayısını 0-3 ordinal skoreye dönüştür (cap at 3).
    0 hedge → score 0 (None)
    1 hedge → score 1 (Low)
    2 hedge → score 2 (Moderate)
    ≥3 hedge → score 3 (High)
    """
    if total == 0: return 0
    if total == 1: return 1
    if total == 2: return 2
    return 3


def download_openi():
    """OpenI dataset'ini indir, XML parse et, 28 eksik raporu dışla."""
    print("OpenI rapor arşivi indiriliyor...")
    url = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    tar = tarfile.open(fileobj=io.BytesIO(response.content))

    reports  = []
    excluded = 0

    for member in tar.getmembers():
        if not member.name.endswith('.xml'):
            continue
        f = tar.extractfile(member)
        if not f:
            continue
        try:
            tree = ET.parse(f)
            root = tree.getroot()
            report_id  = root.attrib.get('id', member.name)
            findings   = ''
            impression = ''
            mesh_terms = []

            for elem in root.iter('AbstractText'):
                label = elem.attrib.get('Label', '')
                if label == 'FINDINGS':
                    findings = elem.text or ''
                elif label == 'IMPRESSION':
                    impression = elem.text or ''

            for elem in root.iter('term'):
                if elem.text:
                    mesh_terms.append(elem.text.strip())

            # Dışlama kriteri: eksik Findings VEYA Impression
            if not findings.strip() or not impression.strip():
                excluded += 1
                continue

            reports.append({
                'id':         report_id,
                'findings':   findings,
                'impression': impression,
                'mesh_terms': '|'.join(mesh_terms)
            })
        except Exception:
            excluded += 1

    print(f"✓ Dahil edilen: {len(reports)} rapor")
    print(f"✗ Dışlanan:     {excluded} rapor (eksik findings/impression)")
    assert len(reports) == 3927, f"Beklenmedik rapor sayısı: {len(reports)}"
    return pd.DataFrame(reports)


def main():
    df = download_openi()
    print(f"\nLexicon analizi yapılıyor (n={len(df)})...")

    df['text_combined'] = df['findings'].fillna('') + ' ' + df['impression'].fillna('')

    epistemic_vals, lexical_vals, evidential_vals = zip(
        *df['text_combined'].apply(count_hedges)
    )
    df['lexicon_epistemic']  = epistemic_vals
    df['lexicon_lexical']    = lexical_vals
    df['lexicon_evidential'] = evidential_vals
    df['lexicon_total']      = (df['lexicon_epistemic'] +
                                df['lexicon_lexical'] +
                                df['lexicon_evidential'])
    df['lexicon_score'] = df['lexicon_total'].apply(hedge_to_score)

    print(f"\n✓ Lexicon skoru dağılımı (stratified sampling için):")
    print(df['lexicon_score'].value_counts().sort_index())

    out_file = 'FINAL_3927_reports_with_claude.xlsx'
    df.to_excel(out_file, index=False)
    print(f"\n✓ Kaydedildi: {out_file}")


if __name__ == '__main__':
    main()
