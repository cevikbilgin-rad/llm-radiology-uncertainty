"""
05_consensus.py
===============
3 annotator skorlarından çoğunluk oyu konsensus + uzman hakemlik
(majority vote with expert adjudication).

Tam anlaşmazlık (TARTIS) → attending kararı (Reader3_Score).

Kullanım:
    python 05_consensus.py
"""

import pandas as pd

XLSX_FILE = "Master_Etiketleme_3Annotator sonuç.xlsx"


def compute_consensus(row: pd.Series) -> int:
    """
    Çoğunluk oyu konsensus.
    Tam anlaşmazlık (TARTIS) → attending (Reader3) decision.
    """
    val = str(row.get('Konsensus', '')).upper().strip()
    if val in ('TARTIS', 'NAN', ''):
        return int(row['Reader3_Score'])
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return int(row['Reader3_Score'])


def main():
    df = pd.read_excel(XLSX_FILE, sheet_name='Etiketlemeler')
    df['Konsensus_Final'] = df.apply(compute_consensus, axis=1)

    tartis = df.apply(
        lambda r: str(r.get('Konsensus', '')).upper().strip() == 'TARTIS', axis=1
    ).sum()

    print("=" * 55)
    print("MAJORITY VOTE WITH EXPERT ADJUDICATION")
    print("=" * 55)
    print(f"Toplam rapor:                {len(df)}")
    print(f"Tam anlaşmazlık (TARTIS):    {tartis}")
    print(f"  → Attending kararı:        {tartis} vaka")
    print(f"\nKonsensus dağılımı:")
    print(df['Konsensus_Final'].value_counts().sort_index())

    df.to_excel(XLSX_FILE, sheet_name='Etiketlemeler', index=False)
    print(f"\n✓ Konsensus_Final sütunu eklendi: {XLSX_FILE}")


if __name__ == '__main__':
    main()
