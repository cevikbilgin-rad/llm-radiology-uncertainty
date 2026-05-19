"""
02_sampling.py
==============
Validation seti için stratified random sampling.
Her hedge frekans katmanından 25 rapor seç (toplam 100).

Kullanım:
    python 02_sampling.py
"""

import pandas as pd
import random

RANDOM_SEED   = 42
N_PER_STRATUM = 25


def main():
    df = pd.read_excel('FINAL_3927_reports_with_claude.xlsx')
    print(f"Toplam rapor: {len(df)}")
    print(f"Lexicon score dağılımı:\n{df['lexicon_score'].value_counts().sort_index()}")

    random.seed(RANDOM_SEED)

    sample_0 = df[df['lexicon_total'] == 0].sample(N_PER_STRATUM, random_state=RANDOM_SEED)
    sample_1 = df[df['lexicon_total'] == 1].sample(N_PER_STRATUM, random_state=RANDOM_SEED)
    sample_2 = df[df['lexicon_total'] == 2].sample(N_PER_STRATUM, random_state=RANDOM_SEED)
    sample_3 = df[df['lexicon_total'] >= 3].sample(N_PER_STRATUM, random_state=RANDOM_SEED)

    validation_set = pd.concat([sample_0, sample_1, sample_2, sample_3]).reset_index(drop=True)

    print(f"\n✓ Validation set: {len(validation_set)} rapor")
    print(f"  0 hedge: {len(sample_0)}")
    print(f"  1 hedge: {len(sample_1)}")
    print(f"  2 hedge: {len(sample_2)}")
    print(f"  ≥3 hedge: {len(sample_3)}")

    validation_set[['id', 'findings', 'impression', 'mesh_terms',
                    'lexicon_total', 'lexicon_score']].to_excel(
        'validation_set_100.xlsx', index=False)
    print("✓ Kaydedildi: validation_set_100.xlsx")


if __name__ == '__main__':
    main()
