import pandas as pd
from pathlib import Path

# Obtain, from the SAMER Readability Lexicon table, a quick lemma lookup dictionary

samer = pd.read_csv('../lexicon/SAMER-Readability-Lexicon.tsv',sep='\t')
samer[["lemma", "pos"]] = samer['lemma#pos'].str.split("#", expand = True)
lemmas = samer[['lemma', 'pos', 'readability (rounded average)']]

lemmas_dict = {}
def add_lemma(row):
    try:
        lemmas_dict[row['lemma']].append(
            {
                'pos': row['pos'],
                'level': row['readability (rounded average)']
            }
        )
    except:
        lemmas_dict[row['lemma']] = [
            {
                'pos': row['pos'],
                'level': row['readability (rounded average)']
            }
        ]

_ = lemmas.apply(lambda row: add_lemma(row), axis=1)

import pickle
with open('../data/lemma_db/quick_lemma_lookup.pkl', 'wb') as f:
    pickle.dump(lemmas_dict, f)