# Exports tables with all POS and Level stats

import pandas as pd
from tqdm import tqdm
import numpy as np

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

from pathlib import Path
S31_DB_PATH = Path('../disambig_db/calima-msa-s31.db')
S31_DB = MorphologyDB(S31_DB_PATH, 'a')
S31_AN = Analyzer(S31_DB, 'NOAN_ALL', cache_size=100000)
bert_disambig = BERTUnfactoredDisambiguator.pretrained('msa', top=1000, pretrained_cache = False)
bert_disambig._analyzer = S31_AN

# aligned fragments data import

frag_train = pd.read_csv('../all_train_aligned.csv')
frag_dev = pd.read_csv('../all_dev_aligned.csv')
frag_test = pd.read_csv('../all_test_aligned.csv')


frag_train = frag_train[frag_train.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_dev = frag_dev[frag_dev.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_test = frag_test[frag_test.apply(lambda x: type(x['0']) == str, axis = 1)]

def sort_score(list_of_analyses):
  list_of_analyses.sort(key = lambda x: x.score, reverse = True)
  highest_score = list_of_analyses[0].score
  analyses_with_equal_score = [x for x in list_of_analyses
                                if x.score == highest_score]
  return analyses_with_equal_score

def score_select(list_of_analyses):
  list_of_analyses = sort_score(list_of_analyses)
  return list_of_analyses[0].analysis

def counts_pipeline(fragment):
  tokens = [t.split('#')[0] for t in fragment.split(' ')]
  gt_levels = [t.split('#')[1] for t in fragment.split(' ')]
  analyses = [token.analyses for token in bert_disambig.disambiguate(tokens)]
  picked_analyses = [score_select(analysis) for analysis in analyses]

  return [[t, g, a['pos']] for t, a, g in zip(tokens, picked_analyses, gt_levels)]

# All sets

all_counts = []
for x in tqdm(frag_train['0']):
  all_counts.extend(counts_pipeline(x))
for x in tqdm(frag_dev['0']):
  all_counts.extend(counts_pipeline(x))
for x in tqdm(frag_test['0']):
  all_counts.extend(counts_pipeline(x))

all_counts_df = pd.DataFrame(all_counts)
summary = all_counts_df.groupby([1, 2]).size().reset_index(name='count')
pivot_table = summary.pivot(index=2, columns=1, values='count').fillna(0)
pivot_table.to_csv('all_pos_aligned.csv')

# Only train set

all_counts_train = []
for x in tqdm(frag_train['0']):
  all_counts_train.extend(counts_pipeline(x))

all_counts_train_df = pd.DataFrame(all_counts_train)
summary_train = all_counts_train_df.groupby([1, 2]).size().reset_index(name='count')
pivot_table_train = summary_train.pivot(index=2, columns=1, values='count').fillna(0)

pivot_table_train.to_csv('all_pos_train_aligned.csv')