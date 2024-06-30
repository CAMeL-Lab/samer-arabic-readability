# Benchmark the Lexicon model (best version as described in the tuning notebook)
import numpy as np
import pandas as pd
import Levenshtein
from pathlib import Path


# setup analyzer and disambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

S31_DB_PATH = Path('/content/drive/My Drive/capstone_data/disambig_db/calima-msa-s31.db')
S31_DB = MorphologyDB(S31_DB_PATH, 'a')
S31_AN = Analyzer(S31_DB, 'NOAN_ALL', cache_size=100000)
bert_disambig = BERTUnfactoredDisambiguator.pretrained('msa', top=1000, pretrained_cache = False)
bert_disambig._analyzer = S31_AN

## train data


words_train = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/train_wordwise_clean.csv')
words_dev = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/dev_wordwise_clean.csv')
words_test = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/test_wordwise_clean.csv')

## testing data in fragments
frag_train = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/all_train_aligned.csv')
frag_dev = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/all_dev_aligned.csv')
frag_test = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/all_test_aligned.csv')

frag_train = frag_train[frag_train.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_dev = frag_dev[frag_dev.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_test = frag_test[frag_test.apply(lambda x: type(x['0']) == str, axis = 1)]

## get ground truth

def get_level_of_frag(fragment):
  gt = [int(t.split('#')[1]) for t in fragment.split(' ')]
  return max(gt)

frag_train['level'] = frag_train.apply(lambda row: get_level_of_frag(row['0']), axis=1)
frag_dev['level'] = frag_dev.apply(lambda row: get_level_of_frag(row['0']), axis=1)
frag_test['level'] = frag_test.apply(lambda row: get_level_of_frag(row['0']), axis=1)

# configure lexicon
import pickle
with open('/content/drive/My Drive/capstone_data/disambig_db/quick_lemma_lookup.pkl', 'rb') as f:
    lemma_db = pickle.load(f)

# default UNK level
def get_rl_0(token, analyses, oov_level = 0):
  return oov_level

# sort analyses from disambiguator

def get_rl_single(analysis, oov_level = 0):
  lex = analysis['lex']
  pos = analysis['pos']
  if pos == 'noun_prop':
    return 3
  result = lemma_db.get(lex)
  if result:
    if len(result) == 1:
      rl = result[0]['level']
    else:
      most_similar_element = None
      max_similarity = -1

      for element in result:
          ### Get closest POS match
          similarity = Levenshtein.ratio(pos, element['pos'])
          if similarity > max_similarity:
              max_similarity = similarity
              most_similar_element = element

      rl = most_similar_element['level']
    return rl
  else:
    return oov_level

def sort_score(list_of_analyses):
  list_of_analyses.sort(key = lambda x: x.score, reverse = True)
  highest_score = list_of_analyses[0].score
  analyses_with_equal_score = [x for x in list_of_analyses
                                if x.score == highest_score]
  return analyses_with_equal_score

def sort_lexlogprob(list_of_analyses):
  list_of_analyses.sort(key = lambda x: x.analysis['lex_logprob'], reverse=True)
  highest_prob = list_of_analyses[0].analysis['lex_logprob']
  analyses_with_equal_prob = [x for x in list_of_analyses
                                if x.analysis['lex_logprob'] == highest_prob]
  return analyses_with_equal_prob

def default_level_oov(word, level, analysis):
  return level

def sort_level(list_of_analyses):
  list_of_analyses.sort(key = lambda x: get_rl_single(x.analysis, 9999))
  lowest_rl = get_rl_single(list_of_analyses[0].analysis, 9999)
  analyses_with_equal_level = [x for x in list_of_analyses
                                    if get_rl_single(x.analysis, 9999) == lowest_rl]
  return analyses_with_equal_level

def score_then_level_then_llp(list_of_analyses):
  list_of_analyses = sort_score(list_of_analyses)
  if len(list_of_analyses) == 1:
    return list_of_analyses[0].analysis

  list_of_analyses = sort_level(list_of_analyses)
  if len(list_of_analyses) == 1:
    return list_of_analyses[0].analysis

  list_of_analyses = sort_lexlogprob(list_of_analyses)
  return list_of_analyses[0].analysis

def get_rl_lexicon(token, analyses, oov_level = 0):
  analysis = score_then_level_then_llp(analyses)

  return get_rl_single(analysis)

## The Lexicon does not need training. We only save the results of the eval and test sets.
def levels_pipeline(fragment, decision_1, decision_2, requires_disambig = False):
  tokens = [t.split('#')[0] for t in fragment.split(' ')]
  gt = [t.split('#')[1] for t in fragment.split(' ')]

  if requires_disambig:
    analyses = [token.analyses for token in bert_disambig.disambiguate(tokens)]
  else:
    analyses = gt

  # decision round 1:
  levels = [decision_1(token, analysis) for token, analysis in zip(tokens, analyses)]

  # decision round 2:
  levels = [l if l != 0 else decision_2(t, a) for l, t, a in zip(levels, tokens, analyses)]

  return {
      'levels': levels,
      'gts': gt,
  }

## Function to round levels 1-2 to 3.
def level_keep0(l):
  if l > 0:
    if l < 3:
      return 3
    else:
      return l
  else:
    return 0

## Get lexicon decisions per fragment on the DEV and TEST sets

lexicon_dev = [levels_pipeline(f, get_rl_lexicon, get_rl_0, requires_disambig = True) for f in frag_dev['0']]
lexicon_test = [levels_pipeline(f, get_rl_lexicon, get_rl_0, requires_disambig = True) for f in frag_test['0']]

## Save
import pickle

with open('lexicon_decisions_dev.pkl', 'wb') as f:
    pickle.dump(lexicon_dev, f)
with open('lexicon_decisions_test.pkl', 'wb') as f:
    pickle.dump(lexicon_test, f)
