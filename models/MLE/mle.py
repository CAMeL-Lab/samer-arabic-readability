## MLE model with decisions

import numpy as np
import pandas as pd
import pickle

words_train = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/train_wordwise_clean.csv')


frag_dev = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/all_dev_aligned.csv')
frag_test = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/all_test_aligned.csv')
frag_dev = frag_dev[frag_dev.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_test = frag_test[frag_test.apply(lambda x: type(x['0']) == str, axis = 1)]

def get_mle_counts_aligned(words, levels):
  dict_levels = {}
  for word, level in zip(words, levels):
      try:
          #assume every entry of dict_levels : {3: int, 4: int, 5: int}
          dict_levels[word][level] += 1
      except:
          dict_levels[word] = {3: 0, 4: 0, 5: 0}
          dict_levels[word][level] += 1
  return dict_levels

def max_frequency_strategy(dict_levels, confidence = 0):
  print(confidence)
  dict_levels_max = {}
  no = 0
  for token in dict_levels.keys():
    cd = max(dict_levels[token].values())/sum(dict_levels[token].values())
    if cd >= confidence:
      dict_levels_max[token] = max(dict_levels[token].items(), key = lambda x: x[1])[0]
    else:
      no += 1
  print(no)
  return dict_levels_max

# get MLE model, by minimum confidence

def mle_training_pipeline_aligned(data, strategy, confidence = 0):
  counts = get_mle_counts_aligned(data['Word'], data['Label'])
  return strategy(counts, confidence = confidence)

mle_85 = mle_training_pipeline_aligned(words_train, max_frequency_strategy, confidence = 0.85)

mle_0 = mle_training_pipeline_aligned(words_train, max_frequency_strategy, confidence = 0)

def get_rl_mle_0(token, oov_level = 0):
    try:
        return mle_0[token]
    except:
        return oov_level

def get_rl_mle_85(token, oov_level = 0):
    try:
        return mle_85[token]
    except:
        return oov_level

def get_rl_0(token, oov_level = 0):
  return oov_level

# levels pipeline
def levels_pipeline(fragment, decision_1, decision_2, requires_disambig = False):
  tokens = [t.split('#')[0] for t in fragment.split(' ')]
  gt = [t.split('#')[1] for t in fragment.split(' ')]


  # decision round 1:
  levels = [decision_1(token) for token in tokens]

  # decision round 2:
  levels = [l if l != 0 else decision_2(t) for l, t in zip(levels, tokens)]

  return {
      'levels': levels,
      'gts': gt,
  }


# all results by mle 85 and 0 (base, no confidence thresholding)

mle_85_dev = [levels_pipeline(f, get_rl_mle_85, get_rl_0) for f in frag_dev['0']]
mle_85_test = [levels_pipeline(f, get_rl_mle_85, get_rl_0) for f in frag_test['0']]

mle_0_dev = [levels_pipeline(f, get_rl_mle_0, get_rl_0) for f in frag_dev['0']]
mle_0_test = [levels_pipeline(f, get_rl_mle_0, get_rl_0) for f in frag_test['0']]


# save the models
with open('mle_85_model.pkl', 'wb') as f:
    pickle.dump(mle_85, f)
with open('mle_0_model.pkl', 'wb') as f:
    pickle.dump(mle_0, f)


# save the decisions

with open('mle_85_decisions_dev.pkl', 'wb') as f:
    pickle.dump(mle_85_dev, f)
with open('mle_85_decisions_test.pkl', 'wb') as f:
    pickle.dump(mle_85_test, f)

with open('mle_0_decisions_dev.pkl', 'wb') as f:
    pickle.dump(mle_0_dev, f)
with open('mle_0_decisions_test.pkl', 'wb') as f:
    pickle.dump(mle_0_test, f)
