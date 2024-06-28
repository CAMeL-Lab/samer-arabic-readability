import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle


# According to the level thresholds found in the training data, we label all words according to their frequency, sorted descending.

cleaned = pd.read_csv('../../data/freq/all_camelbert_freqs.csv')
big_sum = sum(cleaned[1])

l3_threshold = 0.8652
l4_threshold = 0.0900
l5_threshold = 0.0448

words_and_levels = {}

cleaned = cleaned.sort_values(1, ascending = False)

sum = 0

for row in tqdm(cleaned.iterrows()):
  sum += int(row[1][1])
  if sum > big_sum * (l3_threshold + l4_threshold):
    words_and_levels[row[1][0]] = 5
  elif sum > big_sum * l3_threshold:
    words_and_levels[row[1][0]] = 4
  else:
    words_and_levels[row[1][0]] = 3

# Pickle the model

with open('freq_threshold_model.pkl', 'wb') as f:
  pickle.dump(words_and_levels, f)

# Get Dev and Test sets
frag_dev = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/all_dev_aligned.csv')
frag_test = pd.read_csv('/content/drive/My Drive/capstone_data/bashar_data/capstone_data/splits/all_test_aligned.csv')

frag_dev = frag_dev[frag_dev.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_test = frag_test[frag_test.apply(lambda x: type(x['0']) == str, axis = 1)]


# Generate decisions for DEV and TEST sets

def get_rl_0(token, oov_level = 0):
  return oov_level

def levels_pipeline(fragment, decision_1, decision_2):
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

def get_rl_freq(token, oov_level = 0):
    try:
        return words_and_levels[token]
    except:
        return oov_level
    
## Get lexicon decisions per fragment on the DEV and TEST sets
  
freq_threshold_dev = [levels_pipeline(f, get_rl_freq, get_rl_0) for f in frag_dev['0']]
freq_threshold_test = [levels_pipeline(f, get_rl_freq, get_rl_0) for f in frag_test['0']]

## Concatenate, have an array of wordwise decisions.

freq_threshold_decisions_dev = np.concatenate([e['levels'] for e in freq_threshold_dev])
freq_threshold_decisions_test = np.concatenate([e['levels'] for e in freq_threshold_test])

## Save
import pickle

with open('freq_threshold_decisions_dev.pkl', 'wb') as f:
    pickle.dump(freq_threshold_decisions_dev, f)
with open('freq_threshold_decisions_test.pkl', 'wb') as f:
    pickle.dump(freq_threshold_decisions_test, f)