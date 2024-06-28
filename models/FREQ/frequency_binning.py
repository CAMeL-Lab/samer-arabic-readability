from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import pickle

cleaned_freqs = pd.read_csv('../data/freq/all_camelbert_freqs.csv')
with open('../data/levels_db/mle_max_aligned_model.pkl', 'rb') as f:
  known_levels = pickle.load(f)

total_words = sum(cleaned_freqs['1'])

def level_full_pipeline_eq_sized(n_bins, oov_level = 5):

  bin_size = total_words/n_bins
  words = {}
  bins = {i: {3: 0, 4: 0, 5: 0} for i in range(0, n_bins)}

  bin_words = []

  current_sum = 0
  current_bin = 0
  words_in_bin = 0
  for word, freq in zip(cleaned_freqs['0'], cleaned_freqs['1']):
    words[word] = current_bin
    current_sum += freq
    words_in_bin += 1
    if current_sum > bin_size:
      current_sum = current_sum % bin_size
      current_bin += 1
      bin_words.append(words_in_bin)
      words_in_bin = 0

  for word, level in known_levels.items():
    try:
      bin = words[word]
      bins[bin][level] += 1
    except:
      pass


  no_hits_bins = 0

  bins_levelled = {}
  for bin in bins.keys():
    if list(bins[bin].values()).count(0) == 3:
      bins_levelled[bin] = oov_level
      no_hits_bins += 1
    else:
      bins_levelled[bin] = max(bins[bin].items(), key = lambda x: x[1])[0]

  levelled_words = {}
  for word in words:
    levelled_words[word] = bins_levelled[words[word]]

  return levelled_words, bin_words

# Parameter sourced from the tuning notebook
model = level_full_pipeline_eq_sized(10000)[0]

with open('freq_binning_model.pkl', 'wb') as f:
  pickle.dump(model, f)

