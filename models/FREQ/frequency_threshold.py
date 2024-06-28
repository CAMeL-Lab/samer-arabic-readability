import pandas as pd
from tqdm import tqdm
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