import pandas as pd
from tqdm import tqdm
import re

# From the aligned word data, get all words and their corresponding levels.

# We obtain a set of fragments with levelled words in the format "word1#level word2#level ..."

splits = pd.read_csv('../data/splits/data_splits.csv',
                     dtype={
    'book': 'string',
    'chapter': 'string',
    'split': 'string'
})

devs = splits[splits['split'] == 'DEV']
trains = splits[splits['split'] == 'TRAIN']
tests = splits[splits['split'] == 'TEST']

base_path = '../data/splits'
dev_pattern = '/dev/{}/{}.frags.pnx.words.txt'
train_pattern = '/train/{}/{}.frags.pnx.words.txt'
test_pattern = '/test/{}/{}.frags.pnx.words.txt'

def get_all(pattern, splits):
  all = []
  for _, row in tqdm(splits.iterrows()):
    with open(base_path + pattern.format(row['book'], row['chapter'])) as f:
      all.extend([[' '.join(['#'.join(word.split('\t'))                       for word in frag.split('\n')]), row['book'], row['chapter']]                          for frag in f.read().split('\n\n')])
  return all

all_dev_aligned = get_all(dev_pattern, devs)
all_test_aligned = get_all(test_pattern, tests)
all_train_aligned = get_all(train_pattern, trains)

all_dev_aligned = pd.DataFrame(all_dev_aligned)
all_test_aligned = pd.DataFrame(all_test_aligned)
all_train_aligned = pd.DataFrame(all_train_aligned)

def prune_punctuation(fragment):
  all = fragment.split(' ')
  all = [a for a in all if not re.fullmatch(r'^[–•«»!,.:\u060C\u061B\u061F\u2026]+$', a.split('#')[0])]
  if len(all) == 0:
    return 'DELETEME'
  else:
    return ' '.join(all)

def cleanse(df):
  df[0] = df.apply(lambda row: prune_punctuation(row[0]), axis=1)
  df = df[df[0] != 'DELETEME']
  return df

all_dev_aligned = cleanse(all_dev_aligned)
all_train_aligned = cleanse(all_train_aligned)
all_test_aligned = cleanse(all_test_aligned)

# save

base_path = '../data/'

all_dev_aligned.to_csv(base_path + 'all_dev_aligned.csv')

all_train_aligned.to_csv(base_path + 'all_train_aligned.csv')

all_test_aligned.to_csv(base_path + 'all_test_aligned.csv')


