# From the raw Samer Corpus data, get a dataset of clean levelled fragments.

import pandas as pd
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")


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
dev_pattern = '/dev/{}/{}.{}.txt'
train_pattern = '/train/{}/{}.{}.txt'
test_pattern = '/test/{}/{}.{}.txt'

frags = 'frags'
frags_pnx = 'frags.pnx'

columns = ['l3', 'l4', 'l5']

def get_all(pattern, splits, frags = frags_pnx):
  all = pd.DataFrame(columns = columns)
  for _, row in tqdm(splits.iterrows()):
    with open(base_path + pattern.format(row['book'], row['chapter'], frags)) as f:
      all_l5 = f.read().split('\n')
    with open(base_path + pattern.format(row['book'], row['chapter'] + "_l3", frags)) as f:
      all_l3 = f.read().split('\n')
    with open(base_path + pattern.format(row['book'], row['chapter'] + "_l4", frags)) as f:
      all_l4 = f.read().split('\n')

    temp = pd.DataFrame({'l3': all_l3, 'l4': all_l4, 'l5': all_l5})

    all = pd.concat([all, temp])

  return all

# Obtains all fragments and their parallel versions

all_dev = get_all(dev_pattern, devs)
all_train = get_all(train_pattern, trains)
all_test = get_all(test_pattern, tests)

# "Collapses" the fragments, levelling them depending on the changes made in parallel versions

def collapse_and_individualise(orig):
  df = pd.DataFrame(columns =['text','l4','l3', 'level'])
  for _, row in tqdm(orig.iterrows()):
    if row['l3'] == row['l4'] and row['l4'] == row['l5']: # 333
      df = df._append({'text': row['l3'], 'l4': '\'=orig', 'l3': '\'=orig', 'level': 3}, ignore_index=True)

    elif row['l3'] == row['l4'] and row['l4'] != row['l5']: # 533
      df = df._append({'text': row['l5'], 'l4': '\'=l3', 'l3': row['l3'], 'level': 5}, ignore_index=True)
      
    elif row['l3'] != row['l4'] and row['l4'] == row['l5']: # 443
      df = df._append({'text': row['l4'], 'l4': '\'=orig', 'l3': row['l3'], 'level': 4}, ignore_index=True)
     
    elif row['l3'] != row['l4'] and row['l4'] != row['l5']: # 543
      df = df._append({'text': row['l5'], 'l4': row['l4'], 'l3': row['l3'], 'level': 5}, ignore_index=True)
      
  return df

dev_levelled = collapse_and_individualise(all_dev)
train_levelled = collapse_and_individualise(all_train)
test_levelled = collapse_and_individualise(all_test)

# Clean the set, taking punctuation only fragments out

def prune_punctuation(fragment):
  all = fragment.split(' ')
  all = [a for a in all if not re.fullmatch(r'^[–•«»!,.:\u060C\u061B\u061F\u2026]+$', a)]
  if len(all) == 0:
    return 'DELETEME'
  else:
    return ' '.join(all)

def cleanse(df):
  df['text'] = df.apply(lambda row: prune_punctuation(row['text']), axis=1)
  df = df[df['text'] != 'DELETEME']
  return df

dev_levelled = cleanse(dev_levelled)
train_levelled = cleanse(train_levelled)
test_levelled = cleanse(test_levelled)


# Clean blank leftover fragments
dev_levelled = dev_levelled[dev_levelled.apply(lambda x: x['text'] != '', axis = 1)]
train_levelled = train_levelled[train_levelled.apply(lambda x: x['text'] != '', axis = 1)]
test_levelled = test_levelled[test_levelled.apply(lambda x: x['text'] != '', axis = 1)]

base_path = '../data/'

dev_levelled.to_csv(base_path + 'dev_levelled.csv')

train_levelled.to_csv(base_path + 'train_levelled.csv')

test_levelled.to_csv(base_path + 'test_levelled.csv')

