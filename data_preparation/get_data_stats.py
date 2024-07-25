# Get general word and fragment level stats from splits.

import pandas as pd

base_path = '../data/'

all_dev_aligned = pd.read_csv(base_path + 'all_dev_aligned.csv')

all_train_aligned = pd.read_csv(base_path + 'all_train_aligned.csv')

all_test_aligned = pd.read_csv(base_path + 'all_test_aligned.csv')

# Delete empty fragments

all_train_aligned = all_train_aligned[all_train_aligned.apply(lambda x: type(x['0']) == str, axis = 1)]
all_dev_aligned = all_dev_aligned[all_dev_aligned.apply(lambda x: type(x['0']) == str, axis = 1)]
all_test_aligned = all_test_aligned[all_test_aligned.apply(lambda x: type(x['0']) == str, axis = 1)]

# Level of frag = max(level of words)

def get_level(frag):
  levels = [word.split('#')[1] for word in frag.split(" ")]
  return max(levels)

all_dev_aligned['level'] = all_dev_aligned.apply(lambda row: get_level(row['0']), axis = 1)
all_train_aligned['level'] = all_train_aligned.apply(lambda row: get_level(row['0']), axis = 1)
all_test_aligned['level'] = all_test_aligned.apply(lambda row: get_level(row['0']), axis = 1)

# Count fragments and levels

dev_counts = dict(all_dev_aligned['level'].value_counts())
train_counts = dict(all_train_aligned['level'].value_counts())
test_counts = dict(all_test_aligned['level'].value_counts())

dev_counts['split'] = 'dev'
train_counts['split'] = 'train'
test_counts['split'] = 'test'

fragment_counts = pd.DataFrame([ train_counts, dev_counts, test_counts])

fragment_counts['sum'] = fragment_counts.apply(lambda row: row['3'] + row['4'] + row['5'], axis=1) # Calculate row-wise sum of columns 3, 4, and 5
sum_row = fragment_counts.sum()
sum_df = pd.DataFrame(sum_row).transpose()
sum_df['split'] = 'Total'
fragment_counts = pd.concat([fragment_counts, sum_df], ignore_index=True)

# Save fragment-wise data
fragment_counts.to_csv(base_path + 'fragment_counts.csv')


# Count words and levels 
def count_words(frag_list):
  counts = {}
  for frag in frag_list:
    levels = [word.split('#')[1] for word in frag.split(" ")]
    for level in levels:
      counts[level] = counts.get(level, 0) + 1

  return counts

train_wds = dict(count_words(all_train_aligned['0']))
test_wds = dict(count_words(all_test_aligned['0']))
dev_wds = dict(count_words(all_dev_aligned['0']))

dev_wds['split'] = 'dev'
train_wds['split'] = 'train'
test_wds['split'] = 'test'

wd_counts = pd.DataFrame([ train_wds, dev_wds, test_wds])

wd_counts['sum'] = wd_counts.apply(lambda row: row['3'] + row['4'] + row['5'], axis=1) # Calculate row-wise sum of columns 3, 4, and 5
sum_row = wd_counts.sum()
sum_df = pd.DataFrame(sum_row).transpose()
sum_df['split'] = 'Total'
wd_counts = pd.concat([wd_counts, sum_df], ignore_index=True)

wd_counts.to_csv('wd_counts.csv')

