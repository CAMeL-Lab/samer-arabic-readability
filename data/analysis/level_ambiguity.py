# Use the aligned data, and a counting technique to find out the cases 
# in which the same written token corresponds to two different levels

import pandas as pd
import os



base_path_aligned = '../readability_data'

train_aligned = pd.read_csv(base_path_aligned + '/train_pnx.tsv', sep = '\t')

def get_level_counts_aligned(words, levels):
  dict_levels = {}
  for word, level in zip(words, levels):
      try:
          #assume every entry of dict_levels : {3: int, 4: int, 5: int}
          dict_levels[word][level] += 1
      except:
          dict_levels[word] = {3: 0, 4: 0, 5: 0}
          dict_levels[word][level] += 1
  return dict_levels

def counting_pipeline_aligned(data):
  counts = get_level_counts_aligned(data['Word'], data['Label'])
  return counts

counts = counting_pipeline_aligned(train_aligned)

duplicates = []
threes = []
fours = []
fives = []

for x in counts.items():
  # only 5, 4, or 3. single out those with ambiguous leveling
  if x[1][3] != 0 and x[1][4] == 0 and x[1][5] == 0:
    threes.append(x)
  elif x[1][3] == 0 and x[1][4] != 0 and x[1][5] == 0:
    fours.append(x)
  elif x[1][3] == 0 and x[1][4] == 0 and x[1][5] != 0:
    fives.append(x)
  else:
    # more than one level
    duplicates.append(x)

_5and4 = [d for d in duplicates if d[1][5] != 0 and d[1][4] !=0 and d[1][3] == 0]
_5and3 = [d for d in duplicates if d[1][5] != 0 and d[1][4] ==0 and d[1][3] != 0]
_4and3 = [d for d in duplicates if d[1][5] == 0 and d[1][4] !=0 and d[1][3] != 0]
_all = [d for d in duplicates if d[1][5] != 0 and d[1][4] !=0 and d[1][3] != 0]

results = [
   {
      'Type': 'L3 only', 'Count': threes
   },
   {
      'Type': 'L4 only', 'Count': fours
   },
   {
      'Type': 'L5 only', 'Count': fives
   },
   {
      'Type': 'L5 and L4', 'Count': _5and4
   },
   {
      'Type': 'L5 and L3', 'Count': _5and3
   },
   {
      'Type': 'L4 and L3', 'Count': _4and3
   },
   {
      'Type': 'Tagged as', 'Count': _5and4
   }
]

df_results = pd.DataFrame(results, columns= ['Type', 'Count'])

df_results.to_csv('level_ambiguity.csv')

