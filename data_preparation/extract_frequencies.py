from tqdm import tqdm
import os
import pandas as pd
import re

# Obtain, across the different chunks of the dataset, the frequencies for every word

def sum_numbers_by_word(directory):
    word_sums = {}
    for filename in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                for line in file:
                      number, word = line.strip().split()
                      number = int(number)
                      word_sums[word] = word_sums.get(word,0) + number
    return word_sums

directory_path = '../data/camelbert_stats/raw_clean_sents_simple_tok_split_digits_freq/'
word_sums = sum_numbers_by_word(directory_path)

all_freqs_msa = word_sums.items()

all_freqs_df = pd.DataFrame(all_freqs_msa)

pattern = r'[\dA-Za-z]'
cleaned = all_freqs_df[all_freqs_df.apply(lambda row: not re.search(pattern, row[0]), axis =1)]
print(sum(all_freqs_df.apply(lambda row: row[0].isdigit(), axis=1)))

# Save all frequencies

cleaned.to_csv('../data/freq/all_camelbert_freqs.csv')
