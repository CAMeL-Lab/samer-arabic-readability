#!/bin/bash
python BERT/bert_wordwise.py
python FREQ/frequency_binning.py
python FREQ/frequency_threshold.py
python FREQ/frequency_binning.py
python FREQ/frequency_binning.py
python LEX/lexicon.py
python MLE/mle.py
