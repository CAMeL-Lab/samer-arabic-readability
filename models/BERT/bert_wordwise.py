## Objective:
## Fine tune BERT on word-wise labels (3,4,5).
## Then, prepare an architecture for NER, as in BertForTokenClassification.
## (https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1691)

import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import classification_report
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=True, mode='min')
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import numpy as np

# Subword tokenization and labelling

frag_train = pd.read_csv('../data/all_train_aligned.csv')
frag_dev = pd.read_csv('../data/all_dev_aligned.csv')
frag_test = pd.read_csv('../data/all_test_aligned.csv')


frag_train = frag_train[frag_train.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_dev = frag_dev[frag_dev.apply(lambda x: type(x['0']) == str, axis = 1)]
frag_test = frag_test[frag_test.apply(lambda x: type(x['0']) == str, axis = 1)]

model_checkpoint = 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-msa')

# Tag all subwords with the label of the word

def tokenize_step1(data):
  words = []
  tags = []
  for frag in data:
    splits = [w.split('#') for w in frag.split(' ')]

    words.append([f[0] for f in splits])
    tags.append([int(f[1]) - 3 for f in splits])

  return words, tags

def tokenize_unmarked_whole(data):
  words = [frag.split(' ') for frag in data]
  tokenized_inputs = tokenizer(words, max_length = 30, pad_to_max_length=True, truncation=True, is_split_into_words=True)

  return tokenized_inputs


def align_labels(labels, word_ids):
  aligned_labels = []
  for id in word_ids:
    if id is None:
      aligned_labels.append(0)
    else:
      aligned_labels.append(labels[id])

  return aligned_labels

def tokenize_and_align_labels(fragments, labels_lists):
  tokenized_inputs = tokenizer(fragments, max_length = 30, pad_to_max_length=True, truncation=True, is_split_into_words=True)
  new_labels = []
  for i, labels in enumerate(labels_lists):
    word_ids = tokenized_inputs.word_ids(i)
    new_labels.append(align_labels(labels, word_ids))

  tokenized_inputs["labels"] = new_labels
  return tokenized_inputs

def get_all_dataloaders(train, test, dev):
  batch_size = 32

  train_words_step1, train_labels_step1 = tokenize_step1(train)
  test_words_step1, test_labels_step1 = tokenize_step1(test)
  dev_words_step1, dev_labels_step1 = tokenize_step1(dev)

  train_ds = tokenize_and_align_labels(train_words_step1, train_labels_step1)
  test_ds = tokenize_and_align_labels(test_words_step1, test_labels_step1)
  dev_ds = tokenize_and_align_labels(dev_words_step1, dev_labels_step1)

  all_ids = [test_ds.word_ids(i) for i in range(len(test))]

  train_seq = torch.tensor(train_ds['input_ids'])
  train_mask = torch.tensor(train_ds['attention_mask'])
  train_y = torch.tensor(train_ds['labels'])

  dev_seq = torch.tensor(dev_ds['input_ids'])
  dev_mask = torch.tensor(dev_ds['attention_mask'])
  dev_y = torch.tensor(dev_ds['labels'])

  test_seq = torch.tensor(test_ds['input_ids'])
  test_mask = torch.tensor(test_ds['attention_mask'])
  test_y = torch.tensor(test_ds['labels'])

  train_data = TensorDataset(train_seq, train_mask, train_y)
  dev_data = TensorDataset(dev_seq, dev_mask, dev_y)
  test_data = TensorDataset(test_seq, test_mask, test_y)

  class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(np.concatenate(train_labels_step1)), y = np.concatenate(train_labels_step1))
  weights = torch.tensor(class_weights,dtype=torch.float)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, num_workers=2)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

  return train_loader, dev_loader, test_loader, weights, all_ids

# Model architecture

class BertForTokenClassification(pl.LightningModule):
  def __init__(self, bert, lr = 1e-6, weighted = False, weights = []):
    super(BertForTokenClassification, self).__init__()
    self.bert = bert
    self.tok_classifier = nn.Linear(768, 3)

    if weighted:
      self.lossFn = nn.CrossEntropyLoss(weight = weights)
    else:
      self.lossFn = nn.CrossEntropyLoss()
    self.lr = lr

    self.all_pred = []
    self.all_gt = []
    self.all_train_loss = []
    self.all_dev_loss = []
    self.initialize_weights()

  def forward(self, tokens, mask):
    bert_output = self.bert(tokens, attention_mask = mask)
    sequence_output = bert_output[0]
    logits = self.tok_classifier(sequence_output)

    return logits

  def loss(self, logits, labels):
    return self.lossFn(logits.view(-1, 3), labels.view(-1))

  def training_step(self, train, i):
    x, mask, y = train
    probs = self.forward(x, mask)
    loss = self.loss(probs, y)
    self.log('train_loss', loss)
    self.all_train_loss.append(loss)
    return loss

  def validation_step(self, val, i):
    x, mask, y = val
    probs = self.forward(x, mask)
    loss = self.loss(probs, y)
    self.log('val_loss', loss)
    self.all_dev_loss.append(loss)
    return loss

  def test_step(self, test, i):
    x, mask, y = test
    probs = self(x, mask)

    self.all_pred.append(probs)
    self.all_gt.append(y)

    loss = self.loss(probs, y)
    self.log('test_loss', loss)

  def predict_step(self, predict, i):
    x, mask = predict
    probs = self.forward(x, mask)
    return probs


  def initialize_weights(self):
    nn.init.xavier_uniform_(self.tok_classifier.weight)
    nn.init.zeros_(self.tok_classifier.bias)


  def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
      scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
      return optimizer

# Training protocol

def train(model, train_loader, dev_loader, trainer):
  trainer.fit(model, train_loader, dev_loader)

def benchmark(model, test_loader, trainer, freeze = False, weighted = False):
  results = trainer.test(model, test_loader)
  all_pred_test = np.concatenate([x.detach().cpu() for x in model.all_pred])
  all_gt_test = np.concatenate([x.detach().cpu() for x in model.all_gt])
  pred_labels = np.argmax(all_pred_test, axis=2)

  all_labs = []
  all_gts = []

  for labs, gts in zip(pred_labels, all_gt_test):
    all_labs.extend([l for l, g in zip(labs, gts) if g != -100])
    all_gts.extend([g for l, g in zip(labs, gts) if g != -100])


  return {
      'pred': [x.detach().cpu() for x in model.all_pred],
      'apt': all_pred_test,
      'agt': all_gt_test,
      'labels': all_labs,
      'gt': all_gts,
      'report': classification_report(all_gts, all_labs, output_dict = True)
  }

def run_experiment(model, train_set, test_set, dev_set, freeze = False, weighted = False, n = 0):
  print('Importing model and tokenizer...')
  bert_model = AutoModel.from_pretrained(model)
  device = torch.device("cuda")
  bert_model = bert_model.to(device)

  if freeze:
    for param in bert_model.parameters():
      param.requires_grad = False



  print('Setting up data...')
  train_dl, dev_dl, test_dl, weights, test_word_ids = get_all_dataloaders(train_set, test_set, dev_set)
  early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=True, mode='min')

  print('Setting up architecture...')

  arch = BertForTokenClassification(bert_model, 5e-5, weighted = weighted, weights = weights)
  trainer = pl.Trainer(callbacks=[early_stopping],accelerator="gpu", max_epochs = 10)

  print('Training start')
  train(arch, train_dl, dev_dl, trainer)
  return benchmark(arch, test_dl, trainer), arch, test_word_ids

# Experiment

res_dev, model, dev_ids = run_experiment(model_checkpoint, frag_train['0'], frag_dev['0'], frag_dev['0'], freeze = False, weighted = False, n=1)

res_test, _, test_ids = run_experiment(model_checkpoint, frag_train['0'], frag_test['0'], frag_dev['0'], freeze = False, weighted = False, n=1)

def get_wordwise_results(results, ids_set):
    frags = []
    for subwords, ids in zip(np.argmax(results['apt'], axis=2), ids_set):
        w = [0 for i in range(max([i for i in ids if not(i is None)])+1)]
        for i, x in enumerate(subwords):
            corresponding_word = ids[i]
            if not (ids[i] is None):
                w[corresponding_word] = max(w[corresponding_word], x)
        frags.append(w)
    return frags
    

test_wordwise_decisions = get_wordwise_results(res_test, test_ids)

dev_wordwise_decisions = get_wordwise_results(res_dev, dev_ids)

### save decisions
import pickle
with open('pickled_results/test_wordwise_decisions.pkl', 'wb') as f:
  pickle.dump(test_wordwise_decisions, f)

with open('pickled_results/dev_wordwise_decisions.pkl', 'wb') as f:
  pickle.dump(dev_wordwise_decisions, f)


