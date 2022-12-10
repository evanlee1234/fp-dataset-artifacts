import datasets
from transformers import pipeline
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from models import *
import evaluate

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from helpers import get_infersent_embedder
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch.functional as F
# from pynvml import *
import pandas as pd
import nltk
from infersent.models import InferSent
import os
from pathlib import Path
import time
torch.manual_seed(0)
 

def model_eval(model, dev_set, tokenizer, device):
    sampler = BatchSampler(RandomSampler(dev_set), batch_size=1000, drop_last=False)
    dev_loader = DataLoader(dev_set, sampler=sampler)
    # infer_embedder = get_infersent_embedder(K)
    i=0
    correct=0
    model.eval()
    model.zero_grad()
    with torch.no_grad():
        for batch in (dev_loader):
            y = None
            hypothesis = list(map(lambda x: x[0], batch['hypothesis']))

            # print(batch['label'])
            if type(batch['label']) is list:
                y = torch.squeeze(torch.Tensor(batch['label'])).to(device)
            else:
                y = torch.squeeze(batch['label']).to(device)
            premise = list(map(lambda x: x[0], batch['premise']))
            # print(y)
            encoding = tokenizer(premise, hypothesis, 
                                return_tensors='pt', padding=True, truncation=True).to(device)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_ids = encoding['token_type_ids'].to(device)
            y_hat = model.forward(input_ids, attention_mask, token_ids)
            y_hat = torch.argmax(y_hat, dim=1)
            del encoding
            del input_ids
            del attention_mask
            del token_ids
            correct += (y_hat == y).float().sum().to('cpu')
            print(correct)
            del y 
            del y_hat
            torch.cuda.empty_cache()
            i+=1
        
        accuracy = 100 * correct / len(dev_set)
        return accuracy

def test_model(model_folder, dataset, name):
    model_name = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    tokenizer = BertTokenizer.from_pretrained(model_name)

    
    for checkpoint in sorted(Path(model_folder).glob('*.pt')):
        model_state = torch.load(f'{checkpoint}', map_location=device)
        bert_model = BertModel.from_pretrained(model_name, num_labels=3).to(device)
        bert_model.eval()
        model = BERTNLIModel(bert_model,3).to(device)
        model.load_state_dict(model_state['model_state_dict'])
        acc = model_eval(model,dataset, tokenizer, device)    
        accuracies.append((checkpoint,acc))
    pd.DataFrame(accuracies).to_csv(f'{model_folder}_{name}.csv')


def test_hypo_models(dev_set, name):
    model_name = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []

    dev_set = dev_set.with_format('torch', device=device) \
        .filter(lambda ex: ex['label'] != -1)
    sampler = BatchSampler(RandomSampler(dev_set), batch_size=1000, drop_last=False)
    dev_loader = DataLoader(dev_set, sampler=sampler)

    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    infer_tokenizer = get_infersent_embedder(K=100000)

    model_state = torch.load('base_model_params.pt', map_location=device)['model_state_dict']
    infer_model = HypOnly(4096, 4096, 3).to(device)
    infer_model.load_state_dict(model_state)
    infer_model.eval()
    infer_model.zero_grad()

    bert_model = AutoModelForSequenceClassification.from_pretrained('../bert_train_args/checkpoint-6000-optimal'
                    , num_labels=3).to(device)
    bert_model.eval()
    bert_model.zero_grad()   
    i=0
    bert_correct=0
    infer_correct= 0
    with torch.no_grad():
        for batch in (dev_loader):
            y = None
            hypothesis = list(map(lambda x: x[0], batch['hypothesis']))
            if type(batch['label']) is list:
                y = torch.squeeze(torch.Tensor(batch['label'])).to(device)
            else:
                y = torch.squeeze(batch['label']).to(device)

            infer_x = torch.from_numpy(infer_tokenizer.encode(hypothesis)).to(device)
            infer_y_hat = torch.argmax(infer_model.forward(infer_x),dim=1)
            infer_correct += (infer_y_hat == y).float().sum()
            del infer_x
            del infer_y_hat

            encoding = bert_tokenizer(hypothesis, 
                                return_tensors='pt', padding=True, truncation=True).to(device)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_ids = encoding['token_type_ids'].to(device)
            
            bert_y_hat = torch.argmax(bert_model(input_ids, attention_mask=attention_mask, labels=y).get("logits"), dim=1)
            del encoding
            del input_ids
            del attention_mask
            del token_ids
            bert_correct += (bert_y_hat == y).float().sum().to('cpu')
            print(bert_correct)
            print(infer_correct)
            del y 
            del bert_y_hat
            torch.cuda.empty_cache()
            i+=1
        
        infer_accuracy = 100 * infer_correct / len(dev_set)
        bert_accuracy = 100 * bert_correct / len(dev_set)
    accuracies.append(('bert',bert_accuracy))
    accuracies.append(('infer',infer_accuracy))
    pd.DataFrame(accuracies).to_csv(f'hypo_results_{name}.csv')


def test_baseline(dev_set, name):
    accuracies = []

    dev_set = dev_set.filter(lambda ex: ex['label'] != -1).to_pandas()
    print(dev_set)
    accuracy = dev_set.groupby('label').count()/len(dev_set)
    print(accuracy)
    accuracy.to_csv(f'base_results_{name}.csv')


snli_dataset = datasets.load_dataset("snli", split="test")
mnli_dataset = datasets.load_dataset("multi_nli", split="validation_matched")
test_baseline(snli_dataset,'snli')
test_baseline(mnli_dataset,'mnli')
# test_hypo_models(snli_dataset,'snli')
# test_hypo_models(mnli_dataset,'mnli')
# test_model('debiased_model',mnli_dataset, 'mnli_test')
# test_model('debiased_model',snli_dataset, 'snli_test')
# test_model('biased_model',mnli_dataset, 'mnli_test')
# test_model('biased_model',snli_dataset, 'snli_test')
