import datasets
from transformers import pipeline
from transformers import BertTokenizer, BertModel

from models import *

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
from helpers import get_infersent_embedder
import torch.functional as F

import nltk
from infersent.models import InferSent

import time

dataset = datasets.load_dataset("snli", split="train")
biased_model =

# This is a skeleton for train_classifier: you can implement this however you want
def train_model(args):
    print(args)
    nltk.download('punkt')
    batch_size = 64
    K = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = datasets.load_dataset('snli', split='train').with_format('torch', device=device) \
        .filter(lambda ex: ex['label'] != -1)
    # train_loader = train_set.map(lambda x: print(x['hypothesis'][0]), batched=False)
    print(train_set.info.features)
    sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
    train_loader = DataLoader(train_set, sampler=sampler)

    # return
    infer_embedder = get_infersent_embedder(K)

    model = HypOnly(4096, 4096, 3).to(device)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=.001)
    num_epochs = 3
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        loss_fcn = nn.NLLLoss()
        start = time.time()
        print(train_set.dataset_size)
        for batch in (train_loader):
            # print(batch['hypothesis'])
            # print(batch['label'])
            # print(batch)
            # print(batch['hypothesis'])
            x = torch.from_numpy(infer_embedder.encode(list(map(lambda x: x[0], batch['hypothesis'])))).to(device)
            y = torch.squeeze(batch['label'])
            model.zero_grad()
            y_hat = model.forward(x)
            # print(y_hat.shape,y.shape)
            # y_hat = torch.permute(y_hat, (0, 2, 1))
            loss = loss_fcn(y_hat, y)
            # print(loss)
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        end = time.time()
        print(f'total time: {end - start} seconds')
        print("Total loss on epoch %i: %f" % (t, loss_this_epoch))
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_this_epoch,
        }, f'model_params_epoch_{t}.pt')
    model.eval()
    return model


def train_bert_model(args):
    nltk.download('punkt')
    batch_size = 64
    K = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = datasets.load_dataset('snli', split='train').with_format('torch', device=device) \
        .filter(lambda ex: ex['label'] != -1)
    # train_loader = train_set.map(lambda x: print(x['hypothesis'][0]), batched=False)
    print(train_set.info.features)
    sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
    train_loader = DataLoader(train_set, sampler=sampler)
    # return
    infer_embedder = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=.001)
    num_epochs = 3
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        loss_fcn = nn.NLLLoss()
        start = time.time()
        print(train_set.dataset_size)
        for batch in (train_loader):
            x = infer_embedder(batch['hypothesis'],padding=True, return_tensors='pt').to(device)
            y = torch.squeeze(batch['label'])
            model.zero_grad()
            y_hat = model.forward(x)
            # print(y_hat.shape,y.shape)
            # y_hat = torch.permute(y_hat, (0, 2, 1))
            loss = loss_fcn(y_hat, y)
            # print(loss)
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        end = time.time()
        print(f'total time: {end - start} seconds')
        print("Total loss on epoch %i: %f" % (t, loss_this_epoch))
        torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_this_epoch,
        }, f'model_params_epoch_{t}.pt')
    model.eval()
    return model

# eval_model()
nli_interpreter = pipeline('sequence_output_classifier', model='bert-base-uncased')

dev_set = datasets.load_dataset('snli', split='validation').with_format('numpy') \
    .filter(lambda ex: ex['label'] != -1)
print(np.unique(dev_set['label'], return_counts=True))
