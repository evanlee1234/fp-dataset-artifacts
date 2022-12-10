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

import nltk
from infersent.models import InferSent

import time

dataset = datasets.load_dataset("snli", split="train")
nltk.download('punkt')
torch.manual_seed(0)
#
# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # print_gpu_utilization()

# class CustomTrainer(Trainer):
#     def __init__(self, biased_model, *args, **kwargs):
#         # super(self,).__init__()
#         super(CustomTrainer, self).__init__(*args,**kwargs)
#         self.biased_model = biased_model

#     def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
#         """
#         Perform a training step on a batch of inputs.
#         Subclass and override to inject custom behavior.
#         Args:
#             model (`nn.Module`):
#                 The model to train.
#             inputs (`Dict[str, Union[torch.Tensor, Any]]`):
#                 The inputs and targets of the model.
#                 The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#                 argument `labels`. Check your model's documentation for all accepted arguments.
#         Return:
#             `torch.Tensor`: The tensor with training loss on this batch.
#         """
#         print(inputs)
#         return super(CustomTrainer, self).training_step(model, inputs)
#         model.train()
#         print(inputs)
#         inputs = self._prepare_inputs(inputs)

#         if is_sagemaker_mp_enabled():
#             loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
#             return loss_mb.reduce_mean().detach().to(self.args.device)

#         with self.compute_loss_context_manager():
#             loss = self.compute_loss(model, inputs)

#         if self.args.n_gpu > 1:
#             loss = loss.mean()  # mean() to average on multi-gpu parallel training

#         if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
#             # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
#             loss = loss / self.args.gradient_accumulation_steps

#         if self.do_grad_scaling:
#             self.scaler.scale(loss).backward()
#         elif self.use_apex:
#             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
#                 scaled_loss.backward()
#         elif self.deepspeed:
#             # loss gets scaled under gradient_accumulation_steps in deepspeed
#             loss = self.deepspeed.backward(loss)
#         else:
#             loss.backward()

#         return loss.detach()

    
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")

#         # biased_model
#         print(inputs['input_ids'].shape)

#         biased_output = self.biased_model(**inputs)
#         biased_logits = biased_output.get("logits")
#         # print(biased_logits.view(-1, self.model.config.num_labels))
#         # print(biased_output.loss)
#         # print(logits.view(-1, self.model.config.num_labels))
#         # raise Exception
#         # compute custom loss (suppose one has 3 labels with different weights)
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(logits + biased_logits, labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

def model_predict(model, eval_batch, tokenizer, device):
    with torch.no_grad():
        model.eval()
        batch = eval_batch
        
        hypothesis = list(map(lambda x: x[0], batch['hypothesis']))
        premise = list(map(lambda x: x[0], batch['premise']))
        encoding = tokenizer(premise,hypothesis, return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = encoding['input_ids'].to(device)
        token_ids = encoding['token_type_ids'].to(device)
        # print(input_ids)
        attention_mask = encoding['attention_mask'].to(device)
        y = torch.squeeze(batch['label']).to(device)
        outputs = model.forward(input_ids, attention_mask, token_ids)
        output = torch.argmax(outputs,dim=1)
        # print(outputs)
        print('accuracy over 100:', (y == output).sum())

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

    

def train_debiased_bert_model(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state = None
    model_state = torch.load('debiased_model/model_params_epoch_1_150000.pt', map_location=device)
    batch_size = 8
    K = 100000
    batch_checkpoint=1000
    num_epochs = 3
    epoch = 0
    batch_num = 0
    
    model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    data_set = datasets.load_dataset('snli').with_format('torch', device=device) \
        .filter(lambda ex: ex['label'] != -1)
    train_set = data_set['train']
    eval_set = data_set['validation']

    sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
    eval_sampler = BatchSampler(RandomSampler(eval_set), batch_size=100, drop_last=False)
    
    train_loader = DataLoader(train_set, sampler=sampler)
    dev_loader = DataLoader(eval_set, sampler=eval_sampler)
    dev_iter = iter(dev_loader)
    metric = evaluate.load("accuracy")

    biased_model = AutoModelForSequenceClassification.from_pretrained('../bert_train_args/checkpoint-6000-optimal'
                    , num_labels=3).to(device)
    biased_model.eval()

    bert_model = BertModel.from_pretrained(model_name, num_labels=3).to(device)
    bert_model.train()
    model = BERTNLIModel(bert_model,3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    orig_batch_num = batch_num

    
    if model_state:
        model.load_state_dict(model_state['model_state_dict'])
        optimizer.load_state_dict(model_state['optimizer_state_dict'])
        epoch = int(model_state['epoch'])
        batch_num = int(model_state['batch_num'])


    for t in range(epoch, num_epochs):
        loss_this_epoch = 0.0
        loss_this_batch_checkpoint = 0.0
        if model_state:
            loss_this_epoch = model_state['epoch_loss']
            loss_this_batch_checkpoint = model_state['loss']
        loss_fcn1 = nn.CrossEntropyLoss(reduction='none')
        loss_fcn = nn.CrossEntropyLoss()
        start = time.time()
        
        for batch in (train_loader):
            torch.cuda.empty_cache()
            model.train()   
            optimizer.zero_grad()
            y = torch.squeeze(batch['label']).to(device)

            hypothesis = list(map(lambda x: x[0], batch['hypothesis']))
            premise = list(map(lambda x: x[0], batch['premise']))

            encoding_x = tokenizer(hypothesis, return_tensors='pt', padding=True, truncation=True).to(device)
            input_ids = encoding_x['input_ids'].to(device)
            attention_mask = encoding_x['attention_mask'].to(device)
            y_s_hat = biased_model(input_ids, attention_mask=attention_mask, labels=y).get("logits")
            
            encoding = tokenizer(premise, hypothesis, 
                            return_tensors='pt', padding=True, truncation=True).to(device)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_ids = encoding['token_type_ids'].to(device)
            y_d_hat = model.forward(input_ids, attention_mask, token_ids)

            loss = loss_fcn(y_d_hat+ y_s_hat, y)
            loss.backward()
            optimizer.step()
            loss_this_batch_checkpoint += loss.item()/ batch_size
        
            if batch_num % batch_checkpoint==0 and batch_num != orig_batch_num:
                end = time.time()
                loss_this_batch_checkpoint /= batch_checkpoint
                # raise Exception
                print(f'total time: {end - start} seconds')
                try: 
                    eval_batch = next(dev_iter)
                except StopIteration:
                    dev_iter = iter(dev_loader)
                    eval_batch = next(dev_iter)
                model_predict(model, eval_batch, tokenizer, device)
                start = time.time()
                torch.save({
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_this_batch_checkpoint,
                    'epoch_loss': loss_this_epoch,
                    'batch_num': batch_num
                }, f'debiased_model/model_params_epoch_{t}_{batch_num}.pt')
                loss_this_epoch += loss_this_batch_checkpoint
                print('loss this batch: ',loss_this_batch_checkpoint)
                loss_this_batch_checkpoint = 0

            if batch_num == 200000:
                return model
            batch_num +=1
            
        print("Total loss on epoch %i: %f" % (t, loss_this_epoch))

    model.eval()
    return model
    # a_model = UnBiasedModel(biased_model,device=device).to(device)
    # trainer = CustomTrainer(
    #     biased_model,
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_set,
    #     eval_dataset=eval_set,
    #     compute_metrics=compute_metrics)
    # result = trainer.train()
    # print_summary(result)


def train_normal_bert_model(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state = None
    model_state = torch.load('biased_model/model_params_epoch_1_143000.pt', map_location=device)
    batch_size = 8
    K = 100000
    batch_checkpoint=1000
    num_epochs = 3
    epoch = 0
    batch_num = 0
    
    model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name)
    data_set = datasets.load_dataset('snli').with_format('torch', device=device) \
        .filter(lambda ex: ex['label'] != -1)
    train_set = data_set['train']
    eval_set = data_set['validation']

    sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
    eval_sampler = BatchSampler(RandomSampler(eval_set), batch_size=100, drop_last=False)
    
    train_loader = DataLoader(train_set, sampler=sampler)
    dev_loader = DataLoader(eval_set, sampler=eval_sampler)
    dev_iter = iter(dev_loader)
    metric = evaluate.load("accuracy")

    bert_model = BertModel.from_pretrained(model_name, num_labels=3).to(device)
    bert_model.train()
    model = BERTNLIModel(bert_model,3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    orig_batch_num = batch_num

    
    if model_state:
        model.load_state_dict(model_state['model_state_dict'])
        optimizer.load_state_dict(model_state['optimizer_state_dict'])
        epoch = int(model_state['epoch'])
        batch_num = int(model_state['batch_num'])


    for t in range(epoch, num_epochs):
        loss_this_epoch = 0.0
        loss_this_batch_checkpoint = 0.0
        if model_state:
            loss_this_epoch = model_state['epoch_loss']
            loss_this_batch_checkpoint = model_state['loss']
        loss_fcn1 = nn.CrossEntropyLoss(reduction='none')
        loss_fcn = nn.CrossEntropyLoss()
        start = time.time()
        
        for batch in (train_loader):
            torch.cuda.empty_cache()
            model.train()   
            optimizer.zero_grad()
            y = torch.squeeze(batch['label']).to(device)

            hypothesis = list(map(lambda x: x[0], batch['hypothesis']))
            premise = list(map(lambda x: x[0], batch['premise']))
            
            encoding = tokenizer(premise, hypothesis, 
                            return_tensors='pt', padding=True, truncation=True).to(device)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_ids = encoding['token_type_ids'].to(device)
            y_d_hat = model.forward(input_ids, attention_mask, token_ids)

            loss = loss_fcn(y_d_hat, y)
            loss.backward()
            optimizer.step()
            loss_this_batch_checkpoint += loss.item()/ batch_size
        
            if batch_num % batch_checkpoint==0 and batch_num != orig_batch_num:
                end = time.time()
                loss_this_batch_checkpoint /= batch_checkpoint
                # raise Exception
                print(f'total time: {end - start} seconds')
                try: 
                    eval_batch = next(dev_iter)
                except StopIteration:
                    dev_iter = iter(dev_loader)
                    eval_batch = next(dev_iter)
                model_predict(model, eval_batch, tokenizer, device)
                start = time.time()
                torch.save({
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_this_batch_checkpoint,
                    'epoch_loss': loss_this_epoch,
                    'batch_num': batch_num
                }, f'biased_model/model_params_epoch_{t}_{batch_num}.pt')
                loss_this_epoch += loss_this_batch_checkpoint
                print('loss this batch: ',loss_this_batch_checkpoint)
                loss_this_batch_checkpoint = 0

            if batch_num == 200000:
                return model
            batch_num +=1
            
        print("Total loss on epoch %i: %f" % (t, loss_this_epoch))

    model.eval()
    return model
    # a_model = UnBiasedModel(biased_model,device=device).to(device)
    # trainer = CustomTrainer(
    #     biased_model,
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_set,
    #     eval_dataset=eval_set,
    #     compute_metrics=compute_metrics)
    # result = trainer.train()
    # print_summary(result)

# 
# train_debiased_bert_model(None)
train_normal_bert_model(None)

# eval_model()
# nli_interpreter = pipeline('sequence_output_classifier', model='bert-base-uncased')
#
# dev_set = datasets.load_dataset('snli', split='validation').with_format('numpy') \
#     .filter(lambda ex: ex['label'] != -1)
# print(np.unique(dev_set['label'], return_counts=True))
