from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import click
from tqdm import tqdm
from flask import Flask, jsonify, request

import torch
import torch.nn as nn
import numpy as np

from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig
from transformers import AdamW

from qanta.ProjectDataLoader import *


# The actual model - currently unnamed
class BERTModel(nn.Module):

    # Initialize the parameters you'll need for the model.
    def __init__(self, answer_vector_length):    
        super(BERTModel, self).__init__()
        config = BertConfig()
        self.answer_vector_length = answer_vector_length
        self.bert = BertModel(config)
        self.linear_output = nn.Linear(768, answer_vector_length)

    # computes output vector using pooled BERT output
    def forward(self, x):
        return self.linear_output(self.bert(x).pooler_output)

    # Use cosine similarity with answer tensors?
    def evaluate(self, data):
        # TODO

        with torch.no_grad():
            return 0


# This will handle the training and evaluating of the model.
# This will help abstract the process out of the web server.
class BERTAgent():

    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.loss_fn = nn.MSELoss()
        self.optimizer = AdamW(model.parameters())
        self.total_examples = 0

    # Run through a full cycle of training data
    def train_epoch(self, data_manager):
        epoch = data_manager.full_epochs
        while epoch == data_manager.full_epochs:
            inputs, labels = data_manager.get_next_batch()
            self.train_step(epoch, inputs, labels)
            
        # TODO This code is bad and needs to be fixed
        # acc_train = self.model.evaluate(train)
        # acc_test = self.model.evaluate(test)
        # print(f'Epoch: {epoch+1}/{num_epochs}, Example {self.total_examples}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f} test_acc = {acc_test.item():.4f}')
        # return False

    # Runs training step on one batch of tensors
    def train_step(self, epoch, inputs, labels):
        self.optimizer.zero_grad()

        prediction = self.model(inputs)
        loss = self.loss_fn(prediction.to(torch.float32), labels.to(torch.float32))
        loss.backward()
        self.optimizer.step()
        self.total_examples += 1

    # Computes forward on the model - I used this for debugging
    def model_forward(self, input_tensor):
        with torch.no_grad():
            return self.model(input_tensor)


if __name__ == '__main__':
    print("Model-only testing mode")
    MAX_QUESTION_LENGTH = 412
    BATCH_SIZE = 20

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    vocab = load_vocab("data/qanta.vocab")
    data = Project_BERT_Data_Manager(MAX_QUESTION_LENGTH, vocab, BATCH_SIZE, tokenizer)
    model = BERTModel(data.get_answer_vector_length())
    agent = BERTAgent(model, vocab)

    data.load_data("../data/qanta.dev.2018.04.18.json", 10)
    next_data = data.get_next()

    #print(tokenizer.convert_ids_to_tokens(next_data[0][0])
    #print(vocab.decode(next_data[1]))

    print(next_data[0].size())
    print(next_data[1].size())

    print(agent.model_forward(next_data[0]))
    agent.train_step(1, next_data[0], next_data[1])
    print(agent.model_forward(next_data[0]))
    agent.train_step(1, next_data[0], next_data[1])
    print(agent.model_forward(next_data[0]))    
    agent.train_step(1, next_data[0], next_data[1])
    print(agent.model_forward(next_data[0]))    
    agent.train_step(1, next_data[0], next_data[1])
    print(agent.model_forward(next_data[0]))    
    agent.train_step(1, next_data[0], next_data[1])
    print(agent.model_forward(next_data[0]))    
    agent.train_step(1, next_data[0], next_data[1])
    print(agent.model_forward(next_data[0]))    
    agent.train_step(1, next_data[0], next_data[1])
    print(agent.model_forward(next_data[0]))

    print(vocab.decode(agent.model_forward(next_data[0])))