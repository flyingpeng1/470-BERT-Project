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


class BERTTest(nn.Module):
    def __init__(self):
        """
        Initialize the parameters you'll need for the model.
        :param num_features: The number of features in the linear model
        """
        super(BERTTest, self).__init__()

        config = BertConfig()
        self.bert = BertModel(config)

    def forward(self, x):
        """
        Compute the model prediction for an example.
        :param x: Example to evaluate
        """
        return self.bert(x)


    def evaluate(self, data):
        """
        Computes the accuracy of the model. 
        """

        # No need to modify this function.
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc

def step(epoch, ex, model, optimizer, criterion, inputs, labels, vocab=[]):
    """
    :param epoch: The current epoch
    :param ex: Which example / minibatch you're one
    :param model: The model you're optimizing
    :param inputs: The current set of inputs
    :param labels: The labels for those inputs
    """


    prediction = model(inputs)
    optimizer.zero_grad()
    print(prediction.last_hidden_state.size())
    #loss = criterion(prediction, labels)
    #loss.backward()
    #optimizer.step()

    # used for selecting the most important features (again) - only tested to work with batch size 1
    #if (loss.item() < 0.5 and len(vocab) > 0 and epoch > 2):
    #    weights = model.linear.weight[0]
    #    for i in range(0, len(weights)):
    #        feature = vocab[i]
    #        abs_weight_in_prediction = abs(weights[i].item() * inputs[0][i].item())
    #        if (feature in features_dict):
    #            features_dict[feature] = (features_dict[feature][1] + abs_weight_in_prediction , features_dict[feature][1] + 1)
    #        else:
    #            features_dict[feature] = (abs_weight_in_prediction, 1)

    if (ex+1) % 100 == 0:
      acc_train = model.evaluate(train)
      acc_test = model.evaluate(test)
      print(f'Epoch: {epoch+1}/{num_epochs}, Example {ex}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f} test_acc = {acc_test.item():.4f}')


