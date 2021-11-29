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
import math

from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig
from transformers import AdamW

from qanta.ProjectDataLoader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_LOCATION = "cache"

#=======================================================================================================
# The actual model that is managed by pytorch - still needs a name!
#=======================================================================================================
class QuizBERT(nn.Module):

    # Initialize the parameters you'll need for the model.
    def __init__(self, answer_vector_length, cache_dir=""):    
        super(QuizBERT, self).__init__()
        if (not cache_dir==""):
            self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir).to(device) #BERT-large uses too much VRAM
        else:
            print("No pretraining cache provided: falling back to fresh bert model.", flush = True)
            config = BertConfig()
            self.bert = BertModel(config).to(device)

        self.answer_vector_length = answer_vector_length
        self.linear_output = nn.Linear(768, answer_vector_length).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.last_pooler_out = None

    # computes output vector using pooled BERT output
    def forward(self, x):
        self.last_pooler_out = self.bert(x).pooler_output
        ##print(bert_out.size())
        return self.softmax(self.linear_output(self.last_pooler_out))

    # return last pooler output vector - will be used in buzztrain
    def get_last_pooler_output(self):
        if (self.last_pooler_out == None):
            raise ValueError("No pooler output cached - run through a guess first!")
        return self.last_pooler_out

    def load_bert_pretained_weights(self, cache):
        self.bert = self.bert

    def to_device(self, device):
        self.bert = self.bert.to(device)
        self.linear_output = self.linear_output.to(device)
        self = self.to(device)

    # Unimplemented
    def evaluate(self, data):
        # TODO
        with torch.no_grad():
            return 0

#=======================================================================================================
# This will handle the training and evaluating of the model.
# This will help abstract the process out of the web server.
#=======================================================================================================
class BERTAgent():

    def __init__(self, model, vocab):
        self.vocab = vocab
        self.loss_fn = nn.BCELoss()##nn.MSELoss()
        self.loss_fn = self.loss_fn.to(device)
        self.total_examples = 0
        self.checkpoint_loss = 0
        self.epoch_loss = 0
        self.saved_recently = False
        # TODO - save this data correctly!!!
        self.model = None
        self.optimizer = None

        # waiting to create the optimizer until a model is loaded
        if (not model == None):
            self.model = model
            self.model.to_device(device)
            self.optimizer = AdamW(model.parameters())
        else:
            print("Agent is waiting for model load!", flush = True)

    # Save model and its associated metadata 
    def save_model(self, metadata, save_location):
        pickle.dump({"model": self.model, "metadata":metadata}, open(save_location, "wb+"))
        print("Saved model to: \"" + save_location + "\"", flush = True)

    # Load the model and its associated metadata
    def load_model(self, file_name, data_manager=None):
        load = pickle.load(open(file_name,'rb'))
        self.model = load["model"]
        self.model.to_device(device)
        if ("metadata" in load and "epoch" in load["metadata"] and not data_manager == None):
            data_manager.full_epochs = load["metadata"]["epoch"] + 1
            if ("completed" in load["metadata"] and not load["metadata"]["completed"]):
                print("Skipped incomplete epoch: preparing next epoch")
            elif(not "completed" in load["metadata"]):
                print("Could not find completion data in model - epoch may have been skipped")
        else:
            print("No metadata found / no data manager provided - starting from epoch 0")

        self.optimizer = AdamW(self.model.parameters())
        print("Loaded model from: \"" + file_name + "\"", flush = True)

    # Run through a full cycle of training data - save freq and save_loc will determine whether the model is saved after the epoch is finished
    # save_freq
    def train_epoch(self, data_manager, save_freq, save_loc):
        epoch = data_manager.full_epochs
        self.epoch_loss = 0
        print("Starting train epoch #" + str(epoch), flush = True)
        while epoch == data_manager.full_epochs:
            inputs, labels = data_manager.get_next_batch()
            self.train_step(epoch, inputs.to(device), labels.to(device))
            torch.cuda.empty_cache() # clear old tensors from VRAM

            if (int(data_manager.batch % 100) == 0):
                print("Epoch " + str(epoch) + " progress: " + str(data_manager.get_epoch_completion()) + "%")

            # saves mid-epoch at supplied interval
            wants_to_save = int(data_manager.get_epoch_completion()) % (save_freq) == 0 and data_manager.get_epoch_completion() > 1 and not (save_freq >= 100)
            if (wants_to_save and not self.saved_recently):
                self.save_model({"epoch":epoch, "completed":False}, save_loc + "/Model_epoch_" + str(epoch) + "_progress_" + str(int(data_manager.get_epoch_completion())) + "%.model")
                self.saved_recently=True
            elif(not wants_to_save and self.saved_recently):
                self.saved_recently=False

        print('epoch average loss: %.5f' % (self.epoch_loss / (self.total_examples+1 / (epoch+1))), flush = True)

        # saves every epoch if allowed
        if (not save_freq > 100):
            self.save_model({"epoch":epoch, "completed":True}, save_loc + "/Model_epoch_" + str(epoch) + ".model")

    # Runs training step on one batch of tensors
    def train_step(self, epoch, inputs, labels):
        self.optimizer.zero_grad()

        prediction = self.model(inputs)
        
        loss = self.loss_fn(prediction.to(torch.float32).to(device), labels.to(torch.float32).to(device))
        loss.backward()
        self.optimizer.step()
        self.total_examples += 1

        loss_val = loss.data.cpu().numpy()

        if (not np.isnan(loss_val)):
            self.checkpoint_loss += loss_val
            self.epoch_loss += loss_val

        checkpoint = 500
        if self.total_examples % checkpoint == 0 and self.total_examples > 0:
            loss_avg = self.checkpoint_loss / checkpoint
            if (loss_avg == 0):
                print("NO LOSS - something is probably wrong!")
            else:
                print("num exs: " + str(self.total_examples) + ", log loss: " + str(math.log(loss_avg)), flush = True)
            self.checkpoint_loss = 0

    # used to determine whether or not the model is doing autograd and such when running forward
    def model_set_mode(self, mode):
        if (mode == "eval"):
            self.model.eval()
        elif (mode == "train"):
            self.model.train()
        else:
            raise ValueError("No model mode \"" + mode + "\" exists", flush = True)

    # Computes forward on the model - I used this for debugging
    def model_forward(self, input_tensor):
        with torch.no_grad():
            return self.model(input_tensor)


if __name__ == '__main__':
    print("Model-only testing mode")
    MAX_QUESTION_LENGTH = 412
    BATCH_SIZE = 1

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab = load_vocab("../data/QuizBERT.vocab")
    data = Project_BERT_Data_Manager(MAX_QUESTION_LENGTH, vocab, BATCH_SIZE, tokenizer)
    model = QuizBERT(data.get_answer_vector_length(), CACHE_LOCATION)
    model.to_device(device)
    agent = BERTAgent(model, vocab)

    data.load_data("../data/qanta.dev.2018.04.18.json", 100)
    next_data = data.get_next()

    #print(tokenizer.convert_ids_to_tokens(next_data[0][0])
    #print(vocab.decode(next_data[1]))

    #print(agent.model_forward(next_data[0].to(device)))

    '''print(next_data[0].size())
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

    print(vocab.decode(agent.model_forward(next_data[0])))'''

    agent.train_epoch(data, 1000, "training_progress")
    #agent.save_model({}, "training_progress/test_model.model")

    #agent.load_model("training_progress/test_model.model")
    #agent.train_epoch(data, 50, "training_progress")