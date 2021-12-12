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
import json

from transformers import BertTokenizer
from transformers import BertModel
#from transformers import BertForSequenceClassification
from transformers import BertConfig
#from transformers import AdamW

#from warp_loss import warp_loss
import random
import time

from qanta.ProjectDataLoader import *
from qanta.warp_loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda_bool = device == "cuda" 

CACHE_LOCATION = "cache" # transformers cache
BERT_OUTPUT_LENGTH = 768 # constant for version of BERT - 768 is # of outputs
HIDDEN_UNITS = 768 # number of hidden neurons
DROPOUT = .2 # dropout between linear embedding layers
EMB_DIM=1000 # dim of final embedding vectors

#=======================================================================================================
# The actual model that is managed by pytorch - QuizBERT, excelsior!!!
#=======================================================================================================
class QuizBERT(nn.Module):

    # Initialize the parameters we'll need for the model.
    def __init__(self, answer_vector_length, cache_dir=""):    
        super(QuizBERT, self).__init__()

        self.answer_vector_length = answer_vector_length

        self.answers_embeds = nn.Embedding(answer_vector_length, EMB_DIM).to(device)

        self.drop = nn.Dropout(p=DROPOUT).to(device)
        self.question_embeds = nn.Linear(BERT_OUTPUT_LENGTH, EMB_DIM).to(device)#nn.Embedding(BERT_OUTPUT_LENGTH, EMB_DIM).to(device)
        self.question_hidden = nn.Linear(BERT_OUTPUT_LENGTH, HIDDEN_UNITS).to(device)

        if (not cache_dir==""):
            self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir).to(device) #BERT-large uses too much VRAM
        else:
            print("No pretraining cache provided: falling back to fresh bert model.", flush = True)
            config = BertConfig()
            self.bert = BertModel(config).to(device)

        n=12
        modules = [self.bert.embeddings, self.bert.encoder.layer[:n]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    # Used exclusively for training
    def forward(self, question, pos, neg, expect_precalculated_pool=False):

        # allows to precalculate the BERT output for faster training
        if (expect_precalculated_pool):
            self.last_pooler_out = question
        else:
            self.last_pooler_out = self.bert(question).pooler_output
        
        batch_size = neg.size(0)
        question_vector = self.question_embeds(self.drop(self.question_hidden(self.last_pooler_out.to(device))))
        repeated_question_vector = question_vector.repeat((batch_size, 1)).view(batch_size, -1, 1)

        pos_res = torch.bmm(self.answers_embeds(torch.unsqueeze(pos, 0)), repeated_question_vector).squeeze(2)
        neg_res = torch.bmm(self.answers_embeds(neg), repeated_question_vector).squeeze(2)

        return pos_res, neg_res

    # embedds answer
    def embed_answer(self, answer):
        return self.answers_embeds(answer)

    # embeds question
    def embed_question(self, question, expect_precalculated_pool=False):
        if (expect_precalculated_pool):
            return self.question_embeds(self.drop(self.question_hidden(question)))
        else:
            bert = self.bert(question).pooler_output
            return self.question_embeds(self.drop(self.question_hidden(bert)))

    # called to select how many layers of BERT are frozen at any given time
    # This was useful for our dynamic freezing approach to training
    def unfreeze_layers(self, layers):
        modules = [self.bert.embeddings, *self.bert.encoder.layer]
        
        # freeze everything
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        # selectively unfreeze
        unfreeze = []
        print(len(modules))
        for l in layers:
            unfreeze.append(modules[int(l)])

        for module in unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    # return last pooler output vector - can be used in buzztrain
    def get_last_pooler_output(self):
        if (self.last_pooler_out == None):
            raise ValueError("No pooler output cached - run through a guess first!")
        return self.last_pooler_out

    def load_bert_pretained_weights(self, cache):
        self.bert = self.bert

    # extension of pytorch .to() function
    def to_device(self, device):
        self.bert = self.bert.to(device)
        self.answers_embeds = self.answers_embeds.to(device)
        self.drop = self.drop.to(device)
        self.question_hidden = self.question_hidden.to(device)
        self.question_embeds = self.question_embeds.to(device)
        self = self.to(device)

#=======================================================================================================
# This will handle the training and evaluating of the model.
# This will help abstract the process out of the web server.
#=======================================================================================================
class BERTAgent():

    def __init__(self, model, vocab):
        self.vocab = vocab
        self.total_examples = 0
        self.checkpoint_loss = 0
        self.epoch_loss = 0
        self.saved_recently = False
        self.model = None
        self.optimizer = None

        self.time_map = None
        self.answer_vector_cache = None
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        # waiting to create the optimizer until a model is loaded
        if (not model == None):
            self.model = model
            self.model.to_device(device)
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.0001)#, lr=0.01
        else:
            print("Agent is waiting for model load!", flush = True)


    # Helper for randomly sampling negative labels - important to use in WARP loss function
    def get_random_negatives(self, pos, num):
        start_time = time.time()
        vals = torch.randperm(self.vocab.num_answers, device=device)[:num]
        if (not len(vals == pos) == 0):
            vals = torch.randperm(self.vocab.num_answers, device=device)[:num]

        self.time_map["total_rand_time"]+= time.time()-start_time
        return torch.unsqueeze(vals, 0)

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

        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.0001)
        print("Loaded model from: \"" + file_name + "\"", flush = True)

    # Run through a full cycle of training data - save freq and save_loc will determine whether the model is saved after the epoch is finished
    # save_freq
    def train_epoch(self, data_manager, save_freq, save_loc):
        epoch = data_manager.full_epochs
        use_question_cache = data_manager.use_bert_question_cache

        self.time_map = {"total_forward_time":0, "total_rand_time":0, "total_loss_time":0, "total_optimization_time":0, "warp_p1":0, "warp_p2":0, "warp_p3":0, "warp_p4":0}
        start_time = time.time()

        print("Starting train epoch #" + str(epoch), flush = True)
        while epoch == data_manager.full_epochs:
            inputs, labels = data_manager.get_next_batch(encode_index=False)
            self.train_step(epoch, inputs.to(device), labels.to(device), data_manager.batch_size, use_question_cache=use_question_cache)
            torch.cuda.empty_cache() # clear old tensors from VRAM

            if (int(data_manager.batch % 1000) == 0):
                print("Epoch " + str(epoch) + " progress: " + str(data_manager.get_epoch_completion()) + "%", flush=True)

            # saves mid-epoch at supplied interval
            wants_to_save = int(data_manager.get_epoch_completion()) % (save_freq) == 0 and data_manager.get_epoch_completion() > 1 and not (save_freq >= 100)
            if (wants_to_save and not self.saved_recently):
                self.save_model({"epoch":epoch, "completed":False}, save_loc + "/Model_epoch_" + str(epoch) + "_progress_" + str(int(data_manager.get_epoch_completion())) + "%.model")
                self.saved_recently=True
            elif(not wants_to_save and self.saved_recently):
                self.saved_recently=False

        # uncomment this to log more specific time diagnostics
        print("Finished in " + str(time.time()-start_time) + "s")
        #print("Total rand time: " + str(self.time_map["total_rand_time"]) + "s")
        #print("Total forward time: " + str(self.time_map["total_forward_time"]) + "s")
        #print("Total loss time: " + str(self.time_map["total_loss_time"]) + "s")
        #print("Total optimization time: " + str(self.time_map["total_optimization_time"]) + "s")
        #print("Total loss p1 time: " + str(self.time_map["warp_p1"]) + "s")
        #print("Total loss p2 time: " + str(self.time_map["warp_p2"]) + "s")
        #print("Total loss p3 time: " + str(self.time_map["warp_p3"]) + "s")
        #print("Total loss p4 time: " + str(self.time_map["warp_p4"]) + "s")

        # saves every epoch if allowed
        if (not save_freq > 100):
            self.save_model({"epoch":epoch, "completed":True}, save_loc + "/Model_epoch_" + str(epoch) + ".model")

    # Runs training step on one batch of tensors
    def train_step(self, epoch, input_question, answer_label, batch_size, use_question_cache=False):
        self.answer_vector_cache == None

        if (len(answer_label) == 0):
            return 0

        neg_labels = self.get_random_negatives(answer_label, 400)

        self.optimizer.zero_grad()

        #start_time = time.time()
        pos_res, neg_res = self.model(input_question.squeeze(1), answer_label.to(device), neg_labels, use_question_cache)
        #self.time_map["total_forward_time"]+=(time.time() - start_time)

        #start_time = time.time()
        loss = warp_loss(pos_res, neg_res, num_labels=self.vocab.num_answers, device=device, time_map=self.time_map, cuda_bool=cuda_bool)
        #self.time_map["total_loss_time"]+=(time.time() - start_time)

        #start_time = time.time()
        loss.backward()
        self.optimizer.step()
        #self.time_map["total_optimization_time"]+=(time.time() - start_time)

        self.total_examples += 1

        return 0

    # used to determine whether or not the model is doing autograd and such when running forward
    def model_set_mode(self, mode):
        if (mode == "eval"):
            self.model.eval()
        elif (mode == "train"):
            self.model.train()
        else:
            raise ValueError("No model mode \"" + mode + "\" exists", flush = True)

    # Prepare the embedded answers to compare against when guessing.
    # This gets called automatically the first time an answer guess is requested
    def cache_answer_vectors(self):
        self.answer_vector_cache = None
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.answer_vector_cache = self.model.embed_answer(torch.LongTensor(range(0, self.vocab.num_answers)).to(device))

    # returns the K most likely answers to the questions
    # This is able to produce metadata associated with the guess,
    # which eventually gets passed down the pipeline to the buzzer.
    def answer_knn(self, questions, k, question_pooled=False, id_only=False):
        answers = []

        if (self.answer_vector_cache == None):
            print("Must cache answer vectors before calculating KNN", flush=True)
            print("Caching answer vectors...", flush=True)
            self.cache_answer_vectors()
            print("Finished answer caching.. commence evaluation.", flush=True)

        with torch.no_grad():
            qs = self.model.embed_question(questions, expect_precalculated_pool=question_pooled)

            for encoded_q in qs:
                sim = self.cos_sim(encoded_q, self.answer_vector_cache.to(device))
                values,indices = sim.topk(k, largest=True)
                if (id_only):
                    answers.append(indices)
                else:
                    answers.append(self.vocab.decode(indices, values))

        if (id_only):
            return torch.LongTensor(answers)
        else:
            return answers

    # will return whatever buzzer needs to create its input vector
    # only used when running buzzer and guesser in tandem
    def model_topk(self, encoded_questions, tokenizer, k=1):
        output = []
        #questions = self.model.embed_question(encoded_questions, expect_precalculated_pool=False)
        for q in encoded_questions:
            guess = self.answer_knn(q.unsqueeze(0), k)[0]
            full_text = tokenizer.decode(q)
            meta = {
                        "guess" : guess[0][0],
                        "score" : guess[0][1].cpu().tolist(),
                        "guessId" : guess[0][2].cpu().tolist(),
                        "kguess" : [val[0] for val in guess],
                        "kguess_ids" : [val[2].cpu().tolist() for val in guess],
                        "kguess_scores" : [val[1].cpu().tolist() for val in guess],
                        "question_nonzero_tokens": torch.count_nonzero(q).tolist(), 
                        #"pooler_output" : self.model.get_last_pooler_output().cpu().tolist()[0],
                        "full_question" : full_text
                    }
            output.append(meta)
        return {"buzzer_data":output}

    # Runs through a full epoch of data in the data manager, evaluating the accuracy of the model.
    # If provided a save location, will produce a buzztrain file.
    def model_evaluate(self, data_manager, save_loc=None, k=10, tokenizer=None):
        epoch = data_manager.full_epochs
        pooled_questions = data_manager.use_bert_question_cache
        using_gpu = (not device == "cpu")
        acc=0.0
        num_batches = 0.0
        guess_metadata = []
        num_correct = 0

        # must go one at a time when recording pooler output
        if (not save_loc==None):
            data_manager.batch_size = 1

        with torch.no_grad():
            while epoch == data_manager.full_epochs:
                inputs, labels = data_manager.get_next_batch(encode_index=False)
                gpu_inputs = inputs.to(device)
                gpu_labels = labels.to(device)
                
                if (len(gpu_labels) == 0):
                    break

                num_batches+=1

                if (int(data_manager.batch % 5000) == 0):
                    print("Progress: " + str(data_manager.get_epoch_completion()) + "%", flush=True)

                if (save_loc==None): 
                    guesses = self.answer_knn(gpu_inputs, 1, question_pooled=pooled_questions, id_only=True).to(device)
                    acc += (data_manager.batch_size - torch.count_nonzero(torch.sub(gpu_labels, guesses)))/data_manager.batch_size
                else:
                    guesses = self.answer_knn(gpu_inputs, k, question_pooled=pooled_questions, id_only=False)
                    guess = guesses[0]
                    #print(labels[0].tolist(), flush=True)
                    #print("guess: ", flush=True)
                    #print(guess[0][2].cpu().tolist(), flush=True)
                    correct = (labels[0].tolist() == guess[0][2].cpu().tolist())

                    if (correct):
                        num_correct+=1

                    full_text = None
                    if (not tokenizer == None):
                        full_text = tokenizer.decode(data_manager.get_current_info())

                    meta = {
                        "guess" : guess[0][0],
                        "score" : guess[0][1].cpu().tolist(),
                        "guessId" : guess[0][2].cpu().tolist(),
                        "kguess" : [val[0] for val in guess],
                        "kguess_ids" : [val[2].cpu().tolist() for val in guess],
                        "kguess_scores" : [val[1].cpu().tolist() for val in guess],
                        "question_nonzero_tokens": torch.count_nonzero(data_manager.get_current_info()).tolist(), 
                        #"pooler_output" : self.model.get_last_pooler_output().cpu().tolist()[0],
                        "full_question" : full_text,
                        "label" : correct
                    }
                    guess_metadata.append(meta)
                
                # clear old tensors from VRAM
                if (using_gpu):
                    gpu_labels = None
                    gpu_inputs = None
                    torch.cuda.empty_cache() 

            if (save_loc==None):
                print("Final accuarcy score: " + str((acc/num_batches)*100) + "%", flush = True)
            else:
                textfile = open(save_loc, "w")
                json_dict = {
                        "buzzer_data":guess_metadata
                        }
                json.dump(json_dict, textfile)
                textfile.close()
                print(str(num_correct), flush=True)


# used to test this file - please ignore this junk
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

    #print(next_data[0].size())
    #print(next_data[1].size())

    #print(vocab.decode(agent.model_forward(next_data[0])))

    #agent.train_epoch(data, 1000, "training_progress")
    #agent.save_model({}, "training_progress/test_model.model")

    #agent.load_model("training_progress/test_model.model")
    #agent.train_epoch(data, 50, "training_progress")