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

#from transformers import BertModel
#from transformers import BertConfig

import re

# This is the object that gets pickled and saved with metadata about the vocabulary
# The vocabulary consists of all answers that we want to have as options for the guesser to select from.
class AnswerVocab:
    def __init__(self, num_answers, answers_list):
        self.num_answers = num_answers + 1
        self.answers = ["UNKNOWN"] + answers_list
        self.UNK_IDX = 0
        self.list_template = None

    # returns the answers in tensor form
    def encode(self, text_answers):
        self.list_template = [0] * (self.num_answers)
        encoded_answers = []
        for a in text_answers:
            idx = -1
            try:
                idx = self.answers.index(a)
            except:
                pass
            v = self.list_template.copy()
            
            # if it is not in the vocab, go with the UNK index
            if (idx == -1):
                v[self.UNK_IDX] = 1
            else:
                v[idx] = 1

            encoded_answers.append(v)
        return torch.LongTensor(encoded_answers)

    # returns the answers closest to the vectors
    def decode(self, ids):
        answers = []
        for a in ids:
            # Checking to make sure that the vector is the correct size, otherwise this doesn't make any sense
            if (len(a) != self.num_answers):
                raise ValueError("Received invalid vector of length " + str(len(a)) + ": expected " + str(self.num_answers))
            max_value = max(a)
            answers.append(self.answers[a.tolist().index(max_value)])
        return answers


# creates a map of the sums of all the pages (answers) in a given file
def count_pages(self, file_name):
    with open(file_name) as json_data:
        page_map = {}

        data = json.load(json_data)
        qs = data["questions"]

        print("working...")

        for q in qs:
            page = q['page']
            if (page in page_map):
                page_map[page] += 1
            else:
                page_map[page] = 1

        return page_map

# Creates some insights about the data we are working with
def analyze_dataset(filename):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    inputs = tokenizer.tokenize("During his reign, this man established subordinate bodies called the Triers and Ejectors to rule over clergy and teachers. Following a failed coup attempt against him led by John Lambert, he disbanded the Assembly of Saints, and\u00a0this ruler mandated that Parliament follow the \"four fundamentals.\"\u00a0His rule saw the conquering of Jamaica by the British. His government, which fought for the \"Good Old Cause,\" worked to oppress sects like the Fifth Monarchy Men. One of his greatest military campaigns concluded with the Battle of Worcester, a conflict that followed his predecessor's Bishops' Wars. This organizer of the New Model Army was succeeded by his son Richard. For 10 points, name this first Lord Protector of England.")
    inputs = torch.LongTensor(tokenizer.encode(inputs))
    labels = tokenizer.tokenize("Oliver Cromwell")
    labels = torch.LongTensor(tokenizer.encode(labels))

    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)

    '''
    model = BERTTest()
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    step(1, 1, model, optimizer, criterion, inputs, labels, vocab=[])
    '''

    max_question_length = 0
    max_question = []
    min_question_length = 10000000
    min_question = []
    max_answer_length = 0
    max_answer = []
    min_answer_length = 1000000
    min_answer = []

    with open(filename) as json_data:
        data = json.load(json_data)
        #print(data)
        qs = data["questions"]
        number = len(qs)
        completed = 0
        for q in qs:
            text = q["text"]                            
            answer = re.sub(r'\[or .*', '', q["answer"])    # removing secondary choices from answers
            answer = re.sub(r'\(or .*', '', answer)         # removing secondary choices from answers
            answer = re.sub(r'\[accept .*', '', answer)     # removing secondary choices from answers
            answer = re.sub(r'\[prompt .*', '', answer)     # removing secondary choices from answers
            answer = re.sub(r'\[be .*', '', answer)         # removing secondary choices from answers

            if (not "answer:" in text and not "ANSWER:" in text):   # making sure that the question is not an amalgam of multiple questions

                qlen = len(tokenizer.tokenize(text))
                alen = len(tokenizer.tokenize(answer))
                if (qlen > max_question_length):
                    max_question_length = qlen
                    max_question = q["text"]
                if (qlen < min_question_length):
                    min_question_length = qlen
                    min_question = q["text"]
                if (alen > max_answer_length):
                    max_answer_length = alen
                    max_answer = q["answer"]
                if (alen < min_answer_length):
                    min_answer_length = alen
                    min_answer = q["answer"]

                completed += 1
                if (completed%1000 == 0):
                    print("Analyzing dataset - Completed: " + str(completed/number * 100) + "%")

        print("Minimum question length: " + str(min_question_length) + " Maximum: " + str(max_question_length))
        print(min_question)
        print(max_question)

        print("Minimum answer length: " + str(min_answer_length) + " Maximum: " + str(max_answer_length))
        print(min_answer)
        print(max_answer)

    test_set = set(test_answers.keys())
    train_set = set(train_answers.keys())
    diff = test_set - train_set
    print("Number of train answers: " + str(len(train_set)))
    print("Number of test answers: " + str(len(test_set)))
    print("The number we know we won't get: " + str(len(diff)))
    print("Optimal percentage with current data: %" + str((1 - len(diff)/len(test_set))*100))

#=======================================================================================================
# Generates a vocab file using the answer data contained within file_name and saves it to save_location
#=======================================================================================================
def answer_vocab_generator(file_name, save_location):
    print("Preparing to generate answer vocabulary file")
    
    with open(file_name) as json_data:
        page_set = {} # I didn't use set because I wanted to keep the order of the vocab consistent with the data load order

        data = json.load(json_data)
        qs = data["questions"]

        print("working...")

        for q in qs:
            page = q['page']
            if (not page in page_set):
                page_set[page] = 1

        vocab = AnswerVocab(len(page_set.keys()), list(page_set.keys()))
        pickle.dump(vocab, open(save_location, "wb+"))
        print("Saved vocab file to: \"" + save_location + "\"")

#=======================================================================================================
# loads a vocab file from location specified.
#=======================================================================================================
def load_vocab(file_name):
    return pickle.load(open(file_name,'rb'))

#=======================================================================================================
# save a pre-transformed dataset so that it is faster to load in the future.
#=======================================================================================================
def save_data_manager(manager, save_location):
    pickle.dump(manager, open(save_location, "wb+"))
    print("Saved data manager file to: \"" + save_location + "\"")

#=======================================================================================================
# load a previously saved pre-transformed dataset.
#=======================================================================================================
def load_data_manager(file_name):
    print("Loading data manager - this may take a while...")
    return pickle.load(open(file_name,'rb'))

#=======================================================================================================
# Manages loading, transforming, and providing data to the model
#=======================================================================================================
class Project_BERT_Data_Manager:
    def __init__(self, maximum_question_length, answer_vocab, batch_size, tokenizer):
        self.tokenizer = tokenizer
        self.maximum_question_length = maximum_question_length
        self.answer_vocab = answer_vocab
        self.num_questions = 0
        self.questions = []
        self.answers = []
        self.current = 0
        self.batch = 0
        self.full_epochs = 0
        self.batch_size = batch_size

    def load_data(self, file_name, limit):
        with open(file_name) as json_data:
            temp_questions = []
            temp_answers = None
            
            data = json.load(json_data)
            qs = data["questions"]
            questions_length = len(qs)
            number = limit

            for q in qs:
                if (limit > 0 and limit <= self.num_questions):     # stop loading when has enough
                    break;
                text = q["text"]               
                answer = q["page"]             
                #answer = re.sub(r'\[or .*', '', q["answer"])    # removing secondary choices from answers
                #answer = re.sub(r'\(or .*', '', answer)         # removing secondary choices from answers
                #answer = re.sub(r'\[accept .*', '', answer)     # removing secondary choices from answers
                #answer = re.sub(r'\[prompt .*', '', answer)     # removing secondary choices from answers
                #answer = re.sub(r'\[be .*', '', answer)         # removing secondary choices from answers

                if (not "answer:" in text and not "ANSWER:" in text):   # making sure that the question is not an amalgam of multiple questions
                    question_encoded = self.tokenizer.encode(self.tokenizer.tokenize(text))
                    answer_encoded = self.answer_vocab.encode([answer])

                    # Forcing question to be exactly the right length for BERT to accept
                    if (len(question_encoded) > self.maximum_question_length):
                        question_encoded = question_encoded[:self.maximum_question_length]
                    elif (len(question_encoded) < self.maximum_question_length):
                        question_encoded.extend([0] * (self.maximum_question_length - len(question_encoded)))
                    
                    if (temp_answers == None):
                        temp_answers = answer_encoded
                    else:
                        torch.cat((temp_answers, answer_encoded))
                    temp_questions.append(question_encoded)

                    self.num_questions += 1
                    if (self.num_questions%1000 == 0):
                        print("Loading dataset - Completed: " + str(self.num_questions/questions_length * 100) + "%")

            self.questions = torch.LongTensor(temp_questions)
            self.answers = temp_answers
            print("Loaded " + str() + "")

    def get_next(self):
        self.epoch()
        element = (self.questions[self.current:self.current+1], self.answers[self.current:self.current+1])
        self.current += 1
        return element;

    def get_next_batch(self):
        if (self.current + self.batch_size > self.num_questions):
            batch = (self.questions[self.current:], self.answers[self.current:])
            self.current = self.num_questions
            self.epoch()
            return batch
        else:
            batch = (self.questions[self.current:self.current+self.batch_size], self.answers[self.current:self.current+self.batch_size])
            self.current += self.batch_size
            self.batch += 1
            return batch

    def get_epochs(self):
        return self.full_epochs

    def epoch(self):
        if (self.current >= self.num_questions):
            self.full_epochs += 1
            self.batch = 0
            self.current = 0

    def reset_epoch(self):
        self.current = 0
        self.batch = 0

    def get_answer_vector_length(self):
        print(self.answer_vocab.num_answers)
        return self.answer_vocab.num_answers


# used to test this file
if __name__ == '__main__':
    answer_vocab_generator("../data/qanta.train.2018.04.18.json", "data/qanta.vocab")
    
    #vocab = load_vocab("data/qanta.vocab")
    #tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    #loader = Project_BERT_Data_Manager(412, vocab, 10, tokenizer)
    #test_answers = count_pages("../data/qanta.test.2018.04.18.json")
    #train_answers = count_pages("../data/qanta.train.2018.04.18.json")
    

    #loader.load_data("../data/qanta.train.2018.04.18.json", 1)
    #data = loader.get_next()
    #decode = vocab.decode(data[1])
    #print(decode)


    #save_data_manager(loader, "data/train.manager")
    #loader = load_data_manager("data/train.manager")
    #print(loader.get_next_batch())

    #print(loader.get_next_batch())
    #print(loader.get_cycles())