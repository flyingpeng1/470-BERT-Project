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


class Project_BERT_Data_Manager:
    def __init__(self, maximum_question_length, maximum_answer_length, batch_size, tokenizer):
        self.tokenizer = tokenizer
        self.maximum_question_length = maximum_question_length
        self.maximum_answer_length = maximum_answer_length
        self.num_questions = 0
        self.questions = []
        self.answers = []
        self.current = 0
        self.batch = 0
        self.full_cycles = 0
        self.batch_size = batch_size

    def load_data(self, file_name, limit):
        with open(file_name) as json_data:
            temp_questions = []
            temp_answers = []

            data = json.load(json_data)
            qs = data["questions"]
            number = limit
            
            #if (limit == 0):
            #    print("Loaded 0? Why?")
            #    return

            for q in qs:
                print(q)
                if (limit > 0 and limit <= self.num_questions):     # stop loading when has enough
                    break;
                text = q["text"]                            
                answer = re.sub(r'\[or .*', '', q["answer"])    # removing secondary choices from answers
                answer = re.sub(r'\(or .*', '', answer)         # removing secondary choices from answers
                answer = re.sub(r'\[accept .*', '', answer)     # removing secondary choices from answers
                answer = re.sub(r'\[prompt .*', '', answer)     # removing secondary choices from answers
                answer = re.sub(r'\[be .*', '', answer)         # removing secondary choices from answers

                if (not "answer:" in text and not "ANSWER:" in text):   # making sure that the question is not an amalgam of multiple questions
                    question_encoded = self.tokenizer.encode(self.tokenizer.tokenize(text))
                    answer_encoded =   self.tokenizer.encode(self.tokenizer.tokenize(answer))

                    # Forcing question to be exactly the right length for BERT to accept
                    if (len(question_encoded) > self.maximum_question_length):
                        question_encoded = question_encoded[:self.maximum_question_length]
                    elif (len(question_encoded) < self.maximum_question_length):
                        question_encoded.extend([0] * (self.maximum_question_length - len(question_encoded)))

                    temp_questions.append(question_encoded)

                    # Forcing answer to be exactly the right length
                    if (len(answer_encoded) > self.maximum_answer_length):
                        answer_encoded = answer_encoded[:self.maximum_answer_length]
                    elif (len(answer_encoded) < self.maximum_answer_length):
                        answer_encoded.extend([0] * (self.maximum_answer_length - len(answer_encoded)))

                    temp_answers.append(answer_encoded)

                    self.num_questions += 1
                    if (self.num_questions%1000 == 0):
                        print("Loading dataset - Completed: " + str(self.num_questions/number * 100) + "%")

            self.questions = torch.LongTensor(temp_questions)
            self.answers = torch.LongTensor(temp_answers)
            print("Loaded " + str() + "")

    def get_next(self):
        self.cycle()
        element = (self.questions[self.current:self.current+1], self.answers[self.current:self.current+1])
        self.current += 1
        return element;

    def get_next_batch(self):
        if (self.current + self.batch_size > self.num_questions):
            batch = (self.questions[self.current:], self.answers[self.current:])
            self.current = self.num_questions
            self.cycle()
            return batch
        else:
            batch = (self.questions[self.current:self.current+self.batch_size], self.answers[self.current:self.current+self.batch_size])
            self.current += self.batch_size
            self.batch += 1
            return batch

    def get_cycles(self):
        return self.full_cycles

    def cycle(self):
        if (self.current >= self.num_questions):
            self.full_cycles += 1
            self.batch = 0
            self.current = 0

    def reset_cycle(self):
        self.current = 0
        self.batch = 0


# used to test this file
if __name__ == '__main__':
    loader = Project_BERT_Data_Manager(412, 30, 2, BertTokenizer.from_pretrained("bert-large-uncased"))
    loader.load_data("../../data/qanta.dev.2018.04.18.json", 2)
    print(loader.get_next_batch())
    print(loader.get_next_batch())
    print(loader.get_cycles())