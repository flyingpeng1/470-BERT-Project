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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is the object that gets pickled and saved with metadata about the vocabulary
# The vocabulary consists of all answers that we want to have as options for the guesser to select from.
class AnswerVocab:
    def __init__(self, num_answers, answers_list):
        self.num_answers = num_answers + 1
        self.answers = ["UNKNOWN"] + answers_list
        self.UNK_IDX = 0
        self.list_template = None

    def get_indexes(self, text_answers):
        idxs = []
        for a in text_answers:
            idx = -1
            try:
                idx = self.answers.index(a)
            except:
                pass

            # if it is not in the vocab, go with the UNK index
            if (idx == -1):
                idxs.append(self.UNK_IDX)
            else:
                idxs.append(idx)

        return idxs

    def encode_from_indexes(self, idxs):
        self.list_template = [0] * (self.num_answers)
        encoded = []
        for a in idxs:
            v = self.list_template.copy()
            v[a] = 1
            encoded.append(v)
        return torch.LongTensor(encoded)

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
    def decode(self, ids, values):
        answ = [(self.answers[idx], values[loc], idx) for loc, idx in enumerate(ids)]
        answ.sort(reverse=True, key=lambda x: x[1])
        return answ

        #answers = []
        #for a in ids:
            # Checking to make sure that the vector is the correct size, otherwise this doesn't make any sense
            #if (len(a) != self.num_answers):
            #    raise ValueError("Received invalid vector of length " + str(len(a)) + ": expected " + str(self.num_answers))
            #max_value = max(a)
            #answers.append(self.answers[a.tolist().index(max_value)])
        #return answers

    # This has been made obselete in the WARP version of QuizBert
    def decode_top_n(self, ids, n):
        answers = []
        for a in ids:
            # Checking to make sure that the vector is the correct size, otherwise this doesn't make any sense
            if (len(a) != self.num_answers):
                raise ValueError("Received invalid vector of length " + str(len(a)) + ": expected " + str(self.num_answers))
            array = a
            ind = np.argpartition(array, -n)[-n:]
            answ = [self.answers[x] for x in ind.tolist()]
            val = array[ind]

            answers.append(list(zip(answ, val)))
        answers[0].sort(reverse=True, key=lambda x: x[1])
        return answers[0]

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
def answer_vocab_generator(file_name, save_location, category_only=False):
    print("Preparing to generate answer vocabulary file", flush = True)
    
    with open(file_name) as json_data:
        page_set = {} # I didn't use set because I wanted to keep the order of the vocab consistent with the data load order

        data = json.load(json_data)
        qs = data["questions"]

        print("working...", flush = True)

        for q in qs:
            page = None
            if (category_only):
                page = q['category']
            else:
                page = q['page']
  
            if (not page in page_set):
                page_set[page] = 1

        vocab = AnswerVocab(len(page_set.keys()), list(page_set.keys()))
        pickle.dump(vocab, open(save_location, "wb+"))
        print("Saved vocab file to: \"" + save_location + "\"", flush = True)

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
    print("Saved data manager file to: \"" + save_location + "\"", flush = True)

#=======================================================================================================
# load a previously saved pre-transformed dataset.
#=======================================================================================================
def load_data_manager(file_name):
    print("Loading data manager - this may take a while...", flush = True)
    return pickle.load(open(file_name,'rb'))

#=======================================================================================================
# Helper that takes a question and the tokenizer, and encodes the question properly.
#=======================================================================================================
def encode_question(question, tokenizer, maximum_question_length):
    question_encoded = []

    if (len(question) != 0):
        question_encoded = tokenizer.encode(tokenizer.tokenize(question))

    qlen = len(question_encoded)

    # Forcing question to be exactly the right length for BERT to accept
    if (qlen > maximum_question_length):
        question_encoded = question_encoded[:maximum_question_length]
    elif (qlen < maximum_question_length):
        question_encoded.extend([0] * (maximum_question_length - qlen))

    return question_encoded

#=======================================================================================================
# Helper that splits a string into sentences.
#=======================================================================================================
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

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
        self.cached_bert_questions = []
        self.answer_indexes = []
        self.current = 0
        self.batch = 0
        self.full_epochs = 0
        self.use_bert_question_cache = False
        self.batch_size = batch_size


    # will cache bert pooled output of question if provided with bert model
    def load_data(self, file_name, limit, split_sentences=False, category_only=False, bert_model=None):
        with open(file_name) as json_data:
            temp_questions = []
            temp_answers = None
            
            data = json.load(json_data)
            qs = data["questions"]
            questions_length = len(qs)
            temp_answers = []

            for q in qs:
                if (limit > 0 and limit <= self.num_questions):     # stop loading when has enough
                    break

                text = q["text"]
                answer = None
                if (category_only):
                    answer = q["category"]
                else:
                    answer = q["page"]

                if (not "answer:" in text and not "ANSWER:" in text):   # making sure that the question is not an amalgam of multiple questions

                    if (split_sentences):
                        sentences = split_into_sentences(text)
                        # iterate through each sentence if split_sentences=True, else use all sentences at once
                        i = range(len(sentences)) if split_sentences else [len(sentences)-1]
                        for s in i:
                            partial_text = ' '.join(sentences[:s+1])
                            question_encoded = encode_question(partial_text, self.tokenizer, self.maximum_question_length)
                            temp_answers.append(answer)
                            temp_questions.append(question_encoded)
                    else:
                        question_encoded = encode_question(text, self.tokenizer, self.maximum_question_length)
                        temp_answers.append(answer)
                        temp_questions.append(question_encoded)

                    self.num_questions += 1
                    if (self.num_questions%1000 == 0):
                        print("Loading dataset - Completed: " + str(self.num_questions/questions_length * 100) + "%", flush = True)   

            self.questions = torch.LongTensor(temp_questions)
            self.questions = self.questions
            self.answer_indexes = self.answer_vocab.get_indexes(temp_answers) # turn all of the answers into indexes


            if (not bert_model == None):
                self.use_bert_question_cache = True
                bert_model = bert_model.to(device)
                print("caching train question BERT encodings..", flush=True)
                for n, q in enumerate(self.questions):
                    self.cached_bert_questions.append(bert_model(q.to(device).unsqueeze(dim=0)).pooler_output.tolist())
                    if (n%200 == 0):
                        torch.cuda.empty_cache()
                        print("Completed: " + str((n/self.num_questions)*100) + "%", flush=True)
                self.cached_bert_questions = torch.FloatTensor(self.cached_bert_questions)
                print("finished caching train question BERT encodings!", flush=True)
                

            print("Loaded " + str() + "", flush = True)

    def get_next(self, encode_index=True):
        self.epoch()
        encode = lambda x : torch.LongTensor(x)
        if (encode_index):
            encode = lambda x : self.answer_vocab.encode_from_indexes(x)

        question_representation = None
        if (self.use_bert_question_cache):
            question_representation = self.cached_bert_questions
        else:
            question_representation = self.questions

        element = (question_representation[self.current:self.current+1], encode(self.answer_indexes[self.current:self.current+1]))
        self.current += 1
        return element

    def get_next_batch(self, encode_index=True):
        encode = lambda x : torch.LongTensor(x)
        question_representation = None
        if (encode_index):
            encode = lambda x : self.answer_vocab.encode_from_indexes(x)

        if (self.use_bert_question_cache):
            question_representation = self.cached_bert_questions
        else:
            question_representation = self.questions

        if (self.current + self.batch_size > self.num_questions):
            batch = (question_representation[self.current:], encode(self.answer_indexes[self.current:]))
            self.current = self.num_questions
            self.epoch()
            return batch
        else:
            batch = (question_representation[self.current:self.current+self.batch_size], encode(self.answer_indexes[self.current:self.current+self.batch_size]))
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

    def get_epoch_completion(self):
        return (self.batch/(self.num_questions/self.batch_size))*100

    def get_answer_vector_length(self):
        print(self.answer_vocab.num_answers)
        return self.answer_vocab.num_answers

    # This just returns the most recent encoded question 
    def get_current_info(self):
        if (self.current==0):
            raise Exeption("No recent data provided - nothing to provide info on!")
        return self.questions[self.current - 1]


# used to test this file
if __name__ == '__main__':
    #answer_vocab_generator("../data/qanta.train.2018.04.18.json", "data/qanta.vocab")
    
    vocab = load_vocab("../data/QuizBERT.vocab")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir="cache")
    loader = Project_BERT_Data_Manager(412, vocab, 10, tokenizer)
    #test_answers = count_pages("../data/qanta.test.2018.04.18.json")
    #train_answers = count_pages("../data/qanta.train.2018.04.18.json")
    

    loader.load_data("../data/qanta.dev.2018.04.18.json", 10, split_sentences=True)
    print(list(loader.questions))
    #data = loader.get_next()
    
    #data[1][0][0] = 10
    #data[1][0][5] = 11
    #data[1][0][7] = 12

    #print(data)
    #decode = vocab.decode_top_n(data[1], 10)
    #print(decode)


    # save_data_manager(loader, "../data/QBERT_Data.manager")
    #loader = load_data_manager("data/train.manager")
    #print(loader.get_next_batch())

    #print(loader.get_next_batch())
    #print(loader.get_cycles())