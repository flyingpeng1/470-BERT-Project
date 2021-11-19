from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path
import threading

import click
from tqdm import tqdm
from flask import Flask, jsonify, request

from qanta.ProjectModel import *
from qanta.ProjectDataLoader import * 
from qanta import util

if (torch.cuda.is_available()):
    print("CUDA is available")
else:
    print("CPU only")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_LOCATION = "/src/cache"
VOCAB_LOCATION = "/src/data/BERTTest.vocab"
MODEL_LOCATION = "/src/data/BERTTest.model"

TRAIN_CACHE_LOCATION = "cache"
TRAIN_VOCAB_LOCATION = "../data/BERTTest.vocab"
TRAINING_PROGRESS_LOCATION = "train_progress"

MAX_QUESTION_LENGTH = 412
BATCH_SIZE = 8

#=======================================================================================================
# Combines guesser and buzzer outputs
#=======================================================================================================
def guess_and_buzz(guesser, text):
    out = guesser.guess(text)
    return (out, False)

#=======================================================================================================
# Combines batch guesser and buzzer outputs
#=======================================================================================================
def batch_guess_and_buzz(guesser, text):
    out = guesser.batch_guess(text)
    return [(g, False) for g in out]

#=======================================================================================================
# Executed in seperate thread so that the model can load without holding up the server.
#=======================================================================================================
def load_model(callback, vocab_file, model_file):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    vocab = load_vocab(vocab_file)
    agent = BERTAgent(None, vocab)
    agent.load_model(model_file)
    agent.model_set_mode("eval")
    
    callback(agent, tokenizer)

#=======================================================================================================
# Generates gueses using a model from quizbowl questions.
#=======================================================================================================
class Project_Guesser():
    def __init__(self, vocab_file, model_file):
        print("Loading model")
        self.agent = None
        self.tokenizer = None
        self.ready = False
        self.load_thread = threading.Thread(target=load_model, args=[self.load_callback, vocab_file, model_file])
        self.load_thread.start()

    # Called with one question string
    def guess(self, text):
        if (not self.ready):
            self.load_thread.join()

        # Tokenize question
        encoded_question = torch.LongTensor([encode_question(text, self.tokenizer, MAX_QUESTION_LENGTH)])
        encoded_question.to(device)

        output = self.agent.model_forward(encoded_question)
        guesses = self.agent.vocab.decode(output)[0]
        return guesses

    # called with an array of questions, returns a guess batch
    def batch_guess(self, text):
        if (not self.ready):
            self.load_thread.join()

        # Tokenize questions
        encoded_questions = torch.LongTensor([encode_question(t, self.tokenizer, MAX_QUESTION_LENGTH) for t in text])
        encoded_questions.to(device)
        
        output = self.agent.model_forward(encoded_questions)
        guess = self.agent.vocab.decode(output)

        return [x for x in guess]

    # Called to determine if the model has been loaded
    def isReady(self):
        return self.ready

    # Called after the loading thread is finished
    def load_callback(self, agent, tokenizer):
        self.agent = agent
        self.tokenizer = tokenizer
        self.ready = True
        print("Model is loaded!")


#=======================================================================================================
# Called to start qb server.
#=======================================================================================================
def create_app(vocab_file, model_file):
    guesser = Project_Guesser(vocab_file, model_file)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        print(guesser.isReady())
        return jsonify({
            'batch': True,
            'batch_size': 10,
            'ready': guesser.isReady(),
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(guesser, questions)
        ])

    return app


#=======================================================================================================
# Click commands for sending server arguments. 
#=======================================================================================================
@click.group()
def cli():
    pass

# starts the qb answer server
@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
#@click.option('--disable-batch', default=False, is_flag=True)
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--model_file', default=MODEL_LOCATION)
def web(host, port, vocab_file, model_file):    
    app = create_app(vocab_file, model_file)
    print("Starting web app")
    app.run(host=host, port=port, debug=True)

# run to train the model - vocab_file and train_file are required!
@cli.command()
@click.option('--vocab_file', default=TRAIN_VOCAB_LOCATION)
@click.option('--train_file', default="../data/qanta.train.2018.04.18.json")
@click.option('--data_limit', default=-1)
@click.option('--epochs', default=1)
@click.option('--resume', default=False, is_flag=True)
@click.option('--resume_file', default="")
@click.option('--preloaded_manager', default=False, is_flag=True)
@click.option('--manager_file', default="")
def train(vocab_file, train_file, data_limit, epochs, resume, resume_file, preloaded_manager, manager_file):
    print("Loading resources...")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=TRAIN_CACHE_LOCATION)
    vocab = load_vocab(vocab_file)
    data = None
    agent = None

    if (preloaded_manager):
        data = load_data_manager(manager_file)
    else:
        data = Project_BERT_Data_Manager(MAX_QUESTION_LENGTH, vocab, BATCH_SIZE, tokenizer)
        data.load_data(train_file, data_limit)

    if (resume):
        agent = BERTAgent(None, vocab)
        agent.load_model(resume_file)
    else:
        agent = BERTAgent(BERTModel(data.get_answer_vector_length(), TRAIN_CACHE_LOCATION), vocab)
    
    print("Finished loading - commence training.")

    agent.model_set_mode("train")

    current_epoch = data.full_epochs
    while (current_epoch < epochs):
        current_epoch = data.full_epochs
        agent.train_epoch(data, 50, "training_progress")

    print("Training completed - " + str(epochs) + " full epochs")

# Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
@cli.command()
@click.option('--local-qanta-prefix', default='data/')
#@click.option('--retrieve-paragraphs', default=False, is_flag=True) #retrieve_paragraphs
def download(local_qanta_prefix):
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    print("Starting QB")
    cli()
