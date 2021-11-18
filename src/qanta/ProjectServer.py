from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import click
from tqdm import tqdm
from flask import Flask, jsonify, request

from qanta.ProjectModel import *
from qanta.ProjectDataLoader import * 
from qanta import util

CACHE_LOCATION = "/cache"
VOCAB_LOCATION = "/data/qanta.vocab"
MODEL_LOCATION = "/data/BERTTest.model"

MAX_QUESTION_LENGTH = 412
BATCH_SIZE = 1

# We might not actually need this..
def guess_and_buzz(model, text):
    # TODO
    return ("", False)

# We might not actually need this..
def batch_guess_and_buzz(model, text):
    # TODO
    return []


# Executed in seperate thread so that the model can load without holding up the server.
def load_model(callback):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    vocab = load_vocab(VOCAB_LOCATION)
    model = None                                   # Need model saving and loading!
    callback(tokenizer, vocab, model)


class Project_Guesser():
    def __init__(self):
        print("Loading model")
        self.tokenizer = None
        self.vocab = None
        self.model = None
        self.ready = False
        self.load_thread = threading.Thread(target=load_model, args=[self.load_callback])
        self.load_thread.start()

    # Called with one question string
    def guess(self, text):
        if (not self.ready):
            self.load_thread.join()
        return ""

    # called with an array of questions
    def batch_guess(self, text):
        if (not self.ready):
            self.load_thread.join()
        return ["" for x in text]

    # Called to determine if the model has been loaded
    def isReady(self):
        return self.ready

    # Called after the loading thread is finished
    def load_callback(self, tokenizer, vocab, model):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.model = model
        self.ready = True
        print("Model is loaded!")

def create_app(enable_batch=True):
    guesser = Macaw_Guesser()
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
            'batch': enable_batch,
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


@click.group()
def cli():
    pass

@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=True)
    print("Started web app")

@cli.command()
@click.option('--vocab_file', default="src/data/qanta.vocab")
@click.option('--train_file', default="data/qanta.train.2018.04.18.json")
@click.option('--data_limit', default=-1)
@click.option('--epochs', default=1)
@click.option('--resume', default=False, is_flag=True)
@click.option('--resume_file', default="")
@click.option('--preloaded_manager', default=False, is_flag=True)
@click.option('--manager_file', default="")
def train(vocab_file, train_file, data_limit, epochs, resume, resume_file, preloaded_manager, manager_file):
    print("Loading resources...")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    vocab = load_vocab(vocab_file)
    data = Project_BERT_Data_Manager(MAX_QUESTION_LENGTH, vocab, BATCH_SIZE, tokenizer)
    data.load_data(train_file, data_limit)
    model = BERTModel(data.get_answer_vector_length())
    agent = BERTAgent(model, vocab)

    #if (resume):
    #if (preloaded_manager):

    print("Finished loading - commence training.")


    current_epoch = data.full_epochs
    while (current_epoch < epochs):
        current_epoch = data.full_epochs
        agent.train_epoch(data, 50, "training_progress")


    # We need to load the training files and run through them here... 
    return 0

@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    print("Starting QB")
    cli()
