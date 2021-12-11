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
from qanta.ProjectBuzzer import *
from qanta import util

if (torch.cuda.is_available()):
    print("CUDA is available")
else:
    print("CPU only")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_LOCATION = "/src/cache"
VOCAB_LOCATION = "/src/data/QuizBERT.vocab"
DATA_MANAGER_LOCATION = "/src/data/QBERT_Data.manager"
MODEL_LOCATION = "/src/data/QuizBERT.model"
BUZZER_LOCATION = "/src/data/QuizBERTBuzzer.model"
TRAIN_FILE_LOCATION = "/src/data/qanta.train.2018.04.18.json"
TEST_FILE_LOCATION = "/src/data/qanta.test.2018.04.18.json"
TRAINING_PROGRESS_LOCATION = "training_progress"
BUZZTRAIN_LOCATION = "/src/data/buzztrain.json"
LINK_FILE_LOCATION = "/src/data/wiki_links.csv"

LOCAL_CACHE_LOCATION = "cache"
LOCAL_VOCAB_LOCATION = "/src/data/QuizBERT.vocab"
LOCAL_MODEL_LOCATION = "/src/data/QuizBERT.model"
LOCAL_TRAINING_PROGRESS_LOCATION = "train_progress"

MAX_QUESTION_LENGTH = 412
BATCH_SIZE = 1

def get_eval_only_bert_model(cache_location):
    bert = BertModel.from_pretrained("bert-base-uncased", cache_dir=CACHE_LOCATION)
    modules = [bert.embeddings, bert.encoder.layer]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

    return bert

#=======================================================================================================
# Combines guesser and buzzer outputs
#=======================================================================================================
def guess_and_buzz(guesser, buzzer, text):
    output = guesser.guess(text)
    buzz = buzzer.buzz(output)
    return (out["guess"], buzz)

#=======================================================================================================
# Combines batch guesser and buzzer outputs
#=======================================================================================================
def batch_guess_and_buzz(guesser, buzzer, text):
    out = guesser.batch_guess(text)
    guesses = [g["guess"] for g in out]
    buzzes = buzzer.batch_buzz(out)
    return zip(guesses, buzzes)

#=======================================================================================================
# Executed in seperate thread so that the model can load without holding up the server.
#=======================================================================================================
def load_model(callback, vocab_file, model_file):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    vocab = load_vocab(vocab_file)

    #agent = BERTAgent(QuizBERT(25970, cache_dir=CACHE_LOCATION), vocab)
    agent = BERTAgent(None, vocab)
    agent.load_model(model_file)
    agent.model_set_mode("eval")
    
    callback(agent, tokenizer)

#=======================================================================================================
# Executed in seperate thread so that the model can load without holding up the server.
#=======================================================================================================
def load_buzzer(callback, vocab_file, buzzer_file):
    vocab = load_vocab(vocab_file)

    agent = LogRegAgent(None, vocab)
    agent.load_model(buzzer_file)
    agent.model.eval()
    
    callback(agent)

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
        encoded_question = torch.LongTensor([encode_question(text, self.tokenizer, MAX_QUESTION_LENGTH)]).to(device)

        #print(encoded_question)

        output = self.agent.model_topk(encoded_question)

        #print(self.agent.vocab.decode_top_n(output.cpu(), 10))
        #print(self.agent.model.get_last_pooler_output())

        return output

    # called with an array of questions, returns a guess batch
    def batch_guess(self, text):
        if (not self.ready):
            self.load_thread.join()

        # Tokenize questions
        encoded_questions = torch.LongTensor([encode_question(t, self.tokenizer, MAX_QUESTION_LENGTH) for t in text]).to(device)
        
        output = self.agent.model_topk(encoded_questions)

        return output

    # Called to determine if the model has been loaded
    def isReady(self):
        return self.ready

    # Called after the loading thread is finished
    def load_callback(self, agent, tokenizer):
        self.agent = agent
        self.tokenizer = tokenizer
        self.ready = True
        print("Model is loaded!", flush=True)

class Project_Buzzer():
    def __init__(self, buzzer_file, vocab_file):
        self.agent = None
        self.tokenizer = None
        self.ready = False
        self.load_thread = threading.Thread(target=load_buzzer, args=[self.load_callback, vocab_file, buzzer_file])
        self.load_thread.start()

    def load_callback(self, agent):
        self.agent = agent
        self.ready = True
        print("Buzzer is loaded!", flush=True)

    # Will return buzz in future
    def buzz(self, guesser_output):
        return ([False])[0]

    # Will return buzz in future
    def batch_buzz(self, guesser_output):
        return [False for g in guesser_output]


#=======================================================================================================
# Called to start qb server.
#=======================================================================================================
def create_app(vocab_file, model_file, buzzer_file):
    guesser = Project_Guesser(vocab_file, model_file)
    buzzer = Project_Buzzer(buzzer_file)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(guesser, buzzer, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        print(guesser.isReady() and buzzer.isReady())
        return jsonify({
            'batch': True,
            'batch_size': 10,
            'ready': guesser.isReady() and buzzer.isReady(),
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': buzz}
            for guess, buzz in batch_guess_and_buzz(guesser, buzzer, questions)
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
@click.option('--buzzer_file', default=BUZZER_LOCATION)
def web(host, port, vocab_file, model_file, buzzer_file):    
    app = create_app(vocab_file, model_file, buzzer_file)
    print("Starting web app")
    app.run(host=host, port=port, debug=True)

# run to train the model - vocab_file and train_file are required!
@cli.command()
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--train_file', default=TRAIN_FILE_LOCATION)
@click.option('--data_limit', default=-1)
@click.option('--epochs', default=1)
@click.option('--resume', default=False, is_flag=True)
@click.option('--resume_file', default="")
@click.option('--preloaded_manager', default=False, is_flag=True)
@click.option('--manager_file', default=DATA_MANAGER_LOCATION)
@click.option('--save_regularity', default=1000000)
@click.option('--category_only', default=False, is_flag=True)
@click.option('--eval_freq', default=0)
@click.option('--unfreeze_layers', default="13") # num layers to unfreeze, seperated by +

def train(vocab_file, train_file, data_limit, epochs, resume, resume_file, preloaded_manager, manager_file, save_regularity, category_only, eval_freq, unfreeze_layers):
    print("Loading resources...", flush = True)
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    vocab = load_vocab(vocab_file)
    data = None
    agent = None

    if (preloaded_manager):
        data = load_data_manager(manager_file)
        data.batch_size = BATCH_SIZE # set the correct batch size
    else:
        model = get_eval_only_bert_model(CACHE_LOCATION)
        data = Project_BERT_Data_Manager(MAX_QUESTION_LENGTH, vocab, BATCH_SIZE, tokenizer)
        data.load_data(train_file, data_limit, category_only=category_only) #, bert_model=model

    if (resume):
        agent = BERTAgent(None, vocab)
        agent.load_model(resume_file, data)
    else:
        agent = BERTAgent(QuizBERT(data.get_answer_vector_length(), cache_dir=CACHE_LOCATION), vocab)
    
    print("Finished loading - commence training.", flush = True)

    train_layers = unfreeze_layers.split("+")

    agent.model.unfreeze_layers(train_layers)

    saved_recently = False
    current_epoch = data.full_epochs
    while (current_epoch < epochs):
        current_epoch = data.full_epochs
        wants_to_save=(not current_epoch==0 and save_regularity>100 and save_regularity<1000000 and (current_epoch*100) % save_regularity == 0)
        if (wants_to_save and not saved_recently):
            saved_recently = True
            agent.save_model({"epoch":data.full_epochs, "completed":True}, TRAINING_PROGRESS_LOCATION + "/QuizBERT_epoch_" + str(current_epoch) + ".model")
        elif(not wants_to_save):
            saved_recently = False 

        # Evaluating the model accuarcy every n epochs
        if ((eval_freq > 0) and (current_epoch+1)%eval_freq==0):
            data.reset_epoch()
            agent.model_evaluate(data)
            data.reset_epoch()
            data.full_epochs = data.full_epochs-1 # Remove the epoch used for evaluation

        agent.train_epoch(data, save_regularity, TRAINING_PROGRESS_LOCATION)

    agent.save_model({"epoch":data.full_epochs, "completed":True}, TRAINING_PROGRESS_LOCATION + "/QuizBERT.model")

    print("Training completed - " + str(epochs) + " full epochs", flush = True)


# High-efficiency evaluation - has an option to generate buzztrain file
@cli.command()
#@click.option('--disable-batch', default=False, is_flag=True)
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--model_file', default=MODEL_LOCATION)
@click.option('--split_sentences', default=False, is_flag=True)
@click.option('--dobuzztrain', default=False, is_flag=True)
@click.option('--buzztrainfile', default=BUZZTRAIN_LOCATION)
@click.option('--preloaded_manager', default=False, is_flag=True)
@click.option('--manager_file', default=DATA_MANAGER_LOCATION)
@click.option('--data_file', default=TEST_FILE_LOCATION)
@click.option('--top_k', default=10)
@click.option('--category_only', default=False, is_flag=True)
@click.option('--data_limit', default=-1)
def evaluate(vocab_file, model_file, split_sentences, dobuzztrain, buzztrainfile, preloaded_manager, manager_file, data_file, top_k, category_only, data_limit): 
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    vocab = load_vocab(vocab_file)
    data = None
    agent = None

    save_loc = None
    if (dobuzztrain):
        save_loc = buzztrainfile

    agent = BERTAgent(None, vocab)
    agent.load_model(model_file, data)


    if (preloaded_manager):
        data = load_data_manager(manager_file)
        data.batch_size = BATCH_SIZE # set the correct batch size
    else:
        data = Project_BERT_Data_Manager(MAX_QUESTION_LENGTH, vocab, BATCH_SIZE, tokenizer)
        data.load_data(data_file, data_limit, split_sentences=split_sentences, category_only=category_only)

    print("Finished loading - commence evaluation.", flush = True)

    agent.model_evaluate(data, save_loc, top_k, tokenizer)

    print("Finished evaluation")


# Run to generate vocab file in specified location using specified data file.
@cli.command()
@click.option('--save_location', default=VOCAB_LOCATION)
@click.option('--data_file', default=TRAIN_FILE_LOCATION)
@click.option('--category_only', default=False, is_flag=True)
def vocab(save_location, data_file, category_only):
    answer_vocab_generator(data_file, save_location, category_only=category_only)

# Run to generate data manager file in specified location using specified data file.
@cli.command()
@click.option('--vocab_location', default=VOCAB_LOCATION)
@click.option('--save_location', default=DATA_MANAGER_LOCATION)
@click.option('--data_file', default=TRAIN_FILE_LOCATION)
@click.option('--limit', default=-1)
@click.option('--category_only', default=False, is_flag=True)
@click.option('--cache_pool', default=False, is_flag=True)
@click.option('--split_sentences', default=False, is_flag=True)
def makemanager(vocab_location, save_location, data_file, limit, category_only, cache_pool, split_sentences):
    vocab = load_vocab(vocab_location)
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    
    model = None
    if (cache_pool):
        model = get_eval_only_bert_model(CACHE_LOCATION)
    
    loader = Project_BERT_Data_Manager(MAX_QUESTION_LENGTH, vocab, BATCH_SIZE, tokenizer)
    loader.load_data(data_file, limit, category_only=category_only, split_sentences=split_sentences, bert_model=model)
    save_data_manager(loader, DATA_MANAGER_LOCATION)

@cli.command()
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--manager_file', default=DATA_MANAGER_LOCATION)
def managerdatabase(vocab_file, manager_file):
    vocab = load_vocab(vocab_file)
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", cache_dir=CACHE_LOCATION)
    db = Data_Manager_Database(vocab, tokenizer)
    db.load(manager_file, "manager")
    run = True
    while (run):
        run = db.command()


# Run to check if cuda is available.
@cli.command()
def cudatest():
    print(device)

# Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
@cli.command()
@click.option('--local-qanta-prefix', default='data/')
#@click.option('--retrieve-paragraphs', default=False, is_flag=True) #retrieve_paragraphs
def download(local_qanta_prefix):
    util.download(local_qanta_prefix, False)


#/src/data/wiki_links.csv
@cli.command()
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--buzzer_file', default=BUZZER_LOCATION)
@click.option('--data_file', default=BUZZTRAIN_LOCATION)
@click.option('--data_limit', default=-1)
@click.option('--num_epochs', default=10)
@click.option('--link_file', default=LINK_FILE_LOCATION)
def buzztrain(vocab_file, buzzer_file, data_file, data_limit, num_epochs, link_file): 
    vocab = load_vocab(vocab_file)
    data = GuessDataset(vocab, link_file)
    print("Initializing data", flush=True)
    data.initialize(open(data_file))

    print("Training model", flush=True)
    model = LogRegModel(len(data[0][0]))
    agent = LogRegAgent(model, vocab, link_file)
    agent.load_data(data)
    agent.train(num_epochs, model, buzzer_file)


if __name__ == '__main__':
    print("Starting QB")
    cli()
