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
EVALUATION_FILE_LOCATION = "/src/data/evaluation.json"

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
    buzz = buzzer.buzz(output)[0]
    return (output["buzzer_data"][0]["guess"], buzz.cpu().tolist()[0]>0.4, output["buzzer_data"][0]["kguess"],  output["buzzer_data"][0]["kguess_scores"], buzz.cpu().tolist()[0])

#=======================================================================================================
# Combines batch guesser and buzzer outputs
#=======================================================================================================
def batch_guess_and_buzz(guesser, buzzer, text):
    out = guesser.batch_guess(text)
    guesses = [g["guess"] for g in out["buzzer_data"]]
    kguess = [g["kguess"] for g in out["buzzer_data"]]
    kguess_scores = [g["kguess_scores"] for g in out["buzzer_data"]]
    buzzes = buzzer.buzz(out, will_round=False)
    return zip(guesses, buzzes, kguess, kguess_scores)

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
def load_buzzer(callback, vocab_file, buzzer_file, link_file):
    vocab = load_vocab(vocab_file)

    agent = BuzzAgent(None)
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

        output = self.agent.model_topk(encoded_question, self.tokenizer, k=5)

        #print(self.agent.vocab.decode_top_n(output.cpu(), 10))
        #print(self.agent.model.get_last_pooler_output())

        return output

    # called with an array of questions, returns a guess batch
    def batch_guess(self, text):
        if (not self.ready):
            self.load_thread.join()

        # Tokenize questions
        encoded_questions = torch.LongTensor([encode_question(t, self.tokenizer, MAX_QUESTION_LENGTH) for t in text]).to(device)

        output = self.agent.model_topk(encoded_question, self.tokenizer, k=5)

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

    def wait_for_load(self):
        self.load_thread.join()

class Project_Buzzer():
    def __init__(self, buzzer_file, vocab_file, links_file):
        self.agent = None
        self.tokenizer = None
        self.ready = False
        self.load_thread = threading.Thread(target=load_buzzer, args=[self.load_callback, vocab_file, buzzer_file, links_file])
        self.load_thread.start()

    def load_callback(self, agent):
        self.agent = agent
        self.ready = True
        print("Buzzer is loaded!", flush=True)

    # Uses buzzer
    def buzz(self, guesser_output):
        return self.agent.buzz(guesser_output, will_round=False)

    # Uses buzzer in batch
    def batch_buzz(self, guesser_output):
        return self.agent.buzz(guesser_output)

    def wait_for_load(self):
        self.load_thread.join()

    def isReady(self):
        return self.ready


#=======================================================================================================
# Called to start qb server.
#=======================================================================================================
def create_app(vocab_file, model_file, buzzer_file, links_file):
    guesser = Project_Guesser(vocab_file, model_file)
    buzzer = Project_Buzzer(buzzer_file, vocab_file, links_file)
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz, kguess, kguess_scores, confidence = guess_and_buzz(guesser, buzzer, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False, 'kguess':kguess, 'kguess_scores':kguess_scores, "confidence":confidence})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        print(guesser.isReady() and buzzer.isReady())
        return jsonify({
            'batch': False,
            'batch_size': 1,
            'ready': guesser.isReady() and buzzer.isReady(),
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': buzz, 'kguess':kguess, 'kguess_scores':kguess_scores}
            for guess, buzz, kguess, kguess_scores in batch_guess_and_buzz(guesser, buzzer, questions)
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
@click.option('--link_file', default=LINK_FILE_LOCATION)
def web(host, port, vocab_file, model_file, buzzer_file, link_file):    
    app = create_app(vocab_file, model_file, buzzer_file, link_file)
    print("Starting web app")
    app.run(host=host, port=port, debug=True)


# Run through a dataset to evaluate overall model accuracy
@cli.command()
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--model_file', default=MODEL_LOCATION)
@click.option('--buzzer_file', default=BUZZER_LOCATION)
@click.option('--link_file', default=LINK_FILE_LOCATION)
@click.option('--data_file', default=TEST_FILE_LOCATION)
@click.option('--save_loc', default=EVALUATION_FILE_LOCATION)
def playquizbowl(host, port, vocab_file, model_file, buzzer_file, link_file, data_file, save_loc):    
    guesser = Project_Guesser(vocab_file, model_file)
    buzzer = Project_Buzzer(buzzer_file, vocab_file, links_file)
    
    data = json.load(open(data_file))["questions"]

    guesser.wait_for_load()
    buzzer.wait_for_load()

    truly_incorrect = []
    truly_answer_refused = []

    num_questions=0
    num_guessed_correct=0
    num_buzzed_correct=0
    num_corrrect=0
    num_truly_incorrrect=0
    num_truly_refused=0
    num_possible=0

    for q in data:
        question = q["text"]
        answer = q["page"]

        guess, buzz, kguess, kguess_scores, confidence = guess_and_buzz(guesser, buzzer, question)
        num_questions+=1

        possible = not guesser.vocab.get_indexes([answer])[0] == 0

        if (guess == answer):
            num_guessed_correct+=1

        if (((not guess == answer) and not buzz) or guess == answer and buzz):
            num_buzzed_correct+=1

        if (guess == answer and buzz):
            num_correct+=1

        if ((not guess == answer) and buzz and possible):
            num_truly_incorrrect+=1
            truly_incorrect.append({"text":question, "page":answer, "kguess":kguess, "kguess_scores":kguess_scores, "confidence":confidence})

        if ((not guess == answer) and (not buzz) and possible):
            num_truly_refused+=1
            truly_answer_refused.append({"text":question, "page":answer, "kguess":kguess, "kguess_scores":kguess_scores, "confidence":confidence})

        if (possible):
            num_possible+=1


    textfile = open(save_loc, "w")
        json_dict = {
            "num_questions":0, 
            "num_corrrect":num_corrrect, 
            "num_truly_incorrrect":num_truly_incorrrect, 
            "num_guessed_correct":num_guessed_correct,
            "num_buzzed_correct":num_buzzed_correct,
            "num_possible":num_possible,
            "num_truly_refused":num_truly_refused,
            "incorrect":truly_incorrect,
            "truly_answer_refused":truly_answer_refused
            }
        json.dump(json_dict, textfile)
        textfile.close()

    print("Final accuracy (Refused to buzz considered wrong): " + str((num_buzzed_correct/num_questions)*100) + "%")
    print("Final accuracy (Refused to buzz considered correct): " + str((num_guessed_correct/num_questions)*100) + "%")

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
@click.option('--unfreeze_layers', default="12") # num layers to unfreeze, seperated by +
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


@click.option('--link', default='https://drive.google.com/u/0/uc?export=download&confirm=D8Pf&id=1XDTvJyHEozSXlZAAnJHR1FXbFlacgWsj')
@click.option('--location', default='data/')
def download_model(link, location):
    util.shell(f'wget -O {location} {link}')

#/src/data/wiki_links.csv
@cli.command()
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--buzzer_file', default=BUZZER_LOCATION)
@click.option('--data_file', default=BUZZTRAIN_LOCATION)
@click.option('--data_limit', default=-1)
@click.option('--num_epochs', default=10)
@click.option('--link_file', default=LINK_FILE_LOCATION)
@click.option('--batch_size', default=1)
def buzztrain(vocab_file, buzzer_file, data_file, data_limit, num_epochs, link_file, batch_size): 
    vocab = load_vocab(vocab_file)

    print("Initializing data", flush=True)
    model = BuzzModel(4)
    agent = BuzzAgent(model)
    agent.load_data_from_file(data_file, batch_size=batch_size)

    print("Training model", flush=True)
    agent.train(num_epochs, save_loc=buzzer_file)


@cli.command()
@click.option('--vocab_file', default=VOCAB_LOCATION)
@click.option('--buzzer_file', default=BUZZER_LOCATION)
@click.option('--data_file', default=BUZZTRAIN_LOCATION)
@click.option('--data_limit', default=-1)
@click.option('--num_epochs', default=10)
@click.option('--link_file', default=LINK_FILE_LOCATION)
@click.option('--batch_size', default=1)
def buzzeval(vocab_file, buzzer_file, data_file, data_limit, num_epochs, link_file, batch_size): 
    vocab = load_vocab(vocab_file)

    print("Initializing data", flush=True)
    agent = BuzzAgent(model)
    agent.load_model(buzzer_file)
    agent.load_data_from_file(data_file, batch_size=batch_size)

    print("Training model", flush=True)
    print("Buzz accuracy: " + str(agent.evaluate(agent.dataset.to(device))*100) + "%")





if __name__ == '__main__':
    print("Starting QB")
    cli()
