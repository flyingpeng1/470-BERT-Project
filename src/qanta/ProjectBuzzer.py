# Developed from Code from Alex Jian Zheng
import torch
import torch.nn as nn
import numpy as np
import json
import pandas

from torch.utils.data import Dataset, DataLoader
from qanta.ProjectDataLoader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=======================================================================================================
# Util
#=======================================================================================================
def give_confidence(guess, question_text, file_location):
    count = 0
    repeated = ""
    df = pandas.read_csv(file_location,dtype=str)
    for i in df[guess]:
        if str(i) in question_text and str(i) not in repeated and not str(i) == " ":
            repeated += i
            count += 1
    return count

#=======================================================================================================
# Data Classes
#=======================================================================================================
class Sample:
    def __init__(self, guess_data, vocab, link_file_location, labeled=True):
        """
        Create a new example

        label -- The label (0 / 1) of the example
        words -- The words in a list of "word:count" format
        vocab -- The vocabulary to use as features (list)
        """
        self.x = None
        self.y = None

        # FEATURES
        # first chunk of feature vector is one-hot encoding of answer
        #answer_tensor = vocab.encode_from_indexes([guess_data["guessId"]])[0]
        x = torch.FloatTensor().numpy()#answer_tensor.detach().to("cpu").numpy()

        # score
        x = np.append(x, guess_data["score"])
        # diff between top 2 scores
        x = np.append(x, guess_data["kguess_scores"][0] - guess_data["kguess_scores"][1])
        # ave diff between 1st score and others
        temp = [guess_data["kguess_scores"][0] - x for x in guess_data["kguess_scores"][1:]]
        x = np.append(x, sum(temp)/len(temp))
        # question length
        x = np.append(x, guess_data["question_nonzero_tokens"]/412)
        # num links in question
        #x = np.append(x, give_confidence(guess_data["guess"], guess_data["full_question"], link_file_location))

        self.x = x
        if labeled:
            self.y = float(guess_data["label"])

class GuessDataset(Dataset):
    def __init__(self, vocab, link_file_location):
        self.vocab = vocab
        # Add number of additional features to size of vocab to get total num features
        self.num_features = self.vocab.num_answers + 5
        self.feature = None
        self.label = None
        self.num_samples = 0
        self.link_file_location = link_file_location

    def initialize(self, buzzer_data, is_file=True, labeled=True):
        dataset = []
        if is_file:
            data = json.load(buzzer_data)
        else:
            data = buzzer_data

        for guess_data in data["buzzer_data"]:
            ex = Sample(guess_data, self.vocab, self.link_file_location, labeled=labeled)
            dataset.append(ex)
            self.num_samples += 1

        features = np.stack([ex.x for ex in dataset])
        self.feature = torch.from_numpy(features.astype(np.float32))
        if labeled:
            label = np.stack([np.array([ex.y]) for ex in dataset])
            self.label = torch.from_numpy(label.astype(np.float32))

        assert self.num_samples == len(self.feature)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.num_samples


#=======================================================================================================
# Model and Training
#=======================================================================================================
class LogRegModel(nn.Module):
    def __init__(self, num_features, num_hidden_units=50, nn_dropout=.5):
        super(LogRegModel, self).__init__()
        self.linear1 = nn.Linear(num_features, 1)#num_hidden_units)
        #self.linear2 = nn.Linear(num_hidden_units, 1)
        self.buzzer = nn.Sequential(self.linear1)#, nn.ReLU(), nn.Dropout(nn_dropout), self.linear2)

    def forward(self, x):
        y_pred = torch.sigmoid(self.buzzer(x))
        return y_pred

    def evaluate(self, data):
        """
        Computes the accuracy of the model. 
        """
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc

class LogRegAgent():
    def __init__(self, model, vocab, link_file="../wiki_links.csv", learnrate=0.01):
        if (model):
            self.model = model.to(device)
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learnrate)
        else:
            print("Buzzer waiting for model to load", flush=True)

        self.learnrate = learnrate
        self.vocab = vocab
        self.train_data_loader = None
        self.criterion = nn.BCELoss()
        self.link_file = link_file


    def load_data(self, buzzer_data_file, batch=1):
        data = GuessDataset(self.vocab, self.link_file)
        data.initialize(buzzer_data_file)
        self.data = data
        self.train_data_loader = DataLoader(dataset=data,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)

    def step(self, epoch, ex, model, inputs, labels):
        self.optimizer.zero_grad()
        predictions = model(inputs.to(device))
        loss = self.criterion(predictions, labels.to(device))
        loss.backward()
        self.optimizer.step()

    # Train the model 
    def train(self, num_epochs, model, save_loc):
        # Iterations
        for epoch in range(num_epochs):
            print("epoch " + str(epoch), flush=True)
            print(model.evaluate(self.data))
            for ex, (inputs, labels) in enumerate(self.train_data_loader):
                self.step(epoch, ex, model, inputs, labels)
                if (ex % 5000 == 0):
                    print("%"+str((ex/len(self.train_data_loader))*100), flush=True)

        if (save_loc):
            self.save_model({"epochs":num_epochs}, save_loc)

    # Save the model and its associated metadata 
    def save_model(self, metadata, save_location):
        pickle.dump({"model": self.model, "metadata":metadata}, open(save_location, "wb+"))
        print("Saved model to: \"" + save_location + "\"", flush=True)

    # Load the model and its associated metadata
    def load_model(self, location):
        load = pickle.load(open(location,'rb'))
        self.model = load["model"]
        self.model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learnrate)
        print("Loaded model from \"" + location + "\"", flush=True)

    def buzz(self, guess_dict, threshhold=.5):
        data = GuessDataset(self.vocab, self.link_file)
        data.initialize(guess_dict, is_file=False, labeled=False)
        y_pred = self.model.forward(data.feature.to(device))
        return y_pred
