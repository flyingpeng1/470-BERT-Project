import torch
import torch.nn as nn
import numpy as np
import json
import pandas

from torch.utils.data import Dataset, DataLoader
from qanta.ProjectDataLoader import *
from qanta.util import give_confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#=======================================================================================================
# Data Classes
#=======================================================================================================
class Sample:
    def __init__(self, guess_data):
        self.x = None
        self.y = None

        features = []

        # score
        features.append(guess_data["score"])
        # diff between top 2 scores
        features.append(guess_data["kguess_scores"][0] - guess_data["kguess_scores"][1])
        # ave diff between 1st score and others
        temp = [guess_data["kguess_scores"][0] - x for x in guess_data["kguess_scores"][1:]]
        features.append(sum(temp)/len(temp))
        # question length
        features.append(guess_data["question_nonzero_tokens"]/412)

        self.x = np.array(features)
        if "label" in guess_data:
            self.y = np.array([float(guess_data["label"])])
        else:
            self.y = np.array([0.0])

class GuessDataset(Dataset):
    def __init__(self, guess_dataset):
        self.num_features = 4
        self.feature = None
        self.label = None
        self.num_samples = 0

        dataset = []
        for guess_data in guess_dataset["buzzer_data"]:
            ex = Sample(guess_data)
            dataset.append(ex)
            self.num_samples += 1

        features = np.stack([ex.x for ex in dataset])
        self.feature = torch.from_numpy(features.astype(np.float32))
        label = np.stack([ex.y for ex in dataset])
        self.label = torch.from_numpy(label.astype(np.float32))

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.num_samples


#=======================================================================================================
# Model and Training
#=======================================================================================================
class BuzzModel(nn.Module):
    def __init__(self, num_features, num_hidden_units=50, nn_dropout=.5):
        super(BuzzModel, self).__init__()
        self.linear1 = nn.Linear(num_features, num_hidden_units)
        self.linear2 = nn.Linear(num_hidden_units, 1)
        self.buzzer = nn.Sequential(self.linear1, nn.ReLU(), nn.Dropout(nn_dropout), self.linear2)

    def forward(self, x):
        y_pred = torch.sigmoid(self.buzzer(x))
        return y_pred

    def evaluate(self, data):
        with torch.no_grad():
            y_pred = self(data.feature)
            y_pred_cls = y_pred.round()
            acc = y_pred_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc

class BuzzAgent():
    def __init__(self, model, learnrate=0.01):
        if (model):
            self.model = model.to(device)
            self.criterion = nn.BCELoss()
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learnrate)
        else:
            print("Buzzer waiting for model to load", flush=True)

        self.data_loader = None

    def load_data(self, guess_dataset, batch_size=1):
        data = GuessDataset(guess_dataset)
        self.data_loader = DataLoader(dataset=data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    def load_data_from_file(self, guess_dataset_file, batch_size=1):
        with open(guess_dataset_file) as file:
            self.load_data(json.load(file), batch_size)

    # Train the model 
    def train(self, num_epochs=100, save_loc=None):
        # Iterations
        for epoch in range(num_epochs):
            print("epoch " + str(epoch), flush=True)
            for ex, (inputs, labels) in enumerate(self.data_loader):
                y_pred = self.model(inputs.to(device))
                loss = self.criterion(y_pred, labels.to(device))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (ex % 5000 == 0):
                    print("%"+str((ex/len(self.data_loader))*100), flush=True)
                    print("Loss =", loss)

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

    def buzz(self, guess_dataset, threshhold=.5):
        data = GuessDataset(guess_dataset)
        y_pred = self.model(data.feature)
        return [threshhold < y for y in y_pred]


if __name__ == '__main__':
    print("Buzzer Test")
    print()
    
    print("Initializing Model...")
    model = BuzzModel(4)

    agent = BuzzAgent(model)
    print("Initializing Agent...")

    agent.load_data_from_file("guess_dataset_sample.json")
    agent.train()

    with open("guess_dataset_sample.json") as file:
        data = json.load(file)
    print(agent.buzz(data))
