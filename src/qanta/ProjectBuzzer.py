import torch
import torch.nn as nn
import numpy as np
import json
import pandas

from torch.utils.data import Dataset, DataLoader
from qanta.ProjectDataLoader import *
from qanta.util import give_confidence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM = 50

#=======================================================================================================
# Data Classes
#=======================================================================================================
class Sample:
    def __init__(self, guess_data, df=None):
        self.gid = None
        self.x = None
        self.y = None

        gid = []
        features = []
        guess = guess_data["guess"]
        score = guess_data["score"]
        kguess_scores = guess_data["kguess_scores"]
        q_length = guess_data["question_nonzero_tokens"]/412
        q_text = guess_data["full_question"]

        # guess id for embed
        gid.append(guess_data["guessId"])

        # score
        features.append(score)
        # diff between top 2 scores
        features.append(kguess_scores[0] - kguess_scores[1])
        # ave diff between 1st score and others
        temp = [kguess_scores[0] - x for x in kguess_scores[1:]]
        features.append(sum(temp)/len(temp))
        # question length
        features.append(q_length)
        # score X length
        features.append(score * q_length)
        # disambiguation
        if '(' in guess and ')' in guess:
            features.append(float(guess[guess.find("(")+1:guess.find(")")] in q_text))
        else:
            features.append(0.0)
        # wiki links
        #features.append(give_confidence(guess, q_text, df))
        features.append(0)

        self.gid = np.array(gid)
        self.x = np.array(features)
        if "label" in guess_data:
            self.y = np.array([float(guess_data["label"])])
        else:
            self.y = np.array([0.0])

class GuessDataset(Dataset):
    def __init__(self, guess_dataset, df):
        self.num_features = 8
        self.gid = None
        self.feature = None
        self.label = None
        self.num_samples = 0

        dataset = []
        for guess_data in guess_dataset["buzzer_data"]:
            ex = Sample(guess_data, df)
            dataset.append(ex)
            self.num_samples += 1

        gid = np.stack([ex.gid for ex in dataset])
        self.gid = torch.from_numpy(gid.astype(np.long))
        features = np.stack([ex.x for ex in dataset])
        self.feature = torch.from_numpy(features.astype(np.float32))
        label = np.stack([ex.y for ex in dataset])
        self.label = torch.from_numpy(label.astype(np.float32))

    def __getitem__(self, index):
        return self.gid[index], self.feature[index], self.label[index]

    def __len__(self):
        return self.num_samples


#=======================================================================================================
# Model and Training
#=======================================================================================================
class BuzzModel(nn.Module):
    def __init__(self, num_features, vocab_size, num_hidden_units=50, nn_dropout=.5):
        super(BuzzModel, self).__init__()
        self.num_hidden_units = num_hidden_units
        self.guess_embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.linear1 = nn.Linear(EMBED_DIM + num_features, self.num_hidden_units)
        self.linear2 = nn.Linear(self.num_hidden_units, 1)
        self.buzzer = nn.Sequential(self.linear1, nn.ReLU(), nn.Dropout(nn_dropout), self.linear2)

    def forward(self, x):
        ids = x[0]
        features = x[1]
        embeds = self.guess_embed(ids)
        inputs = torch.cat((embeds.squeeze(1), features), 1)
        #print(inputs)
        #print(inputs.size())
        y_pred = torch.sigmoid(self.buzzer(inputs))
        return y_pred

class BuzzAgent():
    def __init__(self, model, links_file_location, learnrate=0.01):
        self.learnrate = learnrate
        self.links_df = None #pandas.read_csv(links_file_location, dtype={'a': str})

        if (model):
            self.model = model.to(device)
            self.criterion = nn.BCELoss()
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learnrate)
        else:
            print("Buzzer waiting for model to load", flush=True)

        self.data_loader = None

    def load_data(self, guess_dataset, batch_size=1):
        data = GuessDataset(guess_dataset, self.links_df)
        self.dataset = guess_dataset
        self.num_features = data.num_features
        self.data_loader = DataLoader(dataset=data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    def load_data_from_file(self, guess_dataset_file, batch_size=1):
        with open(guess_dataset_file) as file:
            self.load_data(json.load(file), batch_size)

    def train(self, num_epochs=100, output=True, save_loc=None):
        for epoch in range(num_epochs):
            if output:
                print("epoch " + str(epoch), flush=True)
            for ex, (ids, inputs, labels) in enumerate(self.data_loader):
                y_pred = self.model((ids.to(device), inputs.to(device)))
                loss = self.criterion(y_pred, labels.to(device))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (ex % 5000 == 0) and output:
                    print("%"+str((ex/len(self.data_loader))*100), flush=True)
                    print("Loss =", loss)

            print("Accuracy: " + str(self.evaluate(self.dataset)), flush=True)

        if (save_loc):
            self.save_model({"epochs":num_epochs}, save_loc)

    def evaluate(self, guess_dataset):
        data = GuessDataset(guess_dataset, self.links_df)
        with torch.no_grad():
            y_pred = self.model((data.gid.to(device), data.feature.to(device)))
            y_pred_cls = y_pred.round()
            acc = y_pred_cls.eq(data.label.to(device)).sum() / float(data.label.to(device).shape[0])
            return acc

    def buzz(self, guess_dataset, will_round=True):
        data = GuessDataset(guess_dataset, self.links_df)
        with torch.no_grad():
            y_pred = self.model((data.gid.to(device), data.feature.to(device)))

            if (will_round):
                y_pred_cls = y_pred.round()
                return y_pred_cls.bool()
            else:
                return y_pred

    # Save the model and its associated metadata 
    def save_model(self, metadata, save_location):
        pickle.dump({"model": self.model, "metadata":metadata}, open(save_location, "wb+"))
        print("Saved model to: \"" + save_location + "\"", flush=True)

    # Save model and its associated metadata 
    def torch_save_model(self, save_location):
        torch.save(self.model, save_location)
        print("torch saved model to: \"" + save_location + "\"", flush = True)

    # Load the model and its associated metadata
    def load_model(self, location):
        load = pickle.load(open(location,'rb'))
        self.model = load["model"]
        self.model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learnrate)
        print("Loaded model from \"" + location + "\"", flush=True)

    # Load the model and its associated metadata
    def torch_load_model(self, file_name):
        self.model = torch.load(file_name, map_location=device)
        print("Loaded model from: \"" + file_name + "\"", flush = True)

if __name__ == '__main__':
    print("Buzzer Test")
    print()
    
    print("Initializing Model...")
    model = BuzzModel(7, 100)

    print("Initializing Agent...")
    agent = BuzzAgent(model, "wiki_links.csv")

    data = json.load(open("guess_dataset_sample.json"))

    print("Pre-train:")
    print(agent.buzz(data))

    agent.load_data_from_file("guess_dataset_sample.json")
    agent.train(output=False)

    print("Post-train:")
    print(agent.buzz(data))
    print(agent.evaluate(data))
    