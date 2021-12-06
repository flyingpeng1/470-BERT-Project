# Developed from Code from Alex Jian Zheng
import torch
import torch.nn as nn
import numpy as np
import json

from torch.utils.data import Dataset, DataLoader
from qanta.ProjectDataLoader import *

#=======================================================================================================
# Data formatting
#=======================================================================================================
class Sample:
    def __init__(self, guess_data, vocab):
        """
        Create a new example

        label -- The label (0 / 1) of the example
        words -- The words in a list of "word:count" format
        vocab -- The vocabulary to use as features (list)
        """
        # first chunk of feature vector is one-hot encoding of answer
        answer_tensor = vocab.encode_from_indexes([guess_data["guessId"]])[0]
        x = answer_tensor.detach().to("cpu").numpy()
        # other features added to end
        x = np.append(x, guess_data["score"])

        self.x = x
        self.y = float(guess_data["label"])

class GuessDataset(Dataset):
    def __init__(self, vocab):
        self.vocab = vocab
        # num_features = answer_space_size + num_additional_features
        self.num_features = self.vocab.num_answers + 1
        self.feature = None
        self.label = None
        self.num_samples = 0

    def initialize(self, buzzer_data_file):
        dataset = []
        data = json.load(buzzer_data_file)
        for guess_data in data["buzzer_data"]:
            ex = Sample(guess_data, self.vocab)
            dataset.append(ex)

        features = np.stack([ex.x for ex in dataset])
        label = np.stack([np.array([ex.y]) for ex in dataset])

        self.feature = torch.from_numpy(features.astype(np.float32))
        self.label = torch.from_numpy(label.astype(np.float32))
        self.num_samples = len(self.label)
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
    def __init__(self, num_features):
        super(LogRegModel, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
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
    def __init__(self, model, vocab, learnrate=0.01):
        self.model = model
        self.vocab = vocab
        self.data = None
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learnrate)

    def load_data(self, buzzer_data_file, batch=1):
        data = GuessDataset(vocab)
        data.initialize(buzzer_data_file)
        self.data = DataLoader(dataset=data,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)

    def step(self, epoch, ex, model, inputs, labels):
        self.optimizer.zero_grad()
        predictions = model(inputs)
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()

    def train(self, num_epochs, model, inputs, labels):
        # Iterations
        for epoch in range(num_epochs):
            for ex, (inputs, labels) in enumerate(self.data):
                self.step(epoch, ex, model, inputs, labels)

        # acc = model.evaluate(test)
        # print("Accuracy: %f" % acc)
        # torch.save(model.state_dict(), "trained_model.th")
