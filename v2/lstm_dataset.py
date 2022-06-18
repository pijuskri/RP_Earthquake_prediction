from math import ceil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, num_classes)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.sigm(out)
        return out


class TimeSeriesDataset(Dataset):
    def __init__(self, transform=None):
        xy = pd.read_pickle('./datasets/dataset.pkl')
        self.x = xy.drop(columns=['label'])
        self.y = xy['label']
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x.iloc[item], self.y.iloc[item]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class DownSample:
    def __init__(self, factor, signal):
        self.factor = factor
        self.signal = signal

    def __call__(self, sample):
        x, y = sample
        res = []
        for val in x:
            val = val[::self.factor]
            val = val[:self.signal]
            res.append(val)
        x = torch.tensor(np.array(res)).float().to(device)
        y = torch.tensor(y).float().to(device)
        return x, y


class LossCounter:
    def __init__(self, batch=1):
        self.batch = batch
        self.loss = 0.0
        self.labels = []
        self.predictions = []

    def update(self, loss, labels, outputs):
        self.loss += loss
        self.labels.append(labels)
        self.predictions.append(outputs)

    def get_results(self):
        return torch.cat(self.labels), torch.cat(self.predictions)

    def get_loss(self):
        loss = self.loss / self.batch
        self.loss = 0.0
        return loss

    def get_acc(self):
        labels = torch.cat(self.labels)
        predictions = torch.cat(self.predictions)
        accuracy = (torch.round(predictions) == labels).sum().item() / labels.shape[0]
        self.labels = []
        self.predictions = []
        return accuracy
