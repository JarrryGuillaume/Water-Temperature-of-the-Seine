import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


class Simple(nn.Module): 
    def __init__(self, hidden_layers=178):
        super(Simple, self).__init__()
        self.hidden_layers = hidden_layers
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 365)
        self.l3 = nn.Linear(365, 1)

    def forward(self, x): 
        x = torch.tanh(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        output = self.l3(x)
        return output

class WaterLSTM(nn.Module): 
    def __init__(self, input_size, hidden_size, seq_length, num_classes, num_layers):
        super(WaterLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.seq_length = seq_length
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.ReLu = nn.ReLU()

    def forward(self, x): 
        h_0 = Variable(torch.zeros(x.size(0), self.hidden_size)) 
        c_0 = Variable(torch.zeros(x.size(0), self.hidden_size))

        print(x, h_0, c_0)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        out = self.ReLu(hn)
        out = self.fc_1(out)
        out = self.ReLu(out)
        out = self.fc(out)
        return out