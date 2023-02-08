import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import torch.optim as optim
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt

from DataSet import WaterTemperatureDataSet
import pandas as pd
import datetime

from Trainer import trainLSTM, plot_prediction, plot_test_loss
from torch.autograd import Variable

from LSTMs import Simple, WaterLSTM



folder = "DataSets" 
WTA1_Dataset = pd.read_csv(folder + "\WTA1_DataSet.csv")

WTA1_Dataset.drop(["location"], axis=1, inplace=True)

Data = WaterTemperatureDataSet(WTA1_Dataset, target_cols=["Water_Avg"], numerical_cols=['Air_Min', 'Air_Max', "Air_Avg", 'season'], 
                                categorical_cols=[""], discriminator_col="TimeSeries")

train_iter, test_iter = Data.get_loaders()


input_size = 4
hidden_size = 2
seq_length = 365
num_classes = 1
num_layers = 365

iterator = train_iter._get_iterator()
next = next(iterator)


loss_function = nn.MSELoss()


model = WaterLSTM(input_size, hidden_size, seq_length, num_classes, num_layers)

simple = Simple()


epochs, total_loss, test_loss = trainLSTM(simple, train_iter, test_iter, learning_rate=0.01, epochs=100, print_every=10, filepath="Parameters/WLSTM1.pt")

path = "Figures/model1_training.png"
plot_test_loss(test_loss, 10, path)