import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

from LSTMs import WaterLSTM

import pandas as pd


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def trainLSTM(neuralNetwork, train_iter, test_iter, learning_rate, epochs, filepath, show_training=True, print_every=5): 
    device = get_default_device()

    neuralNetwork.train()
    neuralNetwork.to(device)

    optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=learning_rate)

    total_loss, shape_loss, temporal_loss = [], [], []
    loss_function = nn.MSELoss()
    test_loss_evolution = []

    for epoch in range(epochs):
        for series, target in train_iter: 
            series.to(device)
            target.to(device)

            optimizer.zero_grad()
            predictions = neuralNetwork(series)
            loss = loss_function(predictions, target)
            
            loss.backward()
            optimizer.step()

        total_loss.append(loss)

        if show_training:
            if epoch % print_every == 0: 
                test_loss = evaluate_training(neuralNetwork, test_iter, loss_function)
                test_loss_evolution.append(test_loss)
                print('epoch ', epoch, ' loss ', loss.item())


    clean_loss = []
    for tensor in total_loss: 
        clean_loss.append(tensor.detach())

    export_training(neuralNetwork, filepath)

    return epochs, clean_loss, test_loss_evolution

def evaluate_training(model, test_set, loss_function):
    total_loss = 0
    for test, target in test_set:
        prediction = model(test)
        total_loss += loss_function(prediction, target)

    return total_loss.detach()



def export_training(model, filepath): 
    torch.save(model.state_dict(), filepath)

def import_parameters(filepath, model): 
    model.load_state_dict(torch.load(filepath))



def plot_prediction(prediction, target): 
    days = pd.date_range(start="01-01-2030", end="31-12-2030")
    plt.plot(days, prediction- 273, c='green', label="predicted")
    plt.plot(days, target - 273, c='red', label="truth")
    plt.title("Model prediction vs actual data")

    plt.ylabel("Water Tempredature")
    plt.xlabel("Days of the year")
    plt.legend()
    plt.show()

def plot_test_loss(test_loss, print_every, path): 
    n_epochs = len(test_loss)
    epochs = [i * print_every for i in range(n_epochs)]
    plt.plot(epochs, test_loss)
    plt.xlabel("epochs")
    plt.ylabel("Test loss")
    plt.savefig(path)
