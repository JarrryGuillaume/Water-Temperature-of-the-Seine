import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pandas as pd

class WaterTemperatureDataSet(nn.Module):
    def __init__(self, data, categorical_cols, target_cols, numerical_cols, discriminator_col):
        self.data = data

        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_cols = target_cols
        self.discriminator_col = discriminator_col

    def preprocessor(self): 
        Y = self.data[self.target_cols + [self.discriminator_col]]
        X = self.data.drop(self.target_cols, axis=1)

        series = X["TimeSeries"].unique()

        features, targets = [], []

        for serie_num in series:
            extract = X.loc[X[self.discriminator_col] == serie_num, self.numerical_cols]
            target = Y.loc[Y[self.discriminator_col] == serie_num, self.target_cols]

            features.append(extract)
            targets.append(target)

        X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, test_size=0.2, shuffle=False)

        return X_train, y_train, X_test, y_test


    def frame_series(self, X, y):
        nb_features,  nb_obs = len(X), len(X[0])
        features, targets, y_hist = [], [], []

        for feature in X: 
            features.append(torch.FloatTensor(feature.to_numpy()))

        for target in y: 
            targets.append(torch.FloatTensor(target.values))

        targets = torch.cat(targets)
        features = torch.cat(features)

        return TensorDataset(features, targets)
        

    def get_loaders(self, batch_size=365):
        X_train, y_train, X_test, y_test = self.preprocessor()

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)


        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_iter, test_iter


def visualize_tensor(serie):
    print("number of points: ", len(serie))
    days = np.arange(0, 365)
    print(serie)

    plt.scatter(days, serie[:, 1] - 273, marker="." )
                  
    plt.fill_between(days, serie[:, 1] - 273, serie[:, 0] - 273, color='red', alpha=0.5)

    plt.fill_between(days, serie[:, 1] - 273, serie[:, 2] - 273, color='blue', alpha=0.5)

    plt.title(f"Unknown year")
    plt.xlabel("date")
    plt.ylabel("Series temperature in °C")
    plt.show()

def visualize_target(target): 
    days = np.arange(0, 365)
    plt.plot(days, target)
    plt.title(f"Unknown year")
    plt.xlabel("date")
    plt.ylabel("Target temperature in °C")
    plt.show()

