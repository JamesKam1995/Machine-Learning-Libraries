#ref: https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch

import numpy as np

class LinearRegression():
    def __init__(self, weight, bias, learning_rate=0.01, iteration=1000) -> None:
        self.learning_rate = learning_rate
        self.iteration = iteration 
        self.weight = weight
        self.bias = bias

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        self.weight = np.zeros(n_feature)
        self.bias = 0 

        for _ in range(self.iteration):
            y_pred = np.dot(X, self.weight) + self.bias

            dw = (1 / n_sample) * np.dot(X.T, (y_pred - y))
            db = (1 / n_sample) * np.sum(y_pred - y)

            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred
