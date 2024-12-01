#ref: https://github.com/casper-hansen/Logistic-Regression-From-Scratch/blob/main/src/logistic_regression/model.py

import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.lossess = []
        self.train_accuracies = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            self.compute_gradients(X, y, pred)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.lossess.append(loss)

    def predict(self, X):
        x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
        prob = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in prob]
    
    def compute_loss(self, y_true, y_pred):
        #binaru cross entropy_loss
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)
    
    def compute_gradients(self, X, y, pred):
        n_samples, n_features = X.shape
        dw = (1 / n_samples) * np.dot(X.T, (pred - y))
        db = (1 / n_samples) * np.sum((pred - y))

        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db

        return self.w, self.b

    
    def _sigmoid(self, X):
        return np.array([self._sigmoid_function(value) for value in X])

    def _sigmoid_function(self, x):
        if x > 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        
        else:
            z = np.exp(x)
            return 1 / (1 + z)
        
