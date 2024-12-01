ref : https://github.com/casper-hansen/Logistic-Regression-From-Scratch/blob/main/src/logistic_regression/model.py

import copy
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.losses = []
        self.train_accuracies = []
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, x, y, n_iters):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(n_iters):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        n_samples, n_features = x.shape
        dw = (1/n_samples) * np.dot(x.T, (difference))
        db = (1/n_samples) * np.sum(difference)
        dw = np.array([np.mean(grad) for grad in dw])
        return dw, db

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - self.lr * error_w
        self.bias = self.bias - self.lr * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)