#ref : https://github.com/casper-hansen/Logistic-Regression-From-Scratch/blob/main/src/logistic_regression/model.py

import copy
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self, lr = 0.1, n_iters = 1000, lambda_ = 0.01, penalty = 'l2'):
        self.losses = []
        self.train_accuracies = []
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_ = lambda_
        self.penalty = penalty
        self.w = None
        self.b = None

    def fit(self, x, y, n_iters):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.w = np.zeros(x.shape[1])
        self.b = 0

        for i in range(n_iters):
            x_dot_weights = np.matmul(self.w, x.transpose()) + self.b
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
        # Derivative of binary cross entropy
        difference = y_pred - y_true
        n_samples, n_features = x.shape
        db = np.mean(difference)
        dw = np.matmul(x.transpose(), difference)
        dw = np.array([np.mean(grad) for grad in dw])

        # Apply regularization if specified
        if self.penalty == 'l2':
            dw += (self.lambda_ / n_samples) * self.w
        elif self.penalty == 'l1':
            dw += (self.lambda_ / n_samples) * np.sign(self.w)
        
        return dw, db

    def update_model_parameters(self, error_w, error_b):
        self.w = self.w - self.lr * error_w
        self.b = self.b - self.lr * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.w.transpose()) + self.b
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