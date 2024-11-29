import numpy as np

class SVM:
    def __init__(self, learning_rate, lambda_param=0.01, n_iter=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b =  None
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def loss_naive_(self, w, X, y, reg):
        pass

    def loss_vectorized_(self, w, X, y, reg):
        pass