import numpy as np

class LinearRegression():
    def __init__(self, learning_rate=0.01, iteration=1000, weight, bias) -> None:
        self.learning_rate = learning_rate
        self.iteration = iteration 
        self.weight = weight
        self.bias = bias

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        self.weight = np.zeros(n_feature)
        self.bias = 0 

        for _ in range(iteration)
        pass

    def predict():
        pass

    def gradient_descent():
        pass
