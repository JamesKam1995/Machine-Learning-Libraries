import numpy as np
from builtins import range
from random import shuffle

class SVM:
    def __init__(self):
        self.w = None
    
    def train(self, 
              X, 
              y, 
              learning_rate=1e-3, 
              reg=1e-5, 
              num_iters=100, 
              batch_size=200, 
              verbose=False):
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            loss, grad = self.loss_vectorized_(self.w, X_batch, y_batch, reg)
            loss_history.append(loss)

            self.w -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        pass

    def loss_naive_(self, w, X, y, reg):
        """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
        """
        dW = np.zeros(w.shape) #initiate the gradient as zeros

        #compute the loss and gradient
        num_classes = w.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        for i in range(num_train):
            scores = X[i].dot(W)
            correct_class_score = scores[y[i]]
            for j in range(num_classes):
                if j == y[i]:
                    continue
                margin = scores[j] - correct_class_score + 1
                if margin > 0:
                    loss += 1
                    dW[:, j] += X[i]
                    dW[:, y[i]] -= X[i]
        loss /= num_train
        loss += reg * np.sum(w*w)

        dW /= num_train
        dW = 2 * reg * w

    def loss_vectorized_(self, w, X, y, reg):
        N = len(y)
        Y_hat = X @ w 
        """
        lets say Y_hat = [2.5 1.0 3.1 2.1
                          3.1 4.3 2.1 1.1
                          2.4 2.6 3.5 4.1]
        
        the correct class is [0, 1, 2]

        we would like the y_hat_true to extract and reshape into
        [2.5
         4.3
         3.5]
        
        """

        y_hat_true = Y_hat[range(N), y][:, np.newaxis]
        margins = np.maximum(0, Y_hat - y_hat_true + 1)
        loss = margins.sum() / N - 1 + reg * np.sum(w**2)

        dW = (margins > 0).astype(int)
        dW[range(N), y] -= dW.sum(axis = 1)
        dW = X.T @ dW / N + 2 * reg * w

        return loss, dW

class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
