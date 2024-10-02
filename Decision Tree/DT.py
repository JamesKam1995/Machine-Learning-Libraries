import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right= None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value if not None

class DecisionTree:
    def __init__(self, min_samples_split = None, max_depth = 100, n_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y) #expand the tree 

    def predict(self, X):
        pass

    def information_gain(self, y, X_column, threshold):
        pass
    
    def _grow_tree(self, X, y, depth=0):

        #############################################################################
        # Grow the tree. Steps:                                                     #
        # 1. check the stopping criteria. If there are no nodes, just return the    #
        #    most common value                                                      #
        # 2. find the best split                                                    #
        # 3. create the child node recursively                                      #
        #############################################################################

        n_sample, n_feature = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_sample <= self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        
        feature_index = np.random_choice(n_feature, self.n_features, replace = False)

        #Find the best split
        best_feature, best_threshold = self._best_split(X, y, feature_index)

        #create child node
        left_idxs, right_idxs = self._split(X[: best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)
        

