import numpy as np
from collections import defaultdict
import h5py
import os
import operator
import pandas as pd
import time


class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5, weights='distance', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _distance(self, data1, data2):
        """1: Manhattan, 2: Euclidean"""    
        if self.p == 1:
            return sum(abs(data1 - data2))          
        elif self.p == 2:
            return np.sqrt(sum((data1 - data2)**2))
        raise ValueError("p not recognized: should be 1 or 2")

    def _compute_weights(self, distances):
        if self.weights == 'uniform':
            return [(1, y) for d, y in distances]
        elif self.weights == 'distance':
            matches = [(1, y) for d, y in distances if d == 0]
            return matches if matches else [(1/d, y) for d, y in distances]
        raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

    def _predict_one(self, test):
        distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))
        weights = self._compute_weights(distances[:self.n_neighbors])
        weights_by_class = defaultdict(list)
        for d, c in weights:
            weights_by_class[c].append(d)
        return max((sum(val), key) for key, val in weights_by_class.items())[1]

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def score(self, X, y):
        return sum(1 for p, t in zip(self.predict(X), y) if p == t) / len(y)





def main():
    with h5py.File('./data/train/images_training.h5','r') as H:
        data_train = np.copy(H['datatrain'])
    with h5py.File('./data/train/labels_training.h5','r') as H:
        label_train = np.copy(H['labeltrain'])

    # using H['datatest'], H['labeltest'] for test dataset.
    with h5py.File('./data/test/images_testing.h5','r') as H:
        data_test = np.copy(H['datatest'])
    with h5py.File('./data/test/labels_testing_2000.h5','r') as H:
        label_test = np.copy(H['labeltest'])




    neighbor = KNeighborsClassifier().fit(data_train[0:3000], label_train[0:3000])

    # print(neighbor.score(data_train[0:3000], label_train[0:3000]))
    print(neighbor.score(data_test[0:500], label_test[0:500]))

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))