import h5py
import numpy as np
import os
import operator
import pandas as pd
import time


class LogisticRegressionOVR(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = []
        m = X.shape[0]

        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])

            for _ in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self._sigmoid(output)
                w += self.eta / m * errors.dot(X)
            self.w.append((w, i))
        return self

    def _predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]

    def predict(self, X):
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]

    def score(self, X, y):
        predict_value = self.predict(X)
        accur_val = sum(predict_value == y) 
        accur_ratio = accur_val/len(y)
        return  accur_ratio,predict_value

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return 1 if np.sum(x) > 0 else 0
    def _softmax_grad(self,X, y, W):
        A = softmax(X.dot(W))    # shape of (N, C)
        id0 = range(X.shape[0])  # number of train data
        A[id0, y] -= 1           # A - Y, shape of (N, C)


def exportCSV(test_labels, pred_labels):
    
    df = pd.DataFrame({"test_labels" :test_labels , "pred_labels" : pred_labels})
    df.to_csv("LogisticExportResult" + str(time.time()) +".csv", index=False)
    # numpy.savetxt("foo.csv", test_labels, delimiter=",")

def get_accuracy(test_labels, pred_labels):
    # 准确率计算函数   
    correct = np.sum(test_labels == pred_labels)  # 计算预测正确的数据个数
    n = len(test_labels)  # 总测试集数据个数
    accur = correct/n
    return accur


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
    train_data = data_train[0:3000]
    train_label = label_train[0:3000]

    test_data = data_test[0:500]
    test_labels = label_test[0:500]


    logi = LogisticRegressionOVR(n_iter=1000).fit(train_data, train_label)

    # accuracy, pred_labels = logi.score(train_data, train_label)

    accuracy, pred_labels = logi.score(test_data,test_labels)

    exportCSV(test_labels, pred_labels)

    print(" Accuracy: %f"%accuracy)

    # print(logi.score(test_data, test_labels))
    # pred_labels = logi.predict(test_data)

    # print(pred_labels)

    # accuracy = get_accuracy(test_labels,pred_labels)

    # exportCSV(test_labels, pred_labels)

    # print(" Accuracy: %f"%accuracy)




if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
