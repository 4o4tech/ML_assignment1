import h5py
import numpy as np
import os
import operator
import pandas as pd
import time


class LogisticRegressionOVR(object):
    def __init__(self):
        pass

    def softmax_fit(self, X, y, W, lr = 0.01, nepoches = 100, tol = 1e-5, batch_size = 10):
    W_old = W.copy()
    ep = 0 
    loss_hist = [self._softmax_loss(X, y, W)] # store history of loss 
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))
    while ep < nepoches: 
        ep += 1 
        mix_ids = np.random.permutation(N) # mix data 
        for i in range(nbatches):
            # get the i-th batch
            batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), N)] 
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= lr*softmax_grad(X_batch, y_batch, W) # update gradient descent # TODO
        loss_hist.append(self._softmax_loss(X, y, W))
        if np.linalg.norm(W - W_old)/W.size < tol:
            break 
        W_old = W.copy()
    return W, loss_hist 


    # def fit(self, X, y):
    #     X = np.insert(X, 0, 1, axis=1)
    #     self.w = []
    #     m = X.shape[0]

    #     for i in np.unique(y):
    #         y_copy = np.where(y == i, 1, 0)
    #         w = np.ones(X.shape[1])

    #         for _ in range(self.n_iter):
    #             z = X.dot(w)
    #             errors = y_copy - self._softmax(z)
    #             w += self.eta / m * errors.dot(X)
    #         self.w.append((w, i))
    #     return self

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
    def _softmax(self,weighted_input):
        # maxes = np.amax(weighted_input, axis=1)
        # maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(weighted_input)
        dist = e / np.sum(e, axis=1, keepdims=True)
        return dist
    def _softmax_grad(self,X,y,w):
        A = self._softmax(X.dot(y))
        getFir = range(X.shape[0])
        A[getFir,y] -= 1
        return X.T.dot(A) / X.shape[0]
    def _softmax_loss(self,X,y,w):
        A = self._softmax(X.dot(y))
        get0 = range(X.shape[0])
        return -np.mean(np.log(A[get0, y])) 


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
