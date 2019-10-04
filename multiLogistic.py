import h5py
import numpy as np
import os
import operator
import pandas as pd
import time
from tqdm import tqdm 
import matplotlib as plt

class SoftmaxRegression(object):
    def __init__(self):
        pass

    def softmax_fit(self, X, y, W, lr = 0.1, n_iter = 100, tol = 1e-5, batch_size = 20):
        W_old = W.copy()
        loss_hist = [self._softmax_loss(X, y, W)] # store history of loss 
        dims = X.shape[0]
        nbatches = int(np.ceil(float(dims)/batch_size))

        for niter in tqdm(range(n_iter)):

            mix_ids = np.random.permutation(dims) # mix data 
            for i in range(nbatches):
                # get the i-th batch
                batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), dims)] 
                X_min_batch, y_min_batch = X[batch_ids], y[batch_ids]
                W -= lr*self._softmax_grad(X_min_batch, y_min_batch, W) 
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

    def score(self, X, y):
        predict_value = self.predict(X)
        accur_val = sum(predict_value == y) 
        accur_ratio = accur_val/len(y)
        return  accur_ratio,predict_value

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x): 
        return 1 if np.sum(x) > 0 else 0
    
    def _softmax_grad(self,X,y,w):
        hx = softmax(X.dot(w))
        getLen = range(X.shape[0])
        hx[getLen,y] -= 1
        return X.T.dot(hx) / X.shape[0]
      
    def _softmax_loss(self,X, y, W):
        A = softmax(X.dot(W))
        id0 = range(X.shape[0])
        return -np.mean(np.log(A[id0, y]))

def softmax(weighted_input):
        # maxes = np.amax(weighted_input, axis=1)
        # maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(weighted_input)
        dist = e / np.sum(e, axis=1, keepdims=True)
        return dist

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def pred(W, X):
    A = softmax(X.dot(W))
    return np.argmax(A, axis = 1)

def exportCSV(test_labels, pred_labels, accuracy, loss_hist):
    
    df = pd.DataFrame({"test_labels" :test_labels , "pred_labels" : pred_labels})
    df['accuracy'] = accuracy
    df['loss_hist'] = pd.Series(loss_hist)

    df.to_csv("LogisticExportResult" + str(time.time()) +".csv", index=False)
    # numpy.savetxt("foo.csv", test_labels, delimiter=",")

def get_accuracy(test_labels, predi_labels):
    # 准确率计算函数   
    correct = np.sum(test_labels == predi_labels)  # 计算预测正确的数据个数
    n = len(test_labels)  # 总测试集数据个数
    accur = correct/n
    return accur

# def predi(W,X):
#     return softmax(X.dot(W))




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

    classNum = 10
    # train_data = data_train[0:30000]
    # train_label = label_train[0:30000]

    data_test = data_test[0:2000] # just using smalle sclice of test data
    # label_test_slice = label_test[0:2000] 

    w_init = np.random.randn(data_train.shape[1], classNum)

    W, loss_hist= SoftmaxRegression().softmax_fit(data_train, label_train, w_init, n_iter=100, batch_size=20)

    predi_labels = pred(W,data_test)

    # print(predi_labels)

    accuracy = get_accuracy(label_test, predi_labels)


    # accuracy, pred_labels = logi.score(test_data,test_label)
    print(" Accuracy: %f"%accuracy)

    exportCSV(label_test, predi_labels,accuracy,loss_hist)


    

    # print(logi.score(test_data, test_labels))
    # pred_labels = logi.predict(test_data)




if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
