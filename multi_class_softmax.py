import h5py
import numpy as np
import os
import operator
import pandas as pd
import time
from tqdm import tqdm 
import matplotlib.pyplot as plt
from numpy import linalg as ll
from random import randrange


class SoftmaxRegression(object):
	def __init__(self):
		pass

	def softmax_fit(self, X, y, W, lr = 0.1,  n_iter= 100, tol = 1e-7, batch_size = 20):
		loss_hist = [self._softmax_loss(X, y, W)] # store history of loss 
		dims = X.shape[0]
		nbatches = int(np.ceil(float(dims)/batch_size))
		W_pre = W.copy()

		for niter in tqdm(range(n_iter)):

			mix_ids = np.random.permutation(dims) # mix data 
			for i in range(nbatches):
				# get the i-th batch
				batch_ids = mix_ids[batch_size*i:min(batch_size*(i+1), dims)] 
				X_min_batch, y_min_batch = X[batch_ids], y[batch_ids]
				W -= lr*self._softmax_grad(X_min_batch, y_min_batch, W) 
			loss_hist.append(self._softmax_loss(X, y, W))
			if np.linalg.norm(W - W_pre)/W.size < tol:
				break 
			W_pre  = W.copy()

		return W, loss_hist 
	# def fit(self, X, y,W, lr = 0.01, n_iter= 100):
	# 	  loss_hist = []
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

def softmax(X):
	X_exp = np.exp(X)
	partition = X_exp / np.sum(X_exp, axis=1, keepdims=True)
	return partition


def pred(W, X):
	y_hat = softmax(X.dot(W))
	return np.argmax(y_hat,axis = 1)


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

def draw_loss(loss_hist):
	plt.plot(loss_hist)
	plt.xlabel('Number of n_iteration', fontsize = 16)
	plt.ylabel('Lost', fontsize = 16)
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.show() 
	# num  = str(randrange(100))
	# plt.savefig('foo_'+num+'.jpg' )

def predi_output(predi_labels):
	with h5py.File("./output/predicted_labels.h5", "w") as f:
		f.create_dataset('label_predi', data=predi_labels)
		f.close()

# def predi(W,X):
#     return softmax(X.dot(W))

class ImageCompress(object):
	"""docstring for ImageCompress"""
	def __init__(self):
		pass

	def img_compress(self,data_train,num_sv=8, img_num=1):
		'''
			compress image 
		'''

		compress_img = []
		
		for i in range(img_num):
	#         plt.imshow(data_train[i], cmap=plt.get_cmap('gray')) 
	#         classNum = label_train[i]
	#         plt.title("class " + str(classNum) + ":" + clothClass[classNum] )    
			U, Sigma, Vt = ll.svd(data_train[i])
			
			sigRecon = np.matrix(np.diag(Sigma[:num_sv])) # cut first 8 
			

			reconMat = data_train[i].T.dot(U[:,:num_sv]).dot(sigRecon.getI())
			# reconMat = np.dot(U[:,:num_sv],sigRecon).dot(Vt[:num_sv,:]) # 截取 U, Vt ; reconvert to matrix
			
			# reconMat = np.reshape(reconMat, (784))
			
			compress_img.append(reconMat)
			
		return  compress_img
#         print(reconMat)
#         plt.imshow(reconMat, cmap=plt.get_cmap('gray'))
#         plt.show()  
#         print(reconMat)

	def compress_dataset(self,img_num,com_np_data):

		print(com_np_data.shape)
		with h5py.File("./data/test/images_compress.h5", "w") as f:
			f.create_dataset('compress_datatest', data=com_np_data)
			f.close()
			# f['compress_datatest'] = np.copy(com_np_data)


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
	# train_data = data_train[0:20000]
	# train_label = label_train[0:20000]

	# data_train = imageCom(data_train)
	# print(data_train.shape)

	data_test = data_test[0:2000]# just using smalle sclice of test data
	# label_test_slice = label_test[0:2000] 

	w_init = np.random.randn(data_train.shape[1], classNum)

	init_predi_labels = pred(w_init,data_test)
	init_accuracy = get_accuracy(label_test, init_predi_labels)
	print(" Initail Accuracy: %f"%init_accuracy)



	W, loss_hist= SoftmaxRegression().softmax_fit(data_train, label_train, w_init, n_iter=100, batch_size=20)

	predi_labels = pred(W,data_test)

	accuracy = get_accuracy(label_test, predi_labels)


	predi_output(predi_labels)

	print("Accuracy: %f"% accuracy)

	# w_init = np.random.randn(train_data.shape[1], classNum)
	# W, loss_hist= SoftmaxRegression().softmax_fit(train_data, train_label, w_init, n_iter=1000, batch_size=50)

	# print(w_init.shape, W.shape)


	# print(logi.score(test_data, test_labels))
	# pred_labels = logi.predict(test_data)

def imageCom(data_train):

	# train_data = data_train[0:30000]
	# train_label = label_train[0:30000]

	imgCom = ImageCompress()

	data_train = data_train.reshape((data_train.shape[0], 28, 28))

	# print(data_train.shape[0])

	com_data = imgCom.img_compress(data_train,num_sv=8, img_num=data_train.shape[0])
	com_np_data = np.array(com_data)
	
	return com_np_data

	# # com_np_data.shape[0]
	# imgCom.compress_dataset(com_np_data.shape[0],com_np_data)

def test():
	with h5py.File('./data/test/images_compress.h5','r') as H:
		data_train = np.copy(H['compress_datatest'])
	print(data_train.shape)


if __name__ == "__main__":
	start_time = time.time()
	main()
	
	# test()

	print("--- %s seconds ---" % (time.time() - start_time))
