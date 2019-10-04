import h5py
import numpy as np
import os
import operator
import pandas as pd
import time
from tqdm import tqdm 
import matplotlib.pyplot as plt
from numpy import linalg as ll

class SoftmaxRegression(object):
	def __init__(self):
		pass

	def softmax_fit(self, X, y, W, lr = 0.01,  n_iter= 100, tol = 1e-6, batch_size = 20):
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

def draw_loss(loss_hist):
	plt.plot(loss_hist[10:])
	plt.xlabel('Number of n_iteration', fontsize = 16)
	plt.ylabel('Lost', fontsize = 16)
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.show() 

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
			
			sigRecon = np.diag(Sigma[:num_sv]) # cut first 8 
			reconMat = np.dot(U[:,:num_sv],sigRecon).dot(Vt[:num_sv,:]) # 截取 U, Vt ; reconvert to matrix
			
			reconMat = np.reshape(reconMat, (784))
			
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
	# train_data = data_train[0:30000]
	# train_label = label_train[0:30000]

	data_test = data_test[0:2000] # just using smalle sclice of test data
	# label_test_slice = label_test[0:2000] 

	w_init = np.random.randn(data_train.shape[1], classNum)

	# W, loss_hist= SoftmaxRegression().softmax_fit(data_train, label_train, w_init, n_iter=2000, batch_size=100)
	W, loss_hist= SoftmaxRegression().stocashtic_gradient_descent(data_train, label_train,w_init , n_iter=10)


	predi_labels = pred(W,data_test)

	# print(predi_labels)

	accuracy = get_accuracy(label_test, predi_labels)


	# accuracy, pred_labels = logi.score(test_data,test_label)
	print(" Accuracy: %f"%accuracy)

	# exportCSV(label_test, predi_labels,accuracy,loss_hist)

	draw_loss(loss_hist)
	

	# print(logi.score(test_data, test_labels))
	# pred_labels = logi.predict(test_data)

def imageCom():
	with h5py.File('./data/train/images_training.h5','r') as H:
		data_train = np.copy(H['datatrain'])

	# train_data = data_train[0:30000]
	# train_label = label_train[0:30000]

	imgCom = ImageCompress()

	data_train = data_train.reshape((data_train.shape[0], 28, 28))

	# print(data_train.shape[0])

	com_data = imgCom.img_compress(data_train,num_sv=8, img_num=data_train.shape[0])
	com_np_data = np.array(com_data)
	# # com_np_data.shape[0]
	imgCom.compress_dataset(com_np_data.shape[0],com_np_data)

def test():
	with h5py.File('./data/test/images_compress.h5','r') as H:
		data_train = np.copy(H['compress_datatest'])
	print(data_train.shape)


if __name__ == "__main__":
	start_time = time.time()
	main()
	# imageCom()
	# test()

	print("--- %s seconds ---" % (time.time() - start_time))
