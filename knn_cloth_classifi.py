import h5py
import numpy as np
import os
import operator
import pandas as pd
import time

'''
KNN algorithm for minst fashion classification


Create by Jimze  2019/9/??

'''


with h5py.File('./data/train/images_training.h5','r') as H:
    data_train = np.copy(H['datatrain'])
with h5py.File('./data/train/labels_training.h5','r') as H:
    label_train = np.copy(H['labeltrain'])

# using H['datatest'], H['labeltest'] for test dataset.
with h5py.File('./data/test/images_testing.h5','r') as H:
    data_test = np.copy(H['datatest'])
with h5py.File('./data/test/labels_testing_2000.h5','r') as H:
    label_test = np.copy(H['labeltest'])


# using H['datatest'], H['labeltest'] for test dataset.

# print(data_train.shape,label_train.shape)

# 欧拉距离公式
def d_euc(x, y):
    d = np.sqrt(np.sum(np.square(x - y)))
    return d


# vote the most  probabolity
def majority_voting(class_count):
	# 多数表决函数
	sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
	return sorted_class_count


def knn_classify(test_data, train_data, labels,k):
	'''
	knn classify for cloth classification
	base on shiyanluo.com  example

	I am rookie (crying T^T

	'''
	distances = np.array([])

	for i_data in train_data:

		d = d_euc(test_data, i_data) # 特麻烦一个个 比较距离， cry again
		distances = np.append(distances, d) # collect each distance, ranking min later

	sorted_distance_index = distances.argsort() # get the  after sort index	
	sorted_distance = np.sort(distances)
	r = (sorted_distance[k] + sorted_distance[k-1])/2

	class_count = {}

	for i in range(k):
		vote_label = labels[sorted_distance_index[i]]
		class_count[vote_label] = class_count.get(vote_label, 0) + 1

	final_label = majority_voting(class_count)
	return final_label[0][0], r


def get_accuracy(test_labels, pred_labels):
    '''
    	准确率计算函数
    '''	
    correct = np.sum(test_labels == pred_labels)  # 计算预测正确的数据个数
    n = len(test_labels)  # 总测试集数据个数
    accur = correct/n
    return accur


def exportCSV(test_labels, pred_labels, accur):
	'''
		export predict data into csv file, 
		做个记录 
	'''
	df = pd.DataFrame({"test_labels" :test_labels , "pred_labels" : pred_labels,"Accuracy":accur})
	df.to_csv("kNNExportResult" + str(time.time()) +".csv", index=False)
	# numpy.savetxt("foo.csv", test_labels, delimiter=",")


def svd_pca(data, k):
	"""
	Reduce DATA using its K principal components.
	"""
	data = data.astype("float64")
	data -= np.mean(data, axis=0)
	U, S, V = np.linalg.svd(data, full_matrices=False)
	return U[:,:k].dot(np.diag(S)[:k,:k])

def main():
	
	train_data = data_train[0:3000]
	train_label = label_train[0:3000]

	test_data = data_test[0:500]
	test_labels = label_test[0:500]

	pred_labels = np.array([])

	# train_data= svd_pca(train_data, 10)
	# test_data= svd_pca(train_data, 10)

	for i_test_data in test_data:
		final, r = knn_classify(i_test_data, train_data, train_label, k=6)

		pred_labels =np.append(pred_labels,final)

	# print(pred_labels)
	# print(test_label)
	accuracy = get_accuracy(test_labels,pred_labels)

	exportCSV(test_labels, pred_labels, accuracy)

	print(" Accuracy: %f"%accuracy)




if __name__ == "__main__":
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))
