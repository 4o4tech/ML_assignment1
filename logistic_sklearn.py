from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(C=1e5,random_state=0, solver='lbfgs',
#                          multi_class='multinomial').fit(X, y)
					
					
# clf.predict(X[:2, :])

# clf.predict_proba(X[:2, :]) 


# result = clf.score(X, y)



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
    train_data = data_train[0:15000]
    train_label = label_train[0:15000]

    test_data = data_test[0:2000]
    test_labels = label_test[0:2000]

    # w_init = np.random.randn(train_data.shape[1], classNum)

    # W, loss_hist= LogisticRegressionOVR().softmax_fit(train_data, train_label,w_init,batch_size=10, nepoches=100)

    clf = LogisticRegression(C=1e2,random_state=0, tol=0.0001,solver='lbfgs',
                         multi_class='ovr',max_iter=5000,n_jobs=4).fit(train_data, train_label)

    clf.predict(test_data[:2, :])
    clf.predict_proba(test_data[:2, :]) 

    result = clf.score(test_data, test_labels)
    print(result)

if __name__ == '__main__':
	main()
