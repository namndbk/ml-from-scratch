import numpy as np

from knn import KKNeighborsClassifier
from perceptron import PerceptronClassifier
from sigmoid import LogisticClassifier
from softmax import SoftmaxClassifier
import utils
import time

from sklearn.model_selection import train_test_split


def check_speed(lr=0.05, max_iter=10000, tol=1e-3):
    X_, y = utils.parse('data/emails.csv')
    DICT = utils.load_dict()

    X = np.zeros((len(X_), len(DICT)))
    for i in range(len(X_)):
        X[i] = utils.bag_of_word(X_[i], DICT)

    X = np.array(X[:, :1000])
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    logistic = LogisticClassifier(lr=lr, max_iter=max_iter, tol=tol)
    perceptron = PerceptronClassifier(lr=lr, max_iter=max_iter, tol=tol)
    softmax = SoftmaxClassifier(lr=lr, max_iter=max_iter, tol=tol)

    clfs = [logistic, perceptron, softmax]

    for clf in clfs:
        start_time = time.time()
        _, it = clf.fit(X_train, y_train)
        end_time = time.time()
        print("\tTime training of {} model = {}".format(str(clf.__class__.__name__), (end_time - start_time)))
        print("\tNumber of iteration = {}".format(int(it)))
        y_pred = clf.predict(X_test)
        print("\tAccuracy = {} %".format(np.sum(y_test == y_pred) * 100 / len(y_pred)))
        print("*" * 50)


if __name__ == "__main__":
    check_speed()