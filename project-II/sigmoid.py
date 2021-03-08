import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import utils
from knn import KKNeighborsClassifier
from perceptron import PerceptronClassifier

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class LogisticClassifier:

    def __init__(self, lr=0.001, max_iter=10000, tol=1e-3):
        """
        @param:
            - lr: learning rate
            - max_iter: max number of iteration
            - tol: the tolerance for the stopping criteria
            - w: weights
        @type:
            - lr: float
            - max_iter: int
            - tol: float (maximum <= 1e-2)
            - w: narray[int]
        """
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None

    def sigmoid(self, z):
        """
        sigmoid function
        @param: z - vector
        @type: array[float]
        @return: vector
        @rtype: array[float]
        """
        return 1.0 / (1.0 + np.exp(-z))


    def cross_entropy(self, X, y, w):
        """
        return cross entropy cost function
        @param:
            X: matrix of data point
            y: label of data
            W: weight
        @type:
            X: narray[float]
            y: narray
            W: array
        @return: cross entropy
        @rtype: float
        """
        y_hat = self.sigmoid(np.dot(X, w))
        return -np.sum(y * np.log(y_hat))


    def fit(self, X, y):
        """
        training sigmoid model with Stochastic Gradient Descent
        @params:
            X: matrix data train
            y: label
            w_init: weight init
            lr: learning rate
            tol:
            max iter: maximum iteration
            regurlar: regular coefficient
        @type:
            X: narray[float]
            y: array
            w_init: array[float]
            lr: float
            tol:float
            max_iter: int
            regular: float
        @return
            w: list of weight, final element is best weight
            max_iter: number of iteration when model coverged
        @rtype:
            w: List[array]
            max_iter: int
        """
        np.random.seed(42)
        X = np.array(X)
        if self.w is None:
            self.w = np.random.randn(X.shape[1])
        N = X.shape[0]
        it = 0
        check_w_after = 20
        # weight after check
        w_check_after = None
        while it < self.max_iter:
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[i]
                yi = y[i]
                zi = self.sigmoid(np.dot(xi, self.w))
                # Update weight via gradient f
                # w_new = W[-1] + self.lr *  ((yi - zi) * xi + regular*W[-1])
                self.w = self.w + self.lr *  (yi - zi) * xi
                it += 1
                if it % check_w_after == 0:
                    # first check, save weights into w_check_after, not check converged
                    if it == 20:
                        w_check_after = self.w
                    # Check converged conditions, If small enough, stop
                    else:
                        if abs(self.cross_entropy(X, y, self.w) - self.cross_entropy(X, y, w_check_after)) < self.tol:
                            return self.w, it
                        else:
                            # if not converged, save weight into w_check_after and loop continue
                            w_check_after = self.w
        return self.w, it
    

    def predict(self, X):
        # Predict data test
        y_pred = self.sigmoid(np.dot(X, self.w))
        for i, c in enumerate(y_pred):
            if c < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        return y_pred

data = load_iris()
X = data.data
y = data.target
X_train, y_train = [], []
for i in range(len(y)):
    if y[i] == 1:
        y_train.append(y[i])
        X_train.append(X[i])
    elif y[i] == 0:
        y_train.append(0)
        X_train.append(X[i])
X = np.array(X_train)
y = np.array(y_train)

if __name__ == "__main__":
    X_, y = utils.parse('data/emails.csv')
    DICT = utils.load_dict()
    X = np.zeros((len(X_), len(DICT)))
    for i in range(len(X_)):
        X[i] = utils.bag_of_word(X_[i], DICT)
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    print(np.array(X_train).shape)
    model = KKNeighborsClassifier()
    _, it = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(np.sum(y_pred == y_test) / len(y_pred))