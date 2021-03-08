import numpy as np


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SoftmaxClassifier:

    def __init__(self, lr=0.05, max_iter=10000, tol=1e-3):
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
            - w: narray
        """
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
    
    def _convert_label(self, y):
        """
        Convert 1d label to a maxtrix label: each row of this matrix coresponding to 1 row in y.
        In i-th row of y_new returned, only one non-zero element located in the y[i] -th position and 1
        e.g if y = [0, 1, 0, 1, 2] unique labels = [0, 1]
        => Y = [[1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
        @param: label of all data point in data train, 1d matrix a.g [1, 0, 2, 1]
        @type: narray, 1d array
        @return: matrix label converted from 1d matrix label
        @rtype: narray
        """
        # Get list of unique label
        labels = np.unique(y)
        C = len(labels)
        y_convert = np.zeros((y.shape[0], C))
        for i, c in enumerate(y):
            y_convert[i][c] = 1
        return y_convert
    
    def softmax(self, z):
        """
        Softmax activation function. Use at the output layer.
            g(z) = e^z / sum(e^z)
        @param: z = np.dot(x, w):
            - X: data point or matrix data point (1d or 2d)
            - w: weight, 2d
        @type: narray 1d or 2d
        @return: softmax value of each row in z
        @rtype: narray, dimension = dimension of input
        """
        z_prime = z - np.max(z, axis=-1, keepdims=True)
        return np.exp(z_prime) / np.sum(np.exp(z_prime), axis=-1, keepdims=True)
    
    def cross_entropy_loss(self, X, y, w):
        """
        return cross entropy cost function
        @param:
            X: matrix of data point
            y: label of data
            W: weight
        @type:
            X: narray (2d)
            y: narray (2d)
            W: array (2d)
        @return: cross entropy
        @rtype: float
        """
        y_hat = self.softmax(np.dot(X, w))
        return - np.sum(y * np.log(y_hat))

    def fit(self, X, y):
        """
        Training sigmoid model with Stochastic Gradient Descent
        @params:
            X: matrix data train
            y: label
        @type:
            X: narray
            y: array
        @return
            w: list of weight, final element is best weight
            it: number of iteration when model coverged
        @rtype:
            w: narray 
            it: int
        """
        # Convert label, from 1d array to 2d matrix
        y = self._convert_label(y)
        # d: number of features in data train
        # C: number of class
        # N: number of data point in data train
        d = X.shape[1]
        C = y.shape[1]
        N = X.shape[0]
        # init weight, dimension = (d, C)
        self.w = np.random.randn(d, C)
        it = 0
        while it < self.max_iter:
            # Get random index list of all data point in data training 
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[i].reshape(1, d)
                yi = y[i]
                # Compute softmax
                ai = self.softmax(np.dot(xi, self.w))
                # Update weight based on gradient descent
                self.w = self.w - self.lr * np.dot(xi.T, (ai - yi))
                it += 1
                # if this is first check, save w_check_after = self.w
                if it == 20:
                    w_check_after = self.w
                else:
                    if it % 20 == 0:
                        # if np.linalg.norm(self.w - w_check_after) < self.tol:
                        # check condition coverges, if the difference about loss function between two consecutive checks is small enough, stop
                        # else: update w_check_after = self.w
                        if abs(self.cross_entropy_loss(X, y, self.w) - self.cross_entropy_loss(X, y, w_check_after)) <= self.tol:
                            return self.w, it
                        else:
                            w_check_after = self.w
        return self.w, it

    def predict(self, X_test):
        """
        Return label of all point data in data test, 1d array
        @param: all point data in data test
        @type: narray (2d)
        @return: List of label
        @rtype: narray (1d)
        """
        return  np.argmax(self.softmax(np.dot(X_test, self.w)), axis=1) 