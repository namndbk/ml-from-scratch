import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PerceptronClassifier:
    
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

    def sgn(self, x, w):
        """
        Return sign of np.dot(x, w)
        if np.dot(x, w) >= 0 return 1 otherwise return -1
        @param:
            - x: narray, one or all data point
            - w: weight
        @type:
            - x: narray
            - w: narray
        @return: narray sign of one or all point data x
        @rtype: narray[int]
        """
        return np.sign(np.dot(x, w))

    def has_converged(self, X, y, w):
        """
        Compute and return total number of correct predictions
        @param:
            - X: all point data in data train
            - y: all label of data train
            - w: weight
        @type:
            - X: narray
            - y: narray
            - w: narray
        @return: total number of correct predictions
        @rtype: int 
        """
        return np.sum(self.sgn(X, w) == y)

    def fit(self, X, y):
        """
        Return weight and number of iteration after model coverged
        @param:
            - X: all point data in data train
            - y: all label of data train
        @type:
            - X: narray
            - y: narray
        @return:
            - w: weight after model coverged
            - it: number of iteration after model coverged
        @rtype:
            - w: narray
            - it: int
        """
        np.random.seed(42)
        # weight init
        w_init = np.random.randn(X.shape[1])
        self.w = w_init
        # N: number of data train
        N = X.shape[0]
        it = 0
        while it < self.max_iter:
            # Get random index list of all data point in data training 
            mix_id = np.random.permutation(N)
            for i in mix_id:
                # Get data point xi and label of it
                xi = X[i]
                yi = y[i]
                # if predict xi with weight w not true, update weight
                if self.sgn(xi, self.w) != yi:
                    self.w = self.w + self.lr * yi * xi
            # check converged condition. If number of incorrect label predicted by weight w <= tol * N, stop.
            # tol: threshold
            # N: number of data train
            if self.has_converged(X, y, self.w) <= self.tol*N:
                return self.w, it
            it += 1
        return self.w, it
    
    def predict(self, X_test):
        """
        Return label of data test based on weight self.w and sgn function
        @param: all data point of data test
        @type: narray
        @return: all label of data point in data test
        @rtype: narray
        """
        y_pred = self.sgn(X_test, self.w)
        return y_pred