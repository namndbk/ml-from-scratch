import numpy as np


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class LogisticClassifier:

    def __init__(self, lr=0.05, max_iter=10000, tol=1e-3, use_bias=True):
        """
        @param:
            - lr: learning rate
            - max_iter: max number of iteration
            - tol: the tolerance for the stopping criteria
            - w: weights
            - use_bias: use or not use bias
        @type:
            - lr: float
            - max_iter: int
            - tol: float (maximum <= 1e-2)
            - w: narray[float]
            - use_bias: bool
        """
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.use_bias = use_bias

    def sigmoid(self, z):
        """
        sigmoid function
        @param: z - vector
        @type: array[float]
        @return: vector
        @rtype: array[float]
        """
        return 1.0 / (1.0 + np.exp(-z))


    def cross_entropy(self, X, y, w, bias=None):
        """
        return cross entropy cost function
        @param:
            X: matrix of data point or one data point
            y: label of data
            W: weight
        @type:
            X: narray 
            y: narray
            W: array
        @return: cross entropy
        @rtype: float
        """
        if bias:
            y_hat = self.sigmoid(np.dot(X, w) + bias)
        else:
            y_hat = self.sigmoid(np.dot(X, w))
        return -np.sum(y * np.log(y_hat))


    def fit(self, X, y):
        """
        Training sigmoid model based on Stochastic Gradient Descent
        @params:
            X: matrix data train
            y: label
        @type:
            X: narray (2d)
            y: array (1d)
        @return
            w: list of weight, final element is best weight
            it: number of iteration when model coverged
        @rtype:
            w: narray (1d)
            it: int
        """
        np.random.seed(42)
        X = np.array(X)
        if self.w is None:
            self.w = np.random.randn(X.shape[1])
        if self.use_bias is True:
            self.bias = np.random.randn()
        N = X.shape[0]
        it = 0
        check_w_after = 20
        # weight after check
        w_check_after = None
        bias_check_after = None
        while it < self.max_iter:
            # Get random index list of all data point in data training 
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[i]
                yi = y[i]
                if self.use_bias:
                    zi = self.sigmoid(np.dot(xi, self.w) + self.bias)
                else:
                    zi = self.sigmoid(np.dot(xi, self.w))
                # Update weight via gradient descent
                # w_new = W[-1] + self.lr *  ((yi - zi) * xi + regular*W[-1])
                self.w = self.w + self.lr * (yi - zi) * xi
                if self.use_bias is True:
                    self.bias = self.bias + self.lr * (yi - zi)
                it += 1
                if it % check_w_after == 0:
                    # first check, save weights into w_check_after, not check converged
                    if it == 20:
                        w_check_after = self.w
                        if self.use_bias:
                            bias_check_after = self.bias
                    # Check converged conditions, If small enough, stop
                    else:
                        if self.use_bias:
                            if abs(self.cross_entropy(X, y, self.w, self.bias) - self.cross_entropy(X, y, w_check_after, bias_check_after)) < self.tol:
                                return self.w, self.bias, it
                            else:
                                bias_check_after = self.bias
                        else:
                            if abs(self.cross_entropy(X, y, self.w) - self.cross_entropy(X, y, w_check_after)) < self.tol:
                                return self.w, it
                        w_check_after = self.w
        if self.use_bias:
            return self.w, self.bias, it
        else:
            return self.w, it
    

    def predict(self, X):
        # Predict data
        if self.use_bias:
            y_pred = self.sigmoid(np.dot(X, self.w) + self.bias)
        else:
            y_pred = self.sigmoid(np.dot(X, self.w))
        for i, c in enumerate(y_pred):
            if c < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        return y_pred