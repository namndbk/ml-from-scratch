import numpy as np


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class LogisticClassifier:

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
        N = X.shape[0]
        it = 0
        check_w_after = 20
        # weight after check
        w_check_after = None
        while it < self.max_iter:
            # Get random index list of all data point in data training 
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[i]
                yi = y[i]
                zi = self.sigmoid(np.dot(xi, self.w))
                # Update weight via gradient descent
                # w_new = W[-1] + self.lr *  ((yi - zi) * xi + regular*W[-1])
                self.w = self.w + self.lr * (yi - zi) * xi
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
        # Predict data
        y_pred = self.sigmoid(np.dot(X, self.w))
        for i, c in enumerate(y_pred):
            if c < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        return y_pred