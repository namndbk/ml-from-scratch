import numpy as np
from sklearn.utils import shuffle


class SimpleSVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


class SVM:

    kernels = {"linear": "_linear_kernel", "poly": "_polynomial_kernel", "rbf": "_gaussian_kernel"}

    def __init__(self, C=1.0, kernel="linear", degree=3, r=0.0, gamma="auto", lr=0.001, regularization_strength=1000, n_iters=5000, cost_threshold=0.001):
        """
        :params
            degree: thua so nhan cho kernel poly
            gamma: he so nhan chuan hoa
            lr: toc do hoc
            regularization_strength: he so chinh quy
            n_iters: so vong lap toi da
        """
        self.C = C
        if kernel not in list(self.kernels.keys()):
            self.kernel = "linear"
        else:
            self.kernel = kernel
        self.degree = degree
        self.regularization_strength = regularization_strength
        self.lr = lr
        self.r = r
        self.n_iters = n_iters
        self.gamma = gamma
        self.cost_threshold = cost_threshold
    
    def _linear_kernel(self, x, z):
        """
        k(x, z) = np.dot(x, z.T)
        """
        return np.dot(x, z.T)
    
    def _polynomial_kernel(self, x, z):
        return (self.r + self.gamma*np.dot(x, z.T))**self.degree
    
     def _gaussian_kernel(self, x, z):
        """
        k(x, z) = exp(-gamma*(norm_2(x, z)**2))
        """
        return np.exp(-self.gamma*(cdist(x, z)**2))

    def _sigmoid_kernel(self, x, z):
        """
        k(x, z) = tanh(gamma*np.dot(x, z) + r)
        """
        def tanh(s):
            return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
        return tanh(self.gamma*np.dot(x, z.T) + self.r)
    
    def compute_cost(self, W, X, Y):
        # calculate hinge loss (cong thuc cua svm)
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.regularization_strength * (np.sum(distances) / N)

        # tinh toan ham mat mat
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost
    
    def calculate_cost_gradient(self, W, X_batch, Y_batch):
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])
        else:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])

        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))

        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.regularization_strength * Y_batch[ind] * X_batch[ind])
            dw += di

        dw = dw/len(Y_batch)  # average
        return dw
    
    def sgd(self):
        max_epochs = 5000
        weights = np.zeros(self.X.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.001  # in percent
        # stochastic gradient descent
        for epoch in range(1, max_epochs):
            # shuffle data - buoc nay rat quan trong
            X, Y = shuffle(self.X, self.y)
            for ind, x in enumerate(X):
                ascent = self.calculate_cost_gradient(weights, x, Y[ind])
                weights = weights - (self.lr * ascent)

            # check sau mot so vong lap nhat dinh
            if epoch == 2 ** nth or epoch == max_epochs - 1:
                cost = self.compute_cost(weights, self.X, self.y)
                print("Epoch is: {} and Cost is: {}".format(epoch, cost))
                # check converged
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return weights
                prev_cost = cost
                nth += 1
        return weights
    
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # using kernel function transform data
        X_train = getattr(self, self.kernels[self.kernel])(X_train, X_train)

        self.X = X_train
        self.y = y_train

        self.W = self.sgd()
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = np.array([X_test])
        
        y_pred = np.array([])
        for i in range(X_test.shape[0]):
            y_ = np.sign(np.dot(X_test[i], self.W.T))
            y_pred = np.append(y_pred, y_)
        return y_pred        
