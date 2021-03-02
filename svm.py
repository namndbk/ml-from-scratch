import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import utils
import metrics


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

    # Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’
    kernels = {"linear": "_linear_kernel", "poly": "_polynomial_kernel", "rbf": "_gaussian_kernel"}

    def __init__(self, kernel="linear", degree=3, r=0.0, gamma="auto", lr=0.001, regularization_strength=1000, n_iters=5000, cost_threshold=0.01):
        """
        Contructors function
        @params
            degree: Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
            gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            lr: learning_rate
            regularization_strength: regular coefficient
            n_iters: max of iter
            cost_threshold
        @type:
            degree: int >= 1
            gamma: string, "auto" or "scale"
            lr: float between 0 and 1, not zero
            regularization_strength: numerical, positive
            n_iters: int
            cost_threshold: float between 0 and 1, not zero
        """
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
            """
            define tanh function
            """
            return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
        return tanh(self.gamma*np.dot(x, z.T) + self.r)
    
    def compute_cost(self, W, X, Y):
        """
        Compute and return cost function on all data
        @param:
            W: weight vector
            X: data points
            Y: label of data points
        @type:
            W: vector, array
            X: narray
            Y: narray
        @return: cost
        @rtype: float
        """
        # calculate hinge loss 
        # N number of data points
        N = X.shape[0]
        # compute distance from vector of data to W
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = self.regularization_strength * (np.sum(distances) / N)

        # compute loss function
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost
    
    def calculate_cost_gradient(self, W, X_batch, Y_batch):
        """
        Compute cost function gradient and return on one data
        @param:
            W: weight
            X_batch: batch of data, size = 1
            Y_batch: batch of label, size = 1
        @type:
            W: array
            X: narray
            Y: narray
        @return: cost gradient
        @rtype: array
        """
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])
        else:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])

        # compute distance from data to W
        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))

        # loop through distance, comput cost gradient
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.regularization_strength * Y_batch[ind] * X_batch[ind])
            dw += di

        dw = dw/len(Y_batch)  # average
        return dw
    
    def sgd(self):
        """
        Stochastic gradient descent for training svm
        """
        # max_epochs: max iteration
        max_epochs = 5000
        print(self.X.shape[1])
        weights = np.zeros(self.X.shape[1])
        # 2**nth number of iteations until converged
        nth = 0
        #prev_cost value of costfunction in prev step
        prev_cost = float("inf")
        cost_threshold = self.cost_threshold  # in percent
        # loop though, update weight based on stochastic gradient descent
        for epoch in range(1, max_epochs):
            # shuffle data
            X, Y = shuffle(self.X, self.y)
            for ind, x in enumerate(X):
                # comput cost gradient on one data
                ascent = self.calculate_cost_gradient(weights, x, Y[ind])
                # Update weights
                weights = weights - (self.lr * ascent)

            # check for stop conditions 
            if epoch % 100 == 0 or epoch == max_epochs - 1:
                # comput cost function over all data
                cost = self.compute_cost(weights, self.X, self.y)
                print("Epoch is: {} and Cost is: {}".format(epoch, cost))
                # Check the change of the cost function, stop when error is small enough
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return weights
                prev_cost = cost
                nth += 1
        return weights
    
    def fit(self, X_train, y_train):
        """
        pass training data and find best hyperplane for classify data
        @param:
            X_train: matrix of data points for training
            Y_train: matrix of data label
        @type:
            X_train: narray
            Y_train: narray
        @return: not return
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # using kernel function transform data
        # X_train = getattr(self, self.kernels[self.kernel])(X_train, X_train)

        self.X = X_train
        self.y = y_train

        # comput weight based on stochastic gradient descent
        self.W = self.sgd()
    
    def predict(self, X_test):
        """
        Predict and return label of data point test
        @param: matrix of data point for predict
        @type: narray
        @return: label of data
        @rtype: narray
        """
        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = np.array([X_test])
        y_pred = np.array([])
        for i in range(X_test.shape[0]):
            # predict each data points based on sign function
            y_ = np.sign(np.dot(X_test[i], self.W.T))
            y_pred = np.append(y_pred, y_)
        return y_pred


    def save_model(self, path):
        try:
            pickle.dump(self.W, open(path, "wb"))
            print("\tModel saved !")
        except:
            print("\tNot save model !")


if __name__ == "__main__":
    X, y = utils.load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
    model = SVM()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\tTesting in {} data point".format(len(y_pred)))
    print("\tAccuracy = {}%".format(metrics.accuracy_score(y_test, y_pred) * 100))
