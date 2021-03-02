import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def sigmoid(z):
    """
    sigmoid function
    @param: z - vector
    @type: array[float]
    @return: vector
    @rtype: array[float]
    """
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy(X, y, w):
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
    y_hat = sigmoid(np.dot(X, w))
    return -np.sum(y * np.log(y_hat))


def logistic_sigmoid(X, y, w_init, lr, tol=1e-5, max_iter=100000, regular=0.0):
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
    w = [w_init]
    N = X.shape[0]
    it = 0
    check_w_after = 20
    while it < max_iter:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i]
            yi = y[i]
            zi = sigmoid(np.dot(xi, w[-1]))
            # cap nhat trong so theo cong thuc dao ham tai tung diem du lieu
            w_new = w[-1] + lr *  ((yi - zi) * xi + regular*w[-1])
            it += 1
            if it % check_w_after == 0:
                # check dieu kien hoi tu
                if abs(cross_entropy(X, y, w_new) - cross_entropy(X, y, w[-check_w_after])) < tol:
                    return w, it
            w.append(w_new)
    return w, max_iter


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
    w, it = logistic_sigmoid(X, y, w_init=np.random.randn(4), lr=0.05, tol=1e-4)
    print(sigmoid(np.dot(X, w[-1])))
    model = LogisticRegression()
    model.fit(X, y)
    print(w[-1], it)
    print(model.coef_)