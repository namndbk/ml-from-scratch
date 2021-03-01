import numpy as np
import utils
from sklearn.model_selection import train_test_split
# from tree import DecisionTreeID3
import math
import pandas as pd
from tree import DecisionTreeClassifier

class MultinomialNB():
    def __init__(self, alpha=1.0):
        """
        alpha - hej so lam tro xac suat
        prob - ma tran xac suat cua tung tu thuoc moi class- prob[i, j] - xac suat tu thu j roi vao class i
        prob_c - luu xac suat cua class
        __classes - danh sach cac nhan
        """
        self.alpha = alpha
        self.__classes = []
        self.prob = None
        self.prob_c = None
    def fit(self, X_train, y_train):
        # lay ra danh sach class co trong data train
        self.__classes = list(set(y_train))
        # ma tran count luu so tu xuat hien count[i, j] - so lan tu j xuat hien trong class i
        count = np.zeros((len(self.__classes), X_train.shape[1]))
        len_class = np.zeros(len(self.__classes))
        # khoi tao ma tran prob_c prob
        self.prob_c = np.zeros(len(self.__classes))
        self.prob = np.zeros((len(self.__classes), X_train.shape[1]))
        # lap qua du lieu
        for i, c in enumerate(y_train):
            # dem so lan class c xuat hien
            self.prob_c[int(c) - 1] += 1
            len_class[int(c) - 1] += np.sum(X_train[i])
            count[int(c) - 1] += X_train[i]
        # tinh xac suat cua class
        self.prob_c = self.prob_c/X_train.shape[0]
        # duyet qua tung class, dem xem trong moi class do co nhung tu nao, xuat hien bao nhieu lan => dua thanh xac suat
        for c in self.__classes:
            self.prob[int(c) - 1] = (count[int(c) - 1] + self.alpha)/(len_class[int(c) - 1] + self.alpha*X_train.shape[1])


    def predict(self, X_test):
        # bien y_pred luu lai nhan
        y_pred = []
        # duyet qua tung diem du lieu test
        for i in range(X_test.shape[0]):
            _max = -99999.0
            _c = 0
            # duyet qua tung class
            for c in self.__classes:
                # tinh xac suat du lieu do roi vao class c la bao nhieu
                _prob = math.log(self.prob_c[int(c) - 1])
                for j in range(X_test.shape[1]):
                    if X_test[i][j] != 0:
                        _prob += math.log(self.prob[int(c) - 1][j])*X_test[i][j]
                # so sanh xac suat voi cac class khac, luu lai class lam xac suat co gia tri lon nhat
                if _prob > _max:
                    _max = _prob
                    _c = c
            y_pred.append(_c)
        return y_pred

    def accuracy(self, y_test, y_pred):
        count = 0
        for i in range(len(y_test)):
            if str(y_test[i]) == str(y_pred[i]):
                count += 1
        print('Acc: %.2f' % (count*100/len(y_test)), end=' %')


if __name__ == '__main__':
    X_, y = utils.parse('data/emails.csv')
    # utils.build_dict(X_)
    DICT = utils.load_dict()
    X = np.zeros((len(X_), len(DICT)))
    for i in range(len(X_)):
        X[i] = utils.bag_of_word(X_[i], DICT)
    for i in range(len(X)):
        for j in range(len(X[0])):
            if X[i, j] > 0:
                X[i, j] = 1
            else:
                X[i, j] = 0
    words = [x for x in DICT]
    # X = pd.DataFrame(X)
    # y = pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    print(y_test)
    # model = MultinomialNB(0.001)
    # model.fit(X_train, y_train)
    # tree = DecisionTreeClassifier(5, words=words)
    # tree.fit(X_train[:1000], y_train[:1000], is_root=True)
    # y_pred = tree.predict(X_test)
    # print(np.sum(y_true == y_pred))
    # tree = DecisionTreeClassifier()
    # tree.fit(X_train, y_train)
    # print(X_test.shape[1])
    # y_pred = tree.predict(X_test)
    # print(np.sum(y_true == y_pred))