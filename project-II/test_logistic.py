import numpy as np
from scipy.spatial.distance import cdist

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sigmoid import LogisticClassifier
import utils

import pickle


def main():
    X_, y = utils.parse('data/emails.csv')
    DICT = utils.load_dict()

    X = np.zeros((len(X_), len(DICT)))
    for i in range(len(X_)):
        X[i] = utils.bag_of_word(X_[i], DICT)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    model = LogisticClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\tAccuracy: {}%".format(np.sum(y_pred == y_test) / len(y_pred) * 100))
    print("\tClassification report: \n{}".format(classification_report(y_test, y_pred)))


if __name__ == "__main__":
    main()