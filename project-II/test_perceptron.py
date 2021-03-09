import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from perceptron import PerceptronClassifier
import utils

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    X_, y = utils.parse('data/emails.csv')
    DICT = utils.load_dict()

    X = np.zeros((len(X_), len(DICT)))
    for i in range(len(X_)):
        X[i] = utils.bag_of_word(X_[i], DICT)
    
    # Converts data points labeled 0 to label - 1
    # Because perceptron model (binary classification) only work with two label: -1 and 1
    for i, c in enumerate(y):
        if c == 0:
            y[i] = -1
    
    X = np.array(X[:, :1000])
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    model = PerceptronClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = np.unique(y_train)

    print("\tAccuracy: {}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\tClassification report: \n{}".format(classification_report(y_test, y_pred, labels=labels)))


if __name__ == "__main__":
    main()