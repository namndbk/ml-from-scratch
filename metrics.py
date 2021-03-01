import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    return float(correct / y_true.shape[0])


def confusion_matrix(y_true, y_pred):
    N_classes = np.unique(y_true).shape[0]
    cm = np.zeros(N, N)
    for n in range(y_true.shape[0]):
        cm[y_true[n], y_pred[n]] += 1
    return cm


def f1_score(y_true, y_pred):
    N = np.unique(y_true)
    precision = []
    recall = []
    for i in N:
        precision.append(float(np.sum(y_true == i & y_pred == i) / np.sum(y_pred == i)))
        recall.append(float(np.sum(y_true == i & y_pred == i) / np.sum(y_true == 1)))