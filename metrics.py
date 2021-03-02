import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy_score(y_true, y_pred):
    """
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    return float(correct / y_true.shape[0])


def confusion_matrix(y_true, y_pred):
    N_classes = np.unique(y_true).shape[0]
    cm = np.zeros((N_classes, N_classes))
    for n in range(y_true.shape[0]):
        cm[y_true[n], y_pred[n]] += 1
    return cm


def f1_score(y_true, y_pred):
    N = np.unique(y_true)
    precision = []
    recall = []
    for i in N:
        precision.append(float(np.sum(y_true == i & y_pred == i) / np.sum(y_pred == i)))
        recall.append(float(np.sum(y_true == i & y_pred == i) / np.sum(y_true == i)))
    pr = np.sum(np.array(precision)) / len(N)
    rc = np.sum(np.array(recall)) / len(N)
    f1 = 2 * pr * rc / (pr + rc)
    print(f1)
    print(precision, recall)


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 1, 2])
    print(confusion_matrix(y_true, y_pred))
    