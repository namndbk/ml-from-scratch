import numpy as np
import sys


_metrics = {"euclidean": "_l2_distance", "manhattan": "_l1_distance", "cityblock": "_cosine_similarity"}

def _cosine_similarity(u, v):
    if isinstance(u, list):
        u = np.array(u)
    if isinstance(v, list):
        v = np.array(v)
    return 1 - np.dot(u, v) / np.sqrt(np.sum(u**2) * np.sum(v**2))


def _l1_distance(u, v):
    if isinstance(u, list):
        u = np.array(u)
    if isinstance(v, list):
        v = np.array(v)
    return np.sum(np.abs(u - v))


def _l2_distance(u, v):
    if isinstance(u, list):
        u = np.array(u)
    if isinstance(v, list):
        v = np.array(v, list)
    return np.sqrt(np.sum((u - v) ** 2))


def cdist(X, Y, metric="euclidean"):
    M = len(X)
    N = len(Y)
    dm = np.empty((M, N))

    funct = getattr(sys.modules[__name__], _metrics[metric])
    
    for i in range(M):
        for j in range(N):
            dm[i][j] = funct(X[i], Y[j])
    return dm


if __name__ == "__main__":
    arr_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr_2 = np.array([[1, 2, 3], [1, 3, 2]])
    print(cdist(arr_1, arr_2))