import numpy as np
from scipy.spatial.distance import cdist


class KKNeighborsClassifier:

    """
    Idea of KNN algorithm:
        - Given a specific dataset, for each row is a n-D dimension vector and the label.
        - Pre-processing dataset (normalization, standardization) into same scale (optional).
        - For any new point in predicting phase, the algorithm finds the distance between that point and all other
            points in training set (Manhattan, Euclidean, Cosine similarity).
        - Base on K hyper-parameter, the algorithm will find K nearest neighbor and classify that point into
            which class.
    """

    _metrics = {"euclidean": "_l2_distance", "manhattan": "_l1_distance", "cosine": "_cosine_similarity"}

    def __init__(self, n_neighbors=3, metric="eulidean"):
        self.n_neighbors = n_neighbors
        if metric not in self._metrics.keys():
            self.metric = "euclidean"
        else:
            self.metric = metric
    
    def _l1_distance(self, X):
        """
        Return manhattan distance of X_test versus all other points in data train
        narray[i, j] = distance from X[i] to self.X[j] based on manhattan distance
        @param: all points data in data test
        @type: narray[int]
        @return: manhattan distance of X_test versus all other points in data train
        @rtype: narray[int]
        """
        return cdist(X, self.X, "cityblock")

    def _l2_distance(self, X):
        """
        Return euclidean distance of X_test versus all other points in data train
        narray[i, j] = distance from X[i] to self.X[j] based on euclidean distance
        @param: all points data in data test
        @type: narray[int]
        @return: euclidean distance of X_test versus all other points in data train
        @rtype: narray[int]"""
        return cdist(X, self.X, "euclidean")
    
    def _cosine_similarity(self, X):
        """
        Return cosine similarity of X_test versus all other points in data train
        narray[i, j] = distance from X[i] to self.X[j] based on  cosine similarity
        @param: all points data in data test
        @type: narray[int]
        @return: cosine similarity distance of X_test versus all other points in data train
        @rtype: narray[int]
        """
        return cdist(X, self.X, "cosine")
    
    def fit(self, X_train, y_train):
        """
        Save all points and label data in train data. Not return
        @param:
            - X_train: all points data in data train
            - y_train: all label of each point data in data train
        @type:
            - X_train: narray[int]
            - y_train: narray[int]
        """
        self.X = X_train
        self.y = y_train
        # Get list of unique labels
        self.classes = np.unique(self.y)
    
    def predict(self, X_test):
        """
        Return all label of each point data in data test
        @param: all points data in data test
        @type: narray[int]
        @return: all label of each point data in data test
        @rtype: narray[int]
        """
        function = getattr(self, self._metrics[self.metric])
        # compute distance or similarity of X_test versus all other points in data train based on self.metric metric
        distances = function(X_test)
        distances = np.argsort(distances, axis=1)
        k_nearest = distances[:, :self.n_neighbors]
        # Get number point data of data train
        N = len(self.X)
        # Get list labels of all point data in data train nearest each point data in data set
        labels = self.y[k_nearest]
        # Declare results variable, save all label of data test returned
        results = []
        for tag in labels:
            label, count = np.unique(tag, return_counts=True)
            results.append(label[np.argmax(count)])
        return np.array(results)