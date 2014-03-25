from sklearn.neighbors import KNeighborsClassifier

from algorithms.learning.persist import PersistantClassifier, STR_PARAM_DELIM


__author__ = 'matt'


KNN_ALGORITHM_TYPES = ['ball_tree', 'kd_tree']


def get_knns_for_all_algorithms(n=5, metric=None):
    return {k: VKNNSubstrateClassifier(n_neighbors=n, algorithm=k, metric=metric) for k in KNN_ALGORITHM_TYPES}


class VKNNSubstrateClassifier(PersistantClassifier):

    def __init__(self, n=5, algorithm='auto', metric=None):
        self._algorithm = algorithm
        self._n = n
        self._metric = metric if metric is not None else 'undefined'
        self._classifier = KNeighborsClassifier(n_neighbors=n,
                                         algorithm=algorithm)

    @property
    def identifier(self):
        return STR_PARAM_DELIM.join([self.algorithm, "n" + str(self._n), self.metric])

    @property
    def algorithm(self):
        return self._algorithm


CLASSIFIER_MAP = {
    'knn': VKNNSubstrateClassifier
}
