from sklearn import svm

from algorithms.learning.persist import PersistantClassifier


__author__ = 'matt'

# These are the kernel types in sklearn.svm, rbf is default
# SVM_KERNEL_TYPES = ['linear', 'poly', 'rbf', 'sigmoid']
SVM_KERNEL_TYPES = ['rbf']


def get_svcs_for_all_kernels(metric=None):
    return {k: VSVMSubstrateClassifier(k, metric) for k in SVM_KERNEL_TYPES}


class VSVMSubstrateClassifier(PersistantClassifier):

    def __init__(self, kernel='rbf', metric=None):
        self._kernel = kernel
        self._metric = metric if metric is not None else 'undefined'
        self._classifier = svm.SVC(kernel=kernel)  # Multi-classification SVM

    @property
    def identifier(self):
        return "_".join([self.kernel_type, self.metric])

    @property
    def kernel_type(self):
        return self._kernel


class VLinearSVCSubstrateClassifier(VSVMSubstrateClassifier):
    """
    No support for different kernel types
    """

    def __init__(self, metric=None):
        self._classifier = svm.LinearSVC()
        self._metric = metric if metric is not None else 'undefined'
        self._kernel = "LinearSVC"


CLASSIFIER_MAP = {
    'svmrbf': VSVMSubstrateClassifier,
    'svmlin': VLinearSVCSubstrateClassifier
}
