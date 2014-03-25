import svm
import knn

__author__ = 'matt'
__doc__ = """

Standard class prefixes:

    Prefix      Definition                      Info
    ==============================================================================
    Vxxx        Video object                    Works for Videos
    Ixxx        Image object                    Works on Images
    xSVM        Support Vector Machine          Uses SVMs for classification
    xKNN        k-Nearest-Neighbour             Uses kNN
    xDTR        Decision Tree
"""


def get_instantiated_classifiers(funcName):
    classifiers = {
        'svms': svm.get_svcs_for_all_kernels(metric=funcName),
        'knns': knn.get_knns_for_all_algorithms(metric=funcName, n=5)
    }

    # Manually stick this on for now
    classifiers['svms']['LinearSVC'] = svm.VLinearSVCSubstrateClassifier(metric=funcName)

    return classifiers


CLASSIFIER_MAP = {
    'svm': svm.CLASSIFIER_MAP,
    'knn': knn.CLASSIFIER_MAP
}

