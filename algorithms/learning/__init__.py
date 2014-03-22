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

CLASSIFIER_MAP = {
    'svm': svm.CLASSIFIER_MAP,
    'knn': knn.CLASSIFIER_MAP
}
