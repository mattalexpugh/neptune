__author__ = 'matt'

import time
import platform # fix this!

from sklearn.externals import joblib


if platform.system() == 'Darwin':
    DIR_CLASSIFIERS = "/Users/matt/Documents/MPhil/classifiers/"
else:
    DIR_CLASSIFIERS = "/home/matt/MPhil/classifiers/"


def load_saved_classifier(classifier_path):
    return joblib.load(classifier_path)


class PersistantClassifier():

    def __init__(self):
        self._classifier = None
        self._metric = None

    def save(self, fname_extra=None):
        this_name = self.state

        if fname_extra is not None:
            prefix = fname_extra.replace(' ', '_').split('.')[0]
            this_name = prefix + "_" + this_name

        fp = DIR_CLASSIFIERS + this_name

        joblib.dump(self, fp)

        return fp

    def classify(self, data):
        return self.raw_classifier.predict(data)

    @property
    def raw_classifier(self):
        return self._classifier

    @property
    def training_data(self):
        return self._training_data

    @training_data.setter
    def training_data(self, data):
        """
        Saves references to whatever data was used to train, for example: frame numbers.
        """
        self._training_data = data

    @property
    def ground_truth(self):
        return self._ground_truth

    @ground_truth.setter
    def ground_truth(self, ground_truth):
        """
        Saves the ground truth used to build this
        """
        self._ground_truth = ground_truth

    @property
    def video_path(self):
        return self._video_path

    @video_path.setter
    def video_path(self, video_path):
        self._video_path = video_path

    @property
    def identifier(self):
        return ""

    @property
    def metric(self):
        return self._metric

    @property
    def state(self):
        timestamp = str(time.time()).split('.')[0]
        file_name = "_".join([self.__class__.__name__, self.identifier, timestamp])

        return file_name
