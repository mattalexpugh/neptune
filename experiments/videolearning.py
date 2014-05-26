__author__ = 'matt'


import time
import datetime
import logging

import billiard as mp
import numpy as np

from algorithms.learning.persist import DIR_CLASSIFIERS, STR_PARAM_DELIM,\
    load_saved_classifier
from core.gui.util.containers import VideoFrameConverter
from data.io import CSVEXPSubstrateWriter
from structures.holders import GTHVideoSubstrates
from structures.video import create_capture
from util.system import get_files_in_dir

log = logging.getLogger(__name__)


class EXPMLVidSubstrateBase(object):

    def __init__(self, gt_path, video_path,
                 metric_func,
                 init=True, classifier=None):
        self._classifier = classifier
        self._capture = create_capture(video_path)
        self._converter = VideoFrameConverter(self._capture)
        self._ground_truth = GTHVideoSubstrates(gt_path, self._capture)
        self._classes = self._ground_truth.classes_present
        self._metric = metric_func

        self._training_frames = []  # Stores all frames used in training
        self._x = []                # frames n_samples * n_features
        self._y = []                # labels n_samples

        self._classifier.ground_truth = gt_path
        self._classifier.video_path = video_path

        if init:
            self.train()

    def __iter__(self):
        """
        Provides a mechanism for introspective iteration, returns a tuple
        of the next frame for given class k, and k itself:

            (np.ndarray: frame, int: class_id)

        Usage:

            for frame, klass in EXPMLVidSubstrateBase: ...
        """
        for k in self.classes:
            log.info("Started class " + str(k))

            for f in self.get_training_frames(k):
                self.converter.set_current_frame(f)
                self.converter.capture_next_frame()

                yield self.converter.current_frame, k

            log.info("Finished class " + str(k))

    def _process_frame(self, frame):
        """
        Default frame processor, simply uses self._metric function pointer on
        the entire frame passed through. Converts the representation to a
        n x 1 vector for speed by default. Returns processed data in the form
        of a generator.
        """
        data = self._metric(frame)

        if isinstance(data, np.ndarray):
            data = data.ravel()

        yield data

    def train(self):
        """
        Trains the classifier, looks to _process_frame to deliver the metric-affected
        version of the whole frame, or subsections thereof. This method doesn't
        really care if _process_frame returns 1 or more results, it just collates
        the training vectors X and Y before passing to sklearn for fitting.
        """

        for frame, k in self:
            for data in self._process_frame(frame):
                self._x.append(data)
                self._y.append(k)

        log.info("Beginning fitting for " + self.classifier.identifier)

        # Keep a reference to the frames we have chosen for this iteration
        self.classifier.training_data = self._training_frames
        self.classifier.raw_classifier.fit(np.array(self._x), np.array(self._y))

    def get_training_frames(self, klass, n=None):
        """
        Generator method that records the training frames selected in this iteration,
        and yields the frames sequentially for given klass. Parameter n is optional,
        and may indeed be deprecated. It is to limit the number below 10% of all frames,
        and was only relevant for Gabor full-frame image representations.
        """
        frames, _ = self.ground_truth.get_sample_frames_for_class(klass, n=n)
        self._training_frames.extend(frames)  # Keep a reference to them

        for f in frames:
            yield f

    @property
    def classifier(self):
        return self._classifier

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def converter(self):
        return self._converter

    @property
    def classes(self):
        return self._classes


class EXPMLVidSubstratePatch(EXPMLVidSubstrateBase):
    """
    Base class for patch-based video learning. To be extended with concrete
    implementations. Params:

        m       Integer         Patch width
        n       Integer         Patch height
    """

    def __init__(self, gt_path, video_path,
                 metric_func, m, n,
                 init=True, classifier=None):
        super(EXPMLVidSubstratePatch, self).__init__(
            gt_path, video_path, metric_func, init, classifier)
        self._m = m
        self._n = n

    def _process_frame(self, frame):
        patches = self.converter.get_frame_mxn_patches_list(self.m, self.n)
        patches = self._prepare_patches(patches)

        for patch in patches:
            data = self._metric(patch)

            if isinstance(data, np.ndarray):
                data = data.ravel()

            yield data

    def _prepare_patches(self, patches):
        return patches

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n


class EXPMLVidSubstratePatchBWRanged(EXPMLVidSubstratePatch):
    """
    This experiment looks at training the classifier based on a patched approach.
    Each patch of each elective frame is analysed for its mean-illumination value.
    A range-based selector is implemented, based on input parameter i_range, that
    determines the +- difference around the median intensity value that is considered
    valid.
    """

    def __init__(self, gt_path, video_path,
                 metric_func, m, n, i_range,
                 init=True, classifier=None):
        super(EXPMLVidSubstratePatchBWRanged, self).__init__(
            gt_path, video_path, metric_func, m, n, init, classifier)

        self._i_max = 255
        self._i_max_half = self._i_max / 2
        self._i_valid_range = i_range
        self._i_valid_max = self._i_max_half + self._i_valid_range
        self._i_valid_min = self._i_max_half - self._i_valid_range

    def _is_in_target_range(self, patch):
        mean = np.mean(patch)
        v_min = mean > self._i_valid_min
        v_max = mean < self._i_valid_max

        return v_min and v_max

    def _prepare_patches(self, patches):
        import cv2
        gs_patches = [(cv2.cvtColor(x[0], cv2.COLOR_BGR2GRAY), x[1]) for x in patches]

        return [x[0] for x in gs_patches if self._is_in_target_range(x[0])]


class EXPMLVidSubstrateFullFrame(EXPMLVidSubstrateBase):
    """
    The default behaviour is to train on full-frames anyway, as no further
    processing steps are required. Consult EXPMLVidSubstratePatch for the API
    to introduce patch / window based overrides in _process_frame()
    """

    pass


class EXPClassifierHandler(object):

    @staticmethod
    def get_all_saved_classifiers(directory):
        files_names = get_files_in_dir(directory)
        root_files = [x for x in files_names if not x.endswith(".npy")]

        return root_files

    @staticmethod
    def run_exp_for_all_classifiers(save_dir=DIR_CLASSIFIERS, parallel=False):
        """
        Runs all the saved classifiers that are located in save_dir.
        parallel, if True, will use the multiprocessing module to run
        multiple experiments at the same time.

        At present, however, this is broken due to the way in which Python
        processes match up to C-lib extensions. In this case, OpenCV just
        kinda dies when processing is attempted in this manner.

        Currently investigating a fix -- until then, just run linear or
        via threads.
        """
        classifiers = EXPClassifierHandler.get_all_saved_classifiers(DIR_CLASSIFIERS)
        classifiers = [x for x in classifiers if not x.endswith(".csv")]

        if parallel:
            max_processes = min(mp.cpu_count(), len(classifiers))
            pool = mp.Pool(processes=max_processes)

            try:
                for c in classifiers:
                    pool.apply_async(
                        EXPClassifierHandler.run_exp_for_classifier,
                        args=(c,)
                    )

                pool.close()
            except KeyboardInterrupt:
                log.info("Terminating pool due to KeyboardInterrupt")
                pool.terminate()
                raise
            except Exception, e:
                log.info("Terminating pool with exception {}".format(e))
                pool.terminate()
            finally:
                pool.join()
        else:
            for c in classifiers:
                EXPClassifierHandler.run_exp_for_classifier(c, save_dir)

    @staticmethod
    def run_exp_from_queue(queue):
        c = queue.get()
        EXPClassifierHandler.run_exp_for_classifier(c)

    @staticmethod
    def run_exp_for_classifier(classifier_file, save_dir=DIR_CLASSIFIERS):
        from app import get_api
        api = get_api()
        clf = load_saved_classifier(DIR_CLASSIFIERS + classifier_file)
        file_name = clf.video_path.split("/")[-1]  # Include c type & params

        fp = save_dir + STR_PARAM_DELIM.join(
            [classifier_file, "SSTRATE", "EXP", file_name, str(time.clock()), ".csv"]
        )

        exp = EXPMLVidSubstrateFullFrame(clf.ground_truth, clf.video_path,
                                None, classifier=clf, init=False)
        c_hash = hash(classifier_file)  # Unique ID for log tracking

        log.info("Testing Classifier: {} Hash: {} Video: {}".format(
            classifier_file, c_hash, clf.video_path
        ))

        training_frames = set(clf.training_data)  # The frames used to train

        with CSVEXPSubstrateWriter(fp, c_hash, clf) as csv_file:
            for klass in exp.classes:
                log.info("{}: Testing class {} begins.".format(c_hash, klass))
                start_time = datetime.datetime.now()

                class_frames = set(exp.ground_truth.get_frames_for_class(klass))
                testing_frames = class_frames.difference(training_frames)

                # Deconstruct the filename c to get the method
                function_name = classifier_file.split(STR_PARAM_DELIM)[-2]
                function_ptr = api.methods.get_function_ptr(function_name)

                for f in testing_frames:
                    exp.converter.set_current_frame(f)
                    exp.converter.capture_next_frame()

                    frame = exp.converter.current_frame
                    data = function_ptr(frame)

                    if isinstance(data, np.ndarray):
                        data = data.ravel()

                    result = exp.classifier.classify(data)  # Get the result
                    line = [f, klass, result[0], int(klass) == int(result)]
                    csv_file.writerow(line)

                end_time = datetime.datetime.now()
                time_delta = end_time - start_time

                log.info("{}: Testing class {} finishes (taken: {})".format(
                    c_hash, klass, time_delta
                ))


EXPERIMENT_MAP = {
    'fullframe': EXPMLVidSubstrateFullFrame,
    'patchbased_all': EXPMLVidSubstratePatch,
    'patchbased_ranged_bw': EXPMLVidSubstratePatchBWRanged
}
