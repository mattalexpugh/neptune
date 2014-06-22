__author__ = 'matt'


import time
import datetime
import logging
import os

import billiard as mp
import numpy as np

from algorithms.correction.colour import col_to_bgr
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
        self.classifier = classifier
        self.capture = create_capture(video_path)
        self.converter = VideoFrameConverter(self.capture)
        self.ground_truth = GTHVideoSubstrates(gt_path, self.capture)
        self.classes = self.ground_truth.classes_present
        self.metric = metric_func

        self.training_frames = []  # Stores all frames used in training
        self.x = []                # frames n_samples * n_features
        self.y = []                # labels n_samples

        self.classifier.ground_truth = gt_path
        self.classifier.video_path = video_path

        self.training_c_to_frames = {}
        self._setup_training_dict()

        if init:
            self.train()

    def _setup_training_dict(self):

        for k in self.classes:
            if k not in self.training_c_to_frames:
                self.training_c_to_frames[k] = [x for x in self.get_training_frames(k)]

    # def process_frame(self, frame):
    #     """
    #     Default frame processor, simply uses self.metric function pointer on
    #     the entire frame passed through. Converts the representation to a
    #     n x 1 vector for speed by default. Returns processed data in the form
    #     of a generator.
    #     """
    #     data = self.metric(frame)
    #
    #     #if isinstance(data, np.ndarray):
    #     #    data = data.ravel()
    #
    #     yield data

    def train(self):
        """
        Trains the classifier, looks to process_frame to deliver the metric-affected
        version of the whole frame, or subsections thereof. This method doesn't
        really care if process_frame returns 1 or more results, it just collates
        the training vectors X and Y before passing to sklearn for fitting.
        """

        # Thread safe lists here? @todo: multiprocessing
        for k, v in self.training_c_to_frames.iteritems():
            self.converter.set_current_frame(v)
            self.x.extend(self.metric(self.converter.capture_next_frame(True)))
            self.y.append(k)

        log.info("Beginning fitting for " + self.classifier.identifier)

        # Keep a reference to the frames we have chosen for this iteration
        self.classifier.training_data = self.training_frames
        self.classifier.raw_classifier.fit(np.array(self.x), np.array(self.y))

    def process_frame(self, frame):

        return self.metric(frame)

    def get_training_frames(self, klass, n=None):
        """
        Generator method that records the training frames selected in this iteration,
        and yields the frames sequentially for given klass. Parameter n is optional,
        and may indeed be deprecated. It is to limit the number below 10% of all frames,
        and was only relevant for Gabor full-frame image representations.
        """
        frames, _ = self.ground_truth.get_sample_frames_for_class(klass, n=n)
        self.training_frames.extend(frames)  # Keep a reference to them

        for f in frames:
            yield f

    def get_data_from_frame(self, frame):
        return frame


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
        self.m = m
        self.n = n

    def train(self):
        """
        Trains the classifier, looks to process_frame to deliver the metric-affected
        version of the whole frame, or subsections thereof. This method doesn't
        really care if process_frame returns 1 or more results, it just collates
        the training vectors X and Y before passing to sklearn for fitting.
        """

        # Thread safe lists here? @todo: multiprocessing
        for k, v in self.training_c_to_frames.iteritems():
            self.converter.set_current_frame(v)
            patches = [x for x in self.process_frame(self.converter.capture_next_frame(True))]
            self.x.extend([self.metric(x) for x in patches])
            self.y.extend([k for i in range(len(patches))])

        log.info("Beginning fitting for " + self.classifier.identifier)

        # Keep a reference to the frames we have chosen for this iteration
        self.classifier.training_data = self.training_frames
        self.classifier.raw_classifier.fit(np.array(self.x), np.array(self.y))

    def process_frame(self, frame):
        patches = self.converter.get_frame_mxn_patches_list(self.m, self.n)

        for patch in patches:
            data = self.metric(patch)

            if isinstance(data, np.ndarray):
                data = data.ravel()

            yield data


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

    def get_data_from_frame(self, patches):

        def viable_patch(patch):
            mean = np.mean(patch)

            return (mean > self._i_valid_min) and (mean < self._i_valid_max)

        gs_patches = [col_to_bgr(x[0]) for x in patches]

        return filter(viable_patch, gs_patches)


class EXPMLVidSubstrateFullFrame(EXPMLVidSubstrateBase):
    """
    The default behaviour is to train on full-frames anyway, as no further
    processing steps are required. Consult EXPMLVidSubstratePatch for the API
    to introduce patch / window based overrides in process_frame()
    """

    pass


class EXPClassifierHandler(object):

    @staticmethod
    def get_all_saved_classifiers(directory):
        files_names = get_files_in_dir(directory)
        root_files = [x for x in files_names if not x.endswith(".npy")]

        return root_files

    @staticmethod
    def run_exp_for_all_classifiers(save_dir=DIR_CLASSIFIERS, parallel=True):
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

        if len(classifiers) == 0:
            log.info("No more experiments to run, exiting.")
            return

        if parallel:
            videos_to_classifiers = {}

            for c in classifiers:
                clf = load_saved_classifier(save_dir + c)
                file_name = clf.video_path.split("/")[-1]

                if file_name not in videos_to_classifiers:
                    videos_to_classifiers[file_name] = []

                clfid = (clf.identifier, c)
                videos_to_classifiers[file_name].append(clfid)

            # So now we've mapped video_file: [classifiers], multiproc by k
            tasks = mp.Queue()
            results = mp.JoinableQueue()
            interim = []
            args = (tasks, results, save_dir)
            n_procs = min(mp.cpu_count(), len(videos_to_classifiers.keys()))

            for k in videos_to_classifiers.keys():
                these_classifiers = videos_to_classifiers[k]
                tasks.put(these_classifiers)

            delegator = EXPClassifierHandler.run_exp_from_mp_queue

            for _ in range(n_procs):
                p = mp.Process(target=delegator, args=args).start()

            for _ in range(len(videos_to_classifiers.keys())):
                interim.append(results.get())
                results.task_done()

            for _ in range(n_procs):
                tasks.put(None)

            results.join()
            tasks.close()
            results.close()
        else:
            for c in classifiers:
                EXPClassifierHandler.run_exp_for_classifier(c, save_dir)

        # Maybe by the time we get here more will be waiting... keep going
        EXPClassifierHandler.run_exp_for_all_classifiers(save_dir, parallel)

    @staticmethod
    def run_exp_from_queue(queue):
        c = queue.get()
        EXPClassifierHandler.run_exp_for_classifier(c)

    @staticmethod
    def run_exp_from_mp_queue(tasks, result, save_dir):
        for videos in iter(tasks.get, None):
            for _, path in videos:
                EXPClassifierHandler.run_exp_for_classifier(path, save_dir)

    @staticmethod
    def run_exp_for_classifier(classifier_file, save_dir=DIR_CLASSIFIERS):
        from app import get_api

        api = get_api()
        full_classifier_path = DIR_CLASSIFIERS + classifier_file
        clf = load_saved_classifier(full_classifier_path)
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
        results = []

        for klass in exp.classes:
            log.info("{}: Testing class {} begins.".format(c_hash, klass))
            start_time = datetime.datetime.now()

            class_frames = set(exp.ground_truth.get_frames_for_class(klass))
            testing_frames = class_frames.difference(training_frames)

            # Deconstruct the filename c to get the method
            function_name = classifier_file.split(STR_PARAM_DELIM)[-2]
            function_ptr = api.methods.get_function_ptr(function_name)

            # This seems like a nice candidate for parellisation
            # We have n_frames and n_procs. Split n_frames / n_procs and allocate
            for f in testing_frames:
                exp.converter.set_current_frame(f)
                exp.converter.capture_next_frame()
                frame = exp.converter.current_frame

                for element in exp.get_data_from_frame(frame):
                    # So this can be a full frame, or a window, or anything really...
                    # Just needs to be consistent with the original exp
                    data = function_ptr(element)

                    if isinstance(data, np.ndarray):
                        data = data.ravel()

                    result = exp.classifier.classify(data)  # Get the result
                    results.append([f, klass, result[0], int(klass) == int(result)])

            time_delta = datetime.datetime.now() - start_time

            log.info("{}: Testing class {} finishes (taken: {})".format(
                c_hash, klass, time_delta
            ))

        with CSVEXPSubstrateWriter(fp, c_hash, clf) as csv_file:
            map(lambda x: csv_file.writerow(x), results)

        os.rename(full_classifier_path, full_classifier_path + ".npy")


EXPERIMENT_MAP = {
    'fullframe': EXPMLVidSubstrateFullFrame,
    'patchbased_all': EXPMLVidSubstratePatch,
    'patchbased_ranged_bw': EXPMLVidSubstratePatchBWRanged
}
