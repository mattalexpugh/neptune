__author__ = 'matt'

import csv
import time
# import multiprocessing as mp
import billiard as mp
import datetime
import logging
log = logging.getLogger(__name__)

import numpy as np

from algorithms import get_api
from algorithms.learning.persist import DIR_CLASSIFIERS, STR_PARAM_DELIM,\
    load_saved_classifier
from core.gui.util.containers import VideoFrameConverter
from structures.holders import GTHVideoSubstrates
from structures.video import create_capture
from util.system import get_files_in_dir


class EXPMLVidSubstrate():

    def __init__(self, gt_path, video_path,
                 frame_func, init=True,
                 classifier=None):
        self._classifier = classifier
        self._capture = create_capture(video_path)
        self._converter = VideoFrameConverter(self._capture)
        self._ground_truth = GTHVideoSubstrates(gt_path, self._capture)
        self._classes = self._ground_truth.classes_present
        self._metric = frame_func

        self._classifier.ground_truth = gt_path
        self._classifier.video_path = video_path

        if init:
            self.train()

    def train(self, n=None):
        X = []                # frames n_samples * n_features
        Y = []                # labels n_samples
        training_frames = []  # Stores all frames used in training

        for k in self._classes:
            frames, ign = self.ground_truth.get_sample_frames_for_class(k, n=n)
            training_frames.extend(frames) # Keep a reference to them

            for f in frames:
                self.converter.set_current_frame(f)
                self.converter.capture_next_frame()

                frame = self.converter.current_raw_frame
                data = self._metric(frame)

                if isinstance(data, np.ndarray):
                    data = data.ravel()

                X.append(data)
                Y.append(k)

            log.info("Learned class " + str(k))

        # Keep a reference to the frames we have chosen
        self.classifier.training_data = training_frames
        log.info("Beginning fitting for " + self.classifier.identifier)
        self.classifier.raw_classifier.fit(np.array(X), np.array(Y))

    @property
    def classifier(self):
        return self._classifier

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def converter(self):
        return self._converter


def get_all_saved_classifiers(directory):
    files_names = get_files_in_dir(directory)
    root_files = [x for x in files_names if not x.endswith(".npy")]

    return root_files


class KeyboardInterruptError(Exception):

    pass


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
    classifiers = get_all_saved_classifiers(DIR_CLASSIFIERS)
    classifiers = [x for x in classifiers if not x.endswith(".csv")]

    if parallel:
        max_processes = min(mp.cpu_count(), len(classifiers))
        pool = mp.Pool(processes=max_processes)

        try:
            for c in classifiers:
                pool.apply_async(
                    run_exp_for_classifier,
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
            run_exp_for_classifier(c, save_dir)


def run_exp_from_queue(queue):
    c = queue.get()
    run_exp_for_classifier(c)


def run_exp_for_classifier(c, save_dir=DIR_CLASSIFIERS):
    api = get_api()
    csv_header = ['Input Frame', 'Expected', 'Received', 'Success']

    # c is a full filepath to the base file of the serialised classifier
    clf = load_saved_classifier(DIR_CLASSIFIERS + c)

    # Make a filename which includes the classifier type and parameters
    vp_full = clf.video_path
    parts = vp_full.split("/")
    file_name = parts[-1]

    fp = save_dir + STR_PARAM_DELIM.join(
        [c, "SSTRATE", "EXP", file_name, str(time.clock()), ".csv"]
    )

    exp = EXPMLVidSubstrate(clf.ground_truth, clf.video_path,
                            None, classifier=clf, init=False)

    # This is the unique ID for the filename, to track in logs.
    c_hash = hash(c)

    log.info("Testing Classifier: {} Hash: {} Video: {}".format(
        c, c_hash, clf.video_path
    ))

    training_frames = set(clf.training_data)  # The frames used to train
    classes = exp._classes

    with open(fp, 'wb') as csvfile:
        log.debug("Writing to CSV file {} ({})".format(fp, c_hash))

        exp_csv_writer = csv.writer(csvfile)
        exp_csv_writer.writerow(["Video:", clf.video_path,
                                 "GT:", clf.ground_truth])
        exp_csv_writer.writerow(csv_header)

        for klass in classes:
            log.info("{}: Testing class {} begins.".format(c_hash, klass))
            start_time = datetime.datetime.now()

            class_frames = set(exp._ground_truth.get_frames_for_class(klass))
            testing_frames = class_frames.difference(training_frames)

            # Deconstruct the filename c to get the method
            parts = c.split(STR_PARAM_DELIM)
            function_name = parts[-2]
            function_ptr = api.methods.get_function_ptr(function_name)

            for f in testing_frames:
                exp._converter.set_current_frame(f)
                exp._converter.capture_next_frame()

                frame = exp._converter.current_raw_frame
                data = function_ptr(frame)

                if isinstance(data, np.ndarray):
                    data = data.ravel()

                result = exp.classifier.classify(data) # Get the result
                line = [f, klass, result[0], int(klass) == int(result)]
                exp_csv_writer.writerow(line)

            end_time = datetime.datetime.now()
            time_delta = end_time - start_time

            log.info("{}: Testing class {} finishes (taken: {})".format(
                c_hash, klass, time_delta
            ))
