__author__ = 'matt'

import csv
import time
import multiprocessing as mp
import logging
log = logging.getLogger(__name__)

import numpy as np

import algorithms.learning.svm as nepsvm
import algorithms.learning.knn as nepknn
import app.paths as p

from algorithms.learning.persist import DIR_CLASSIFIERS, load_saved_classifier
from core.gui.util.containers import VideoFrameConverter
from structures.holders import GTHVideoSubstrates
from structures.video import create_capture
from util.system import get_files_in_dir

import algorithms.features.histograms as histo
import algorithms.features.wavelets as wavelets

metrics = {
    'color-hist': histo.cv2_calc_hist_bgr,
    'gray-hist': histo.cv2_calc_hist_gray,
    'color-retinex-hist': histo.cv2_calc_hist_retinex_bgr,
    'gray-retinex-hist': histo.cv2_calc_hist_retinex_gray,
    # 'gabor': wavelets.gabor_composite_from_thetas,
    'gabor-lbp-1': wavelets.gabor_composite_lbp1,
    'gabor-lbp-2': wavelets.gabor_composite_lbp2,
    'gabor-lbp-3': wavelets.gabor_composite_lbp3,
    'gabor-histo': wavelets.gabor_histo_at_4_rotations
}

restricted = ['gabor', 'gabor-histeq']
skip = []
gabor_methods = ['gabor-histo', 'gabor-lbp-1', 'gabor-lbp-2', 'gabor-lbp-3']

class EXPMLVidSubstrate():

    def __init__(self, gt_path, video_path, frame_func, init=True, classifier=None):
        self._classifier = classifier if classifier is not None else nepsvm.VSVMSubstrateClassifier()
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
        X = []                                          # frames n_samples * n_features
        Y = []                                          # labels n_samples
        training_frames = []                            # Stores all frames used in training

        for k in self._classes:
            frames, ign = self.ground_truth.get_sample_frames_for_class(k, n=n)
            training_frames.extend(frames)              # Keep a reference to them

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

        self.classifier.training_data = training_frames  # Keep a reference to the frames we have chosen
        log.info("Beginning fitting for " + self.ground_truth + " and " + self.classifier.identifier)
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



def cross_test_results(classifiers, videos):
    """
    Need to load in the saved classifiers and their associated videos.
    Test each classifier against each video for matching region classifications.
    """
    pass


def get_instantiated_classifiers(funcName):

    classifiers = {
        'svms': nepsvm.get_svcs_for_all_kernels(metric=funcName),
        'knns': nepknn.get_knns_for_all_algorithms(metric=funcName, n=5)
    }

    # Manually stick this on for now
    classifiers['svms']['LinearSVC'] = nepsvm.VLinearSVCSubstrateClassifier(metric=funcName)

    return classifiers


def train_classifier(fp, vfp, experiments=None, fname_extra=None):
    experiments = {} if experiments is None else experiments

    for function_name in sorted(metrics.keys()):

        if function_name in skip:
            continue

        classifiers = get_instantiated_classifiers(function_name)

        for classifier_type in classifiers.keys():

            if function_name not in experiments:
                experiments[function_name] = {}

            for classifier in classifiers[classifier_type]:
                log.info("Starting " + classifier_type + " with " + classifier + " and function " + function_name)

                if function_name in restricted:
                    exp = EXPMLVidSubstrate(fp, vfp, metrics[function_name],
                                            classifier=classifiers[classifier_type][classifier],
                                            init=False)
                    exp.train(n=200)
                else:
                    exp = EXPMLVidSubstrate(fp, vfp, metrics[function_name],
                                            classifier=classifiers[classifier_type][classifier])

                # The classifier saves to DIR_CLASSIFIERS in os-scope. Returns a reference to the filepath.
                experiments[function_name][classifier_type + classifier] = exp.classifier.save(fname_extra=fname_extra)
                experiments[function_name]['gt'] = fp
                experiments[function_name]['vf'] = vfp

                del exp  # Pray the garbage collector isn't on strike


def train_only_gabor_classifiers(sources):
    """
    This should do the gabor stuff all at the same time.
    For every frame in a target video:
        Get a list of the rotation responses for different scales
            1) Gabor Histo - hist horizontal stack
            2) Composite - histo
            3) LBP 1
            4) LBP 2
            5) LBP 3
            Save separate classifiers
    """
    experiments = {k: get_instantiated_classifiers(k) for k in gabor_methods}
    log.info("Saved experiments (Gabor): " + str(experiments))

    for method in experiments.keys():
        for classifier_type in experiments[method].keys():
            pass

    pass


def train_classifiers_parallel(sources):

    sources_list = []

    for k in sources.keys():
        # Build up the full fp and vfp, pass through
        for fp, vfp in sources[k]:
            fp = p.get_path(p.KEY_GT) + fp
            mvfp = p.get_path(k) + vfp

            sources_list.append((fp, mvfp, None, vfp))

    max_processes = min(mp.cpu_count(), len(sources_list))
    pool = mp.Pool(processes=max_processes)

    for source in sources_list:
        pool.apply_async(train_classifier, args=source)

    pool.close()
    pool.join()


def train_classifiers_linear(sources):

    experiments = {}

    for dataset in sources.keys():
        datas = sources[dataset]

        for fp, vfp in datas:
            fp = p.get_path(p.KEY_GT) + fp
            mvfp = p.get_path(dataset) + vfp

            train_classifier(fp, mvfp, experiments, fname_extra=vfp)

    return experiments


def get_all_saved_classifiers(directory):
    files_names = get_files_in_dir(directory)
    root_files = [x for x in files_names if not x.endswith(".npy")]

    return root_files


def run_exp_for_all_classifiers(save_dir=DIR_CLASSIFIERS):
    classifiers = get_all_saved_classifiers(DIR_CLASSIFIERS)
    classifiers = [x for x in classifiers if not x.endswith(".csv")]

    for c in classifiers:
        # c is a full filepath to the base file of the serialised classifier object
        clf = load_saved_classifier(DIR_CLASSIFIERS + c)
        log.info("Testing Classifier: " + c)

        # Make a filename which includes the classifier type and parameters
        vp_full = clf.video_path
        parts = vp_full.split("/")
        fname = parts[-1]
        fp = save_dir + "_".join([c, "SSTRATE", "EXP", fname, str(time.clock()), ".csv"])

        exp = EXPMLVidSubstrate(clf.ground_truth, clf.video_path,
                                None, classifier=clf, init=False)

        training_frames = set(clf.training_data)  # The frames used to train clf
        classes = exp._classes

        with open(fp, 'wb') as csvfile:
            log.debug("Writing to CSV file " + fp)
            exp_csv_writer = csv.writer(csvfile)
            exp_csv_writer.writerow(["Video:", clf.video_path, "GT:", clf.ground_truth])
            exp_csv_writer.writerow(['Input Frame', 'Expected', 'Received', 'Success'])

            for klass in classes:
                class_frames = set(exp._ground_truth.get_frames_for_class(klass))
                # So here I need to deduct the overlap of X from  allframes:
                # I want to subtract all elements from class_frames that appear in both it and training_frames
                testing_frames = class_frames.difference(training_frames)

                # So I know the class that every frame in testing_frames is (klass)
                # Deconstruct the filename c to get the method
                parts = c.split('_')
                function_name = parts[-2]

                # Return the function handler
                function_ptr = metrics[function_name]

                # Iterate over the frames
                for f in testing_frames:
                    exp._converter.set_current_frame(f)
                    exp._converter.capture_next_frame()

                    frame = exp._converter.current_raw_frame
                    data = function_ptr(frame)

                    if isinstance(data, np.ndarray):
                        data = data.ravel()

                    # Get the classifier result
                    result = exp.classifier.classify(data)

                    # Write it to CSV
                    # @todo: check the difference of ccw vs internal schema and add as column
                    line = [f, klass, result[0], int(klass) == int(result)]
                    exp_csv_writer.writerow(line)

                log.info("Testing class " + str(klass))


def run_trainers():
    sources = {
        p.KEY_GOPRO: [
            ("MATT_1387311857_COMP_0.1.json", "Site 10.MP4"),
            ("PHIL_HUGHES_1391025035_COMP_0.1.json", "Site 2.MP4"),
            ("PHIL_HUGHES_1391026047_COMP_0.1.json", "Site 3.MP4")
        ]
    }

    train_classifiers_parallel(sources)


if __name__ == '__main__':

    sources = {
        p.KEY_GOPRO: [
            ("MATT_1387311857_COMP_0.1.json", "Site 10.MP4"),
            ("PHIL_HUGHES_1391025035_COMP_0.1.json", "Site 2.MP4")
            ("PHIL_HUGHES_1391026047_COMP_0.1.json", "Site 3.MP4")
        ],

        p.KEY_FOCB: [
            ("MATT_PUGH_1391010753_COMP_0.1.json", "aber harbour 26th sep 08 tape 2.wmv"),
            ("MATT_PUGH_1391012089_COMP_0.1.json", "clarach consti and harbour pollock.wmv")
        ]
    }

    train_classifiers_linear(sources)
    # train_classifiers_parallel(sources)
    run_exp_for_all_classifiers()
