import logging
log = logging.getLogger(__name__)

import app.paths as p
from algorithms import get_api
from experiments.videolearning import EXPMLVidSubstrate
from util.system import load_json


__author__ = 'matt'

KEY_DESCR = "description"
KEY_CLASS = "experiment"

# JSON specific keys
KEY_ALLFUNC = "allFunctions"
KEY_EXPTYPE = "experimentType"
KEY_SELFUNC = "selectedFunctions"
KEY_ALLCLAS = "allClassifiers"
KEY_SELCLAS = "selectedClassifiers"
KEY_SELVIDS = "selectedVideos"

API = get_api()

STR_GT_DELIM = "|"
STR_VID_COL_DELIM = "::"


def get_exp_harness(json_file):
    exp_json = load_json(json_file)

    if KEY_EXPTYPE in exp_json:
        klass = API_MAP[exp_json[KEY_EXPTYPE]][KEY_CLASS]
        exp = klass(exp_json)
    else:
        exp = None

    return exp


class EXPHarnessLoader(object):
    """
    This will allow experiments to be defined in a JSON file, to be loaded
    and parsed in this object.

    Experiment manifests should have the following:

        * Methods to be run on all videos
        * Source videos (full paths) to be run
            - Potential override to define subset of methods to run on each video
    """

    def __init__(self, json_data):
        self._json = json_data

    def run(self):
        pass

    @property
    def json(self):
        return self._json

    @staticmethod
    def get_exp_for(json_file):
        return get_exp_harness(json_file)


class EXPSubstrateHarness(EXPHarnessLoader):

    def __init__(self, json_file):
        super(EXPSubstrateHarness, self).__init__(json_file)

        self._all_functions = self.json[KEY_ALLFUNC]
        self._all_classifiers = self.json[KEY_ALLCLAS]

        if not self.use_all_funcs or not self.use_all_classifiers:
            self._load_selected_only()

        self._validate_and_select_targets()
        self._validate_and_select_videos()

    def _load_selected_only(self):
        self._selected_funcs = self.json[KEY_SELFUNC]
        self._selected_classifiers = self.json[KEY_SELCLAS]

    def _validate_and_select_targets(self):
        if self.use_all_funcs:
            funcs = API.methods.all_ptrs
        else:
            # This was originally a list comprehension, but error handling
            # for missing functions would be better.
            funcs = []

            for func_api_ptr in self.selected_funcs:
                func_ptr = API.methods.get_function_ptr(func_api_ptr)

                if func_ptr is None:
                    log.info("Unknown API method called: " + func_api_ptr)
                else:
                    # So we can have a name for the method when saving metric
                    funcs.append((func_api_ptr, func_ptr))

        if self.use_all_classifiers:
            classifiers = API.classifiers.all_ptrs
        else:
            classifiers = []

            for classifier_api_ptr in self.selected_classifiers:
                classifier_ptr = API.classifiers.get_function_ptr(classifier_api_ptr)

                if classifier_ptr is None:
                    log.info("Unknown API classifier: " + classifier_api_ptr)
                else:
                    classifiers.append(classifier_ptr)

        self._funcs = funcs
        self._classifiers = classifiers

    def _validate_and_select_videos(self):
        gt_vid_pairs = []

        for details in self.json[KEY_SELVIDS]:
            # This would look much nicer in regex, but it works for now...
            gt_vid_split = details.split(STR_GT_DELIM)
            gt = gt_vid_split[0]

            vid_details = gt_vid_split[1].split(STR_VID_COL_DELIM)
            vid_type = vid_details[0]
            vid_name = vid_details[1]

            gt_file_path = p.get_path(p.KEY_GT) + gt
            vid_file_path = p.get_path(vid_type.lower()) + vid_name

            pair = (gt_file_path, vid_file_path)
            gt_vid_pairs.append(pair)

        self._videos = gt_vid_pairs

    def run(self):
        for video_gt_pair in self._videos:
            gt = video_gt_pair[0]
            fp = video_gt_pair[1]

            for func in self._funcs:
                func_name = func[0]
                func_ptr = func[1]

                for classifier in self._classifiers:
                    exp = EXPMLVidSubstrate(gt, fp, func_ptr, False,
                                            classifier(metric=func_name))
                    exp.train()
                    saved_classifier = exp.classifier.save()
                    del exp

                    log.info("Saved classifier: " + saved_classifier)

    @property
    def use_all_funcs(self):
        return self._all_functions

    @property
    def use_all_classifiers(self):
        return self._all_classifiers

    @property
    def selected_funcs(self):
        return self._selected_funcs

    @property
    def selected_classifiers(self):
        return self._selected_classifiers


API_MAP = {
    'VIDEOSUBSTRATE': {
        KEY_DESCR: "Video Substrate Classification Experiment",
        KEY_CLASS: EXPSubstrateHarness
    }
}
