from algorithms import get_api
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

    def _load_selected_only(self):
        self._selected_funcs = self.json[KEY_SELFUNC]
        self._selected_classifiers = self.json[KEY_SELCLAS]

    def run(self):
        if self.use_all_funcs:
            funcs = API.methods.all_ptrs
        else:
            funcs = [API.methods.get_function_ptr(x) for x in self.selected_funcs]

        if self.use_all_classifiers:
            classifiers = API.classifiers.all_ptrs
        else:
            classifiers = [API.classifiers.get_function_ptr(x) for x in self.selected_classifiers]

        for func in funcs:
            for classifier in classifiers:
                pass


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
