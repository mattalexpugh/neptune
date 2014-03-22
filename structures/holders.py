__author__ = 'matt'
__doc__ = """

Containers to represent data internally. Standard class prefixes:

    Prefix      Definition                      Info
    ==============================================================================
    GTH         GroundTruthHolder               Loading ground truth into app
    GTM         GroundTruthMarkup               Storing ground truth during markup
    HLD         Holder                          Generic holder datastructure

"""


import time
from json import dumps
from random import randint

from core.gui.util.containers import VideoFrameConverter
from util.system import load_json
from config import APP_VERSION

### MODULE CONSTANTS ####

STR_DATA = 'data'
STR_AUTH = 'author'
STR_DATE = 'created'
STR_VERS = 'version'
STR_COMP = 'complete'
STR_FILE = 'filepath'
STR_EXST = 'supercedes'

STR_TRUTH_NOTES = 'notes'
STR_TRUTH_KLASS = 'class'

STR_FILE_EXT = '.json'
STR_RANGE_DELIM = '-'


class GTMVideoSubstrates():
    """
    Wraps an OpenCV VideoCapture class to find a set of images
    from a video file, the next image being one that is deemed
    sufficiently different as to possibly indicate a different
    substrate.
    """

    DEF_FRAMESKIP = 25

    def __init__(self, capture, video_src, output_dir, annotator, existing_data=None):
        """
        frameskip at 700 is *roughly* 30seconds
        """
        self._capture = capture
        self._converter = VideoFrameConverter(capture)
        self._frames_num_total = self._converter.num_frames
        self._output_dir = output_dir
        self._video_src = video_src
        self._complete = False
        self._existing_data = existing_data

        self.set_annotator(annotator)

        self._ground_truth = {
            STR_AUTH: annotator,
            STR_DATE: time.time(),
            STR_VERS: APP_VERSION,
            STR_COMP: False,
            STR_FILE: video_src,
            STR_DATA: {},
            STR_EXST: None
        }

        if self._existing_data is not None:
            self._build_from_existing_data()

    def _build_from_existing_data(self):
        self._ground_truth[STR_EXST] = self._existing_data
        data = self._existing_data[STR_DATA]

        # Convert the unicode strings to ints for lookup
        for key in data.keys():
            self._ground_truth[STR_DATA][int(key)] = data[key]

    def _get_target_filename(self):
        gt = self.gt
        dt = str(gt[STR_DATE]).split('.')[0]
        vr = gt[STR_VERS]
        cm = "COMP" if gt[STR_COMP] else "INCOMP"
        an = "_".join(self.annotator.upper().split())

        order = [an, dt, cm, vr]

        return "_".join(order) + STR_FILE_EXT

    def _get_target_abspath(self):
        return self._output_dir + "/" + self._get_target_filename()

    def add_truth(self, frame_id, klass, notes=None):
        if len(notes) < 1:
            notes = None
        else:
            notes = str(notes)

        tmp = {
            STR_TRUTH_KLASS: klass,
            STR_TRUTH_NOTES: notes
        }

        self._ground_truth[STR_DATA][int(frame_id)] = tmp

    def has_truth_for(self, target):
        return target in self.gt[STR_DATA].keys()

    def get_truth_for(self, target):
        if not self.has_truth_for(target):
            return None

        return self.gt[STR_DATA][target]

    def set_annotator(self, annotator):
        self.annotator = annotator

    def set_complete(self):
        self.gt[STR_COMP] = True

    def save_file(self):
        json_str = dumps(self.gt, indent=4, sort_keys=True)
        file_path = self._get_target_abspath()
        fp = open(file_path, 'w')
        fp.write(json_str)
        fp.close()

        return file_path

    def remove_truth_for(self, target):
        if not self.has_truth_for(target):
            return False

        del self.gt[STR_DATA][target]

    @property
    def converter(self):
        return self._converter

    @property
    def frame_number(self):
        return int(self.converter.frame_number) + 1

    @property
    def frame_numbers_with_data(self):
        return list(set(self.gt[STR_DATA].keys()))

    @property
    def next_frame(self):
        frame = self.converter.get_next_frame()
        i = self.converter.frame_number + 1

        if i > self._frames_num_total:
            i = self._frames_num_total

        self.converter.set_current_frame(i)
        return frame

    @property
    def gt(self):
        return self._ground_truth


class GTHVideoSubstrates():

    def __init__(self, input_json, capture):
        self._input_json = input_json
        self._capture = capture
        self._converter = VideoFrameConverter(capture)
        self._classes = {}
        self._parse_json(input_json)

    def _parse_json(self, input_json):
        """
        Loads the JSON from the filename and executes the correct
        parsing functions in order.
        """
        self._raw_json = load_json(input_json)
        self._gtruth = gt = self._parse_json_data(self._raw_json)

        rawdata = [(int(k), gt[k][STR_TRUTH_KLASS]) for k in gt.keys()]
        self._frame2class = sorted(rawdata)

        self._count_foi()
        self._concatenate_classes()

    def _parse_json_data(self, js_data):
        """
        Recursively search the nested groupings and build into
        one list.
        """
        data = js_data[STR_DATA]

        if js_data[STR_EXST] is not None:
            data.extend(self._parse_json_data(js_data[STR_EXST]))

        return data

    def _concatenate_classes(self):
        """
        Go through what's been collected, concatenating the ranges
        based on class.
        """
        classes = self.class_ranges_usable

        for klass in classes:
            rng, kl = klass

            if kl not in self._classes:
                self._classes[kl] = []

            self._classes[kl].append(rng)

    def _count_foi(self):
        """
        Counts the frames of interest in the video, will
        disregard any that have class == 0
        """
        f2c = self.f2c
        ranges = []
        start = None
        range_klass = None
        first = True

        for pair in f2c:
            fnum, klass = pair

            if first:
                start = 0
                range_klass = klass if fnum == 0 else 0
                first = False

            ln = str(start) + STR_RANGE_DELIM + str(fnum)
            pr = (ln, range_klass)

            ranges.append(pr)

            start = fnum
            range_klass = klass

        self._class_ranges_orig = ranges
        self._class_ranges_usable = [x for x in ranges if x[1] != 0]
        self._foi = 0

        for pair in self._class_ranges_usable:
            rng, klass = pair
            lrng = [int(x) for x in rng.split(STR_RANGE_DELIM)]
            diff = lrng[1] - lrng[0]
            self._foi += diff

    def get_ranges_for_class(self, klass):
        """
        Get a list of strings which are the ranges of a
        specified class.
        """
        return self.class_ranges[klass]

    def get_frames_for_class(self, klass):
        """
        Returns a generator operator that will iterate through
        all the frame numbers in all the ranges in which this
        specified class has been marked.
        """
        rngs = self.get_ranges_for_class(klass)

        for rng in rngs:
            parts = rng.split(STR_RANGE_DELIM)
            st, end = int(parts[0]), int(parts[1])

            for x in range(st, end):
                yield x

    def get_sample_frames_for_class(self, klass, tr_ratio=0.1, n=None):
        """
        Returns a random sample of 10% of any given klass's
        frames from throughout the video.

            klass - the label to be searched for, matches frames with that class
            tr_ratio - training / testing split. tr_ratio * num_frames = training set
            n - [optional] return a maximum of n frames for class klass
        """
        rngs = self.get_ranges_for_class(klass)
        total_frames = []
        target_frames = []

        for rng in rngs:
            parts = rng.split(STR_RANGE_DELIM)
            st, end = int(parts[0]), int(parts[1])
            total_frames.extend(range(st, end))

        num_frames = len(total_frames)
        ten_pc = int(num_frames * tr_ratio)

        if n is not None and ten_pc > n:
            ten_pc = n  # Set the upper bounds to be n in this case.

        for i in range(ten_pc):
            found = False
            candidate = None

            while not found:
                next_frame_num = randint(0, num_frames - 1)
                candidate = total_frames[next_frame_num]
                found = candidate not in target_frames

            target_frames.append(candidate)

        return target_frames, (num_frames, ten_pc, klass)

    def get_estimated_buffer(self, w=1920, h=1080, c=3):
        """
        Estimates the size in bytes of the uncompressed video
        frames from all frames of interest.
        """
        foi = self.num_foi
        sng = w * h * c
        return foi * sng

    @property
    def class_ranges(self):
        """
        Returns the main class: [ranges] dictionary
        """
        return self._classes

    @property
    def class_ranges_usable(self):
        return self._class_ranges_usable

    @property
    def classes_present(self):
        """
        Returns a list of all the classifications present in
        the video.
        """
        return self.class_ranges.keys()

    @property
    def num_foi(self):
        """
        Returns integer number of Frames of Interest (foi),
        i.e. the number of frames that have been marked up as
        not being 0.
        """
        return self._foi

    @property
    def f2c(self):
        return self._frame2class


class HLDTrackers():
    """
    Holds a list of different trackers, abstracted for use
    both with cv2 and PyQt4
    """

    def __init__(self):
        self.trackers = None
        self.last_tracker = None

        # Now initialise everything to their required state
        self.reset()

    def __iter__(self):
        return iter(self.trackers.values())

    def get(self, i):
        return self.trackers[i]

    def add(self, tracker):
        self.last_tracker = tracker
        self.trackers[tracker.get_id()] = tracker

    def remove(self, tracker):
        if tracker != '':
            del self.trackers[tracker]

    def reset(self):
        self.trackers = {}
        self.last_tracker = None

    def update_all(self, frame):
        for tracker in self:
            tracker.update(frame)

    def write_to_frame(self, frame):
        for tracker in self:
            tracker.draw_state(frame)

    @property
    def last_tracker(self):
        return self.last_tracker

    @property
    def num_trackers(self):
        return len(self.trackers)

    @property
    def active(self):
        return self.num_trackers > 0
