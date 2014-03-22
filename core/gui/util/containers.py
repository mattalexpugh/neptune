__author__ = 'matt'

from cv2 import cvtColor, COLOR_BGR2RGB, cv
import numpy as np
from math import ceil
from PyQt4.QtGui import QImage, QPixmap

### UTIL FUNCTIONS ###

def convert_single_frame(frame):
    try:
        h, w = frame.shape[:2]
        img = QImage(frame, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        return img
    except:
        return None

### CLASSES ###

class VideoFrameConverter():
    """
    This wraps around an OpenCV VideoCapture object, converting a frame
    to a QPixmap image for use in a QLabel. Bridges the gap between cv2
    & PyQt4
    """

    def __init__(self, capture):
        self.capture = capture
        self.current_frame = np.array([])

    def capture_next_frame(self):
        ret, read_frame = self.capture.read()

        if ret:
            self.current_frame = cvtColor(read_frame, COLOR_BGR2RGB)

    def convert_frame(self):
        frame = self.current_frame

        try:
            h, w = frame.shape[:2]
            self.previous_frame = frame
            img = QImage(frame, w, h, QImage.Format_RGB888)
            return QPixmap.fromImage(img)
        except:
            return None

    def get_next_frame(self):
        self.capture_next_frame()
        return self.convert_frame()

    def set_current_frame(self, i):
        self.capture.set(cv.CV_CAP_PROP_POS_FRAMES, ceil(i))

    @property
    def frame_number(self):
        return self.capture.get(cv.CV_CAP_PROP_POS_FRAMES) - 1

    @property
    def current_raw_frame(self):
        return self.current_frame

    @property
    def num_frames(self):
        return self.capture.get(cv.CV_CAP_PROP_FRAME_COUNT)

