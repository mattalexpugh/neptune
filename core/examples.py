import cv2

from structures.video import create_capture
from structures.interactive import CV2RectangleSelector
from structures.holders import HLDTrackers
from algorithms.tracking.mosse import MOSSETracker

__author__ = 'matt'


class MOSSECVWindow(object):

    FRAME_NAME = 'Video Frame'
    FRAME_NAME_TRACKER = 'Tracker Status'

    def __init__(self, video_src, paused=False):
        self.video_src = video_src
        self.reset()

        _, self.frame = self.cap.read()
        cv2.imshow(self.FRAME_NAME, self.frame)

        self.rect_sel = CV2RectangleSelector(self.FRAME_NAME, self.on_rect, self)
        self.paused = paused

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def reset(self):
        self.trackers = HLDTrackers()
        self.cap = create_capture(self.video_src)

    def add_tracker(self, tracker):
        self.trackers.add(tracker)

    def on_rect(self, rect):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.add_tracker(MOSSETracker(frame_gray, rect))

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()

                if not ret:
                    self.reset()
                    ret, self.frame = self.cap.read()

                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.trackers.update_all(frame_gray)

            vis = self.frame.copy()
            self.trackers.write_to_frame(vis)

            if self.trackers.is_active():  # Filter window
                cv2.imshow(self.FRAME_NAME_TRACKER,  self.trackers.last_tracker.state_vis)

            self.rect_sel.draw(vis)
            cv2.imshow(self.FRAME_NAME, vis)
            ch = cv2.waitKey(10)

            if ch == 27:
                   break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []
            if ch == ord('r'):
                self.reset()
            if ch == ord('q'):
                break
