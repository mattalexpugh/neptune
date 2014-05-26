__author__ = 'matt'

import os

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from cv2 import cvtColor, COLOR_BGR2GRAY, COLOR_GRAY2RGB

from core.gui.components import QWVideoPositionSlider
from core.gui.util.containers import VideoFrameConverter, convert_single_frame
from structures.video import create_capture
from structures.holders import HLDTrackers
from algorithms.tracking.mosse import TRACKER_TYPES


### MOSSE TRACKING VIDEO COMPONENTS ###

class QWVideoFrameLabel(QLabel):
    """
    Enables the selection of image regions through the QRubberBand class,
    coordinates are then given to parent to assign to a new tracker.
    """

    def __init__(self, parent):
        super(QWVideoFrameLabel, self).__init__(parent=parent)
        self.parent = parent
        self.origin = 0
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        self.parent.pause()
        self.origin = event.pos()
        self.rubber_band.setGeometry(
            QRect(self.origin, QSize()))
        self.rubber_band.show()

        QLabel.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.rubber_band.isVisible():
            self.rubber_band.setGeometry(
                QRect(self.origin, event.pos()).normalized())

        QLabel.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        # @todo: check boundaries of box
        if self.rubber_band.isVisible():
            self.rubber_band.hide()
            rect = self.rubber_band.geometry()
            tl = rect.topLeft()
            br = rect.bottomRight()
            coords = (tl.x(), tl.y(), br.x(), br.y())

            self.parent.add_tracker(coords)
            self.parent.unpause()

        QLabel.mouseReleaseEvent(self, event)


class WTrackerOutputWindow(QWidget):
    """
    Sub window, displays the output of the image area of interest,
    kernel and filter response.
    """

    STR_WIN_TITLE = 'Filter Output'

    def __init__(self, parent):
        super(WTrackerOutputWindow, self).__init__()
        self.parent = parent
        self.frm_vid_filters = QLabel(self)

        self.lyt_main = QVBoxLayout(self)
        self.lyt_main.setAlignment(Qt.AlignTop)
        self.lyt_main.addWidget(self.frm_vid_filters)

        self.setLayout(self.lyt_main)
        self.setGeometry(40, 20, 200, 400)
        self.setWindowTitle(self.STR_WIN_TITLE)

    def update_image(self, image):
        self.frm_vid_filters.setPixmap(image)
        geo = self.geometry()
        tl = geo.topLeft()
        x1, y1 = tl.x(), tl.y()
        w, h = image.width(), image.height()
        self.setGeometry(x1, y1, w, h)


class WTrackerMarking(QMainWindow):
    """
    Main Tracking Window
    """

    STR_WIN_TITLE = 'Mark Video'
    STR_CLK = 'clicked()'
    STR_SPECIFY_VIDEO = 'Specify Video'

    def __init__(self, def_path=None):
        super(WTrackerMarking, self).__init__(None)
        self._trackers = HLDTrackers()
        self.paused = False
        self.timer = QTimer(self)

        # Widgets
        lbl_path_str = QLabel("Video Path:")
        path = def_path if def_path is not None else os.getcwd()

        self.lbl_path = QLabel(path)
        self.lbl_path.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        lbl_sld_pos = QLabel("Position:")

        btn_path = QPushButton("&Path...")
        btn_load = QPushButton("&Load")
        btn_vid_toggle = QPushButton("&Start/Stop")
        btn_reset_video = QPushButton("&Reset Video")
        btn_reset_track = QPushButton("Reset &Trackers")

        self.sld_pos = QWVideoPositionSlider(Qt.Horizontal, self, self.pause, self.unpause)
        self.sld_pos.valueChanged[int].connect(self.update_position)

        btn_rem_tracker = QPushButton("&Delete")

        self.cmb_trackers = QComboBox(self)
        self.cmb_tracker_types = QComboBox(self)
        self._build_cmb_trackers()

        # Images
        self.win_vid_filters = WTrackerOutputWindow(self)
        self.frm_vid_output = QWVideoFrameLabel(self)
        self.frm_vid_output.setGeometry(10, 10, 400, 100)

        # Callbacks
        self.connect(btn_path, SIGNAL(self.STR_CLK), self.set_path)
        self.connect(btn_load, SIGNAL(self.STR_CLK), self.load_video)
        self.connect(btn_vid_toggle, SIGNAL(self.STR_CLK), self.toggle_video)
        self.connect(btn_reset_video, SIGNAL(self.STR_CLK), self.do_reset_video)
        self.connect(btn_reset_track, SIGNAL(self.STR_CLK), self.do_reset_trackers)
        self.connect(btn_rem_tracker, SIGNAL(self.STR_CLK), self.remove_tracker)

        # Layouts
        lyt_main = QVBoxLayout()

        lyt_top = QHBoxLayout()
        lyt_top.addWidget(lbl_path_str)
        lyt_top.addWidget(self.lbl_path, 1)
        lyt_top.addWidget(btn_path)

        lyt_btm = QHBoxLayout()
        lyt_btm.addWidget(btn_load)
        lyt_btm.addWidget(btn_vid_toggle)
        lyt_btm.addWidget(btn_reset_video)
        lyt_btm.addWidget(btn_reset_track)
        lyt_btm.addWidget(self.cmb_tracker_types)
        lyt_btm.addWidget(lbl_sld_pos)
        lyt_btm.addWidget(self.sld_pos)
        lyt_btm.addWidget(self.cmb_trackers)
        lyt_btm.addWidget(btn_rem_tracker)

        lyt_main.setAlignment(Qt.AlignTop)
        lyt_main.addLayout(lyt_top)
        lyt_main.addLayout(lyt_btm)
        lyt_main.addWidget(self.frm_vid_output)
        lyt_main.setMargin(10)
        lyt_main.setSpacing(10)

        widget = QWidget()
        widget.setLayout(lyt_main)

        self.setCentralWidget(widget)
        self.setGeometry(260, 20, 1000, 55)
        self.setWindowTitle(self.STR_WIN_TITLE)

    ## Overridden functions, handle child window
    def show(self):
        super(WTrackerMarking, self).show()
        self.win_vid_filters.show()

    def hide(self):
        self.win_vid_filters.hide()
        super(WTrackerMarking, self).hide()

    def close(self):
        self.win_vid_filters.close()
        super(WTrackerMarking, self).close()

    ## Object functions
    def _build_cmb_trackers(self, default='MOSSE'):
        for t in TRACKER_TYPES.keys():
            self.cmb_tracker_types.addItem(t)

        id_ref = self.cmb_tracker_types.findText(default)
        self.cmb_tracker_types.setCurrentIndex(id_ref)

    def add_tracker(self, rect):
        tracker_name = self.cmb_tracker_types.currentText()
        tracker = TRACKER_TYPES[str(tracker_name)]

        frame_gray = cvtColor(self.vid.current_frame, COLOR_BGR2GRAY)
        # This now depends on the value of the combobox
        new_tracker = tracker(frame_gray, rect)
        self.trackers.add(new_tracker)

        tracker_id = str(new_tracker.get_id())
        self.cmb_trackers.addItem(tracker_id)
        id_ref = self.cmb_trackers.findText(tracker_id)
        self.cmb_trackers.setCurrentIndex(id_ref)

    def remove_tracker(self):
        tracker_id = self.current_tracker_id
        self.trackers.remove(tracker_id)

        tracker_ref = self.cmb_trackers.findText(tracker_id)
        self.cmb_trackers.removeItem(tracker_ref)

    def do_reset_video(self):
        self.timer.stop()
        self.load_video()

    def do_reset_trackers(self):
        self.trackers.reset()
        self.cmb_trackers.clear()

    def toggle_video(self):
        self.paused = not self.paused

    def set_path(self):
        path = QFileDialog.getOpenFileName(self,
                self.STR_SPECIFY_VIDEO, self.lbl_path.text())
        if path:
            self.lbl_path.setText(QDir.toNativeSeparators(path))

    def load_video(self):
        path = self.lbl_path.text()
        self.cap = create_capture(path)
        self.vid = VideoFrameConverter(self.cap)
        self.sld_pos.setRange(0, self.vid.num_frames)
        self.sld_pos.setValue(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._play)
        self.timer.start(27)
        self.update()

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def _play(self):
        if self.paused:
            return

        frame = self.vid.get_next_frame()

        if frame:
            if self.trackers.active:
                raw_frame = self.vid.current_frame
                frame_gray = cvtColor(raw_frame, COLOR_BGR2GRAY)
                self.trackers.update_all(frame_gray)
                self.trackers.write_to_frame(raw_frame)
                frame = convert_single_frame(raw_frame)

                tracker = self.trackers.get(self.current_tracker_id)
                vis = cvtColor(tracker.state_vis, COLOR_GRAY2RGB)
                q_vis = convert_single_frame(vis)
                self.win_vid_filters.update_image(q_vis)

            self.frm_vid_output.setPixmap(frame)
        else:
            self.timer.stop()
            self.load_video()

    def update_position(self, value):
        self.vid.set_current_frame(value)

    @property
    def trackers(self):
        return self._trackers

    @trackers.setter
    def trackers(self, new_trackers):
        self._trackers = new_trackers

    @property
    def current_tracker_id(self):
        return str(self.cmb_trackers.currentText())
