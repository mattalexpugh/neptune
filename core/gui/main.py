__author__ = 'matt'

import sys
import logging
log = logging.getLogger(__name__)

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from core.gui.markup import WTrackerMarking
from core.gui.interactive import WVideoFrameWavelet
from core.gui.groundtruth import WVideoGroundTruthNewSetup, WVideoGroundTruthExistingSetup

from config import APP_NAME


class WAppSelectAction(QMainWindow):

    STR_WIN_TITLE = "Select Function"
    STR_CLK = "clicked()"

    SUB_WINDOWS = {
        'WVideoGroundTruthNewSetup': WVideoGroundTruthNewSetup,
        'WVideoGroundTruthExistingSetup': WVideoGroundTruthExistingSetup,
        'WTrackerMarking': WTrackerMarking,
        'WVideoFrameWavelet': WVideoFrameWavelet
    }

    def __init__(self):
        super(WAppSelectAction, self).__init__(None)
        self._open_windows = []

        ## Widgets
        grp_markup = QGroupBox("Ground Truth (Video)")
        grp_markup.setAlignment(Qt.AlignTop)
        grp_interactive = QGroupBox("Interactive")
        grp_interactive.setAlignment(Qt.AlignTop)

        # Ground Truth / Input
        btn_mark_video_new = QPushButton("Start New Session")
        btn_mark_video_existing = QPushButton("Open Existing Session")

        # Interactive / Output
        btn_do_mosse = QPushButton("&MOSSE Tracking")
        btn_do_wavelet = QPushButton("&Wavelet Decomposition Testing")

        # Util
        btn_do_quit = QPushButton("&Quit")

        # Callbacks
        self.connect(btn_mark_video_new, SIGNAL(self.STR_CLK), self.load_mark_video_new)
        self.connect(btn_mark_video_existing, SIGNAL(self.STR_CLK), self.load_mark_video_existing)
        self.connect(btn_do_mosse, SIGNAL(self.STR_CLK), self.load_mosse_interactive)
        self.connect(btn_do_wavelet, SIGNAL(self.STR_CLK), self.load_wavelet_interactive)
        self.connect(btn_do_quit, SIGNAL(self.STR_CLK), self.do_quit)

        # Layouts
        lyt_markup = QVBoxLayout()
        lyt_markup.addWidget(btn_mark_video_new)
        lyt_markup.addWidget(btn_mark_video_existing)
        grp_markup.setLayout(lyt_markup)

        lyt_interactive = QVBoxLayout()
        lyt_interactive.addWidget(btn_do_mosse)
        lyt_interactive.addWidget(btn_do_wavelet)
        grp_interactive.setLayout(lyt_interactive)

        lyt_main = QVBoxLayout()
        lyt_main.setMargin(10)
        lyt_main.setSpacing(10)
        lyt_main.setAlignment(Qt.AlignTop)
        lyt_main.addWidget(grp_markup)
        lyt_main.addWidget(grp_interactive)
        lyt_main.addWidget(btn_do_quit)

        widget = QWidget()
        widget.setLayout(lyt_main)

        self.setCentralWidget(widget)
        self.setGeometry(50, 40, 250, 50)
        self.setWindowTitle(APP_NAME + " - " + self.STR_WIN_TITLE)

    def _add_open_window(self, win):
        self._open_windows.append(win)

    def _load_window(self, window_name):
        win = self.SUB_WINDOWS[window_name]()
        win.show()
        self._add_open_window(win)

    def load_mark_video_new(self):
        self._load_window('WVideoGroundTruthNewSetup')

    def load_mark_video_existing(self):
        self._load_window('WVideoGroundTruthExistingSetup')

    def load_mosse_interactive(self):
        self._load_window('WTrackerMarking')

    def load_wavelet_interactive(self):
        self._load_window('WVideoFrameWavelet')

    def do_quit(self):
        err = False

        for win in self.open_windows:
            if "close" in win.__dict__ and not win.close():  # Returns True if no issues
                err = True

        if err:
            log.error("There was an error closing the children windows")

        sys.exit(0)

    @property
    def open_windows(self):
        return self._open_windows
