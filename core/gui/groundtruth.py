__author__ = 'matt'

from os import getcwd

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from config import APP_NAME
from core.gui.components import QWVideoPositionSlider
from data.classifications.ccw import CLASSES_CCW, STR_CCW_TITLE, STR_CCW_DESC
from structures.holders import GTMVideoSubstrates, STR_TRUTH_NOTES, STR_TRUTH_KLASS, load_json
from structures.video import create_capture


class WVideoGroundTruthSetupBase(QMainWindow):

    STR_CLICKED = "clicked()"
    STR_SPECIFY_VIDEO = "Path to Video File"
    STR_SPECIFY_DIR = "Output Directory"

    def _set_path(self):
        path = QFileDialog.getOpenFileName(self,
                self.STR_SPECIFY_VIDEO, self._txt_videopath.text())
        if path:
            self._txt_videopath.setText(QDir.toNativeSeparators(path))

    def _set_save_path(self):
        path = QFileDialog.getExistingDirectory(self,
                self.STR_SPECIFY_DIR)
        if path:
            self._txt_outputpath.setText(QDir.toNativeSeparators(path))
            self._save_path = True

    def _do_cancel(self):
        self.hide()
        self.close()


class WVideoGroundTruthNewSetup(WVideoGroundTruthSetupBase):
    """
    Allow the entry of information to bootstrap a ground-truth marking session.
    """

    STR_WIN_TITLE = "Configure & Launch Ground Truth"

    def __init__(self, video_src=None):
        super(WVideoGroundTruthNewSetup, self).__init__(None)
        self._video_src = video_src
        self._save_path = False
        self._annotator = False
        self._video_path = video_src is not None

        # Widgets
        btn_cancel = QPushButton("&Cancel")
        self.btn_load = QPushButton("&Load")

        # Callbacks
        self.connect(btn_cancel, SIGNAL(self.STR_CLICKED), self._do_cancel)
        self.connect(self.btn_load, SIGNAL(self.STR_CLICKED), self._do_load)

        # Layouts
        lyt_main = QVBoxLayout()
        lyt_main.setMargin(10)
        lyt_main.setSpacing(10)
        lyt_main.setAlignment(Qt.AlignTop)
        lyt_main.addWidget(self._create_top_path_group())
        lyt_main.addWidget(self._create_mid_markup_group())
        lyt_main.addWidget(self._create_btm_output_group())

        lyt_btm = QHBoxLayout()
        lyt_btm.addWidget(btn_cancel)
        lyt_btm.addWidget(self.btn_load)

        lyt_main.addLayout(lyt_btm)

        widget = QWidget()
        widget.setLayout(lyt_main)

        self.statusBar().showMessage("Awaiting Session Setup")
        self.setCentralWidget(widget)
        self.setGeometry(50, 50, 550, 200)
        self.setWindowTitle(APP_NAME + " - " + self.STR_WIN_TITLE)

    def _create_top_path_group(self):
        group_box = QGroupBox("Input Video Source")
        lbl_path = QLabel("Path")
        self._txt_videopath = QLineEdit(self._video_src)
        btn_path = QPushButton("...")
        btn_path.setFixedWidth(30)

        # Callbacks
        self.connect(btn_path, SIGNAL(self.STR_CLICKED), self._set_path)

        # Layouts
        lyt_top = QHBoxLayout()
        lyt_top.addWidget(lbl_path)
        lyt_top.addWidget(self._txt_videopath)
        lyt_top.addWidget(btn_path)

        vbox = QVBoxLayout()
        vbox.addLayout(lyt_top)
        vbox.addStretch(1)
        group_box.setLayout(vbox)

        return group_box

    def _create_mid_markup_group(self):
        group_box = QGroupBox("Markup Options")
        lbl_annotator = QLabel("Annotator:")
        lbl_frameskip = QLabel("Frameskip:")
        lbl_frameskip_info = QLabel("* Frameskip of 250 ~= 30 seconds")
        self._txt_annotator = QLineEdit()
        self._txt_frameskip = QLineEdit(str(GTMVideoSubstrates.DEF_FRAMESKIP))
        
        rdo_class_ccw = QRadioButton('CCW', self)
        rdo_class_ccw.setChecked(True)

        rdo_class_int = QRadioButton('INT', self)
        rdo_class_int.setChecked(True)

        # Layouts
        lyt_top = QHBoxLayout()
        lyt_top.addWidget(lbl_annotator)
        lyt_top.addWidget(self._txt_annotator)

        lyt_mid = QHBoxLayout()
        lyt_mid.addWidget(lbl_frameskip)
        lyt_mid.addWidget(self._txt_frameskip)

        vbox = QVBoxLayout()
        vbox.addLayout(lyt_top)
        vbox.addLayout(lyt_mid)
        vbox.addWidget(lbl_frameskip_info)
        vbox.addWidget(rdo_class_ccw)
        vbox.addWidget(rdo_class_int)
        vbox.addStretch(1)
        group_box.setLayout(vbox)

        return group_box

    def _create_btm_output_group(self):
        group_box = QGroupBox("Output Options")
        lbl_save_path = QLabel("Directory")
        self._txt_outputpath = QLineEdit()
        self._txt_outputpath.setEnabled(False)
        btn_save_path = QPushButton("...")
        btn_save_path.setFixedWidth(30)
        lbl_save_path_info = QLabel("* Directory to save the output file")

        # Callbacks
        self.connect(btn_save_path, SIGNAL(self.STR_CLICKED),
                     self._set_save_path)

        # Layouts
        lyt_btm = QHBoxLayout()
        lyt_btm.addWidget(lbl_save_path)
        lyt_btm.addWidget(self._txt_outputpath)
        lyt_btm.addWidget(btn_save_path)

        vbox = QVBoxLayout()
        vbox.addLayout(lyt_btm)
        vbox.addWidget(lbl_save_path_info)
        vbox.addStretch(1)
        group_box.setLayout(vbox)

        return group_box

    def _do_load(self):
        path = str(self._txt_videopath.text())
        annotator = str(self._txt_annotator.text())
        frameskip = int(self._txt_frameskip.text())
        output_dir = str(self._txt_outputpath.text())

        self._annotator = len(annotator) > 1
        self._video_path = len(path) > 1

        if not self._save_path:
            self.statusBar().showMessage("No output directory has been set")
            return
        elif not self._annotator:
            self.statusBar().showMessage("No annotator name has been set")
            return
        elif not self._video_path:
            self.statusBar().showMessage("No input video path has been set")
            return

        self.statusBar().showMessage("All OK, loading the markup window")
        self._markup_win = WVideoSubstrateGroundTruthing(path, output_dir,
                                                         annotator, frameskip)
        self._markup_win.show()


class WVideoGroundTruthExistingSetup(WVideoGroundTruthSetupBase):
    """
    Restores a previous session, allowing the annotator to review / correct
    classifications. Requires the path to the saved JSON as its primary argument.
    """
    STR_WIN_TITLE = "Modify Existing Session"

    __input_mask = "JSON (*.json *.JSON)"

    def __init__(self):
        super(WVideoGroundTruthExistingSetup, self).__init__(None)
        self._json_src = self._get_file_path()
        self._json = load_json(self._json_src)
        self._klass = GTMVideoSubstrates

        # Widgets
        btn_cancel = QPushButton("&Cancel")
        btn_load = QPushButton("&Load")

        # Callbacks
        self.connect(btn_cancel, SIGNAL(self.STR_CLICKED), self._do_cancel)
        self.connect(btn_load, SIGNAL(self.STR_CLICKED), self._do_load)

        # Layouts
        lyt_main = QVBoxLayout()
        lyt_main.setMargin(10)
        lyt_main.setSpacing(10)
        lyt_main.setAlignment(Qt.AlignTop)
        lyt_main.addWidget(self._create_grp_confirmation())
        
        lyt_buttons = QHBoxLayout()
        lyt_buttons.addWidget(btn_cancel)
        lyt_buttons.addWidget(btn_load)

        lyt_main.addLayout(lyt_buttons)

        widget = QWidget()
        widget.setLayout(lyt_main)

        self.setCentralWidget(widget)
        self.setWindowTitle(APP_NAME + " - " + self.STR_WIN_TITLE)

    def _create_grp_confirmation(self):
        group_box = QGroupBox("Confirm Parameters")
        lbl_annotator = QLabel("Annotator:")
        lbl_frameskip = QLabel("Frameskip:")
        lbl_videopath = QLabel("Video Path:")
        lbl_outputpath = QLabel("Output Path:")
        self._txt_annotator = QLineEdit(self._json[self._klass.STR_AUTH])
        self._txt_frameskip = QLineEdit(str(GTMVideoSubstrates.DEF_FRAMESKIP))
        self._txt_videopath = QLineEdit(self._json[self._klass.STR_FILE])
        self._txt_outputpath = QLineEdit(getcwd())

        btn_path = QPushButton("...")
        btn_path.setFixedWidth(30)
        btn_save_path = QPushButton("...")
        btn_save_path.setFixedWidth(30)

        # Callbacks
        self.connect(btn_save_path, SIGNAL(self.STR_CLICKED),
                     self._set_save_path)
        self.connect(btn_path, SIGNAL(self.STR_CLICKED), self._set_path)

        # Layouts
        lyt_top = QHBoxLayout()
        lyt_top.addWidget(lbl_annotator)
        lyt_top.addWidget(self._txt_annotator)

        lyt_mid = QHBoxLayout()
        lyt_mid.addWidget(lbl_frameskip)
        lyt_mid.addWidget(self._txt_frameskip)

        lyt_vid = QHBoxLayout()
        lyt_vid.addWidget(lbl_videopath)
        lyt_vid.addWidget(self._txt_videopath)
        lyt_vid.addWidget(btn_path)

        lyt_out = QHBoxLayout()
        lyt_out.addWidget(lbl_outputpath)
        lyt_out.addWidget(self._txt_outputpath)
        lyt_out.addWidget(btn_save_path)

        vbox = QVBoxLayout()
        vbox.addLayout(lyt_top)
        vbox.addLayout(lyt_mid)
        vbox.addLayout(lyt_vid)
        vbox.addLayout(lyt_out)
        vbox.addStretch(1)
        group_box.setLayout(vbox)

        return group_box

    def _get_file_path(self):
        return QFileDialog.getOpenFileName(self, "Load JSON Markup File",
                                           getcwd(), self.__input_mask)

    def _do_load(self):
        path = str(self._txt_videopath.text())
        output_dir = str(self._txt_outputpath.text())
        annotator = str(self._txt_annotator.text())
        frameskip = int(self._txt_frameskip.text())

        self._markup_win = WVideoSubstrateGroundTruthing(path,
                                                         output_dir,
                                                         annotator,
                                                         frameskip,
                                                         self._json)
        self._markup_win.show()


### VIDEO FRAME GROUND TRUTHING ###

class WVideoSubstrateGroundTruthing(QMainWindow):
    """
    Presents the user with images, requesting classification
    """

    STR_WIN_TITLE = 'Underwater Video Substrate Ground-Truthing'
    STR_CLK = 'clicked()'
    STR_BTN_PLAY = '|>'
    STR_BTN_PAUSE = '||'

    def __init__(self, video_src, output_dir, annotator, frameskip=None,
                 existing_data=None):
        super(WVideoSubstrateGroundTruthing, self).__init__(None)
        self._capture = create_capture(video_src)
        self._frameskip = frameskip
        self._holder = GTMVideoSubstrates(self._capture, video_src,
                                                       output_dir, annotator,
                                                       existing_data)
        self._annotator = annotator
        self._frame_num = 0
        self._frames_num_total = self._holder.converter.num_frames
        self._create_video_frame()
        self._playback_timer = QTimer(self)
        self._playback_speed = 27
        self._playing = False
        self._frames_with_data = self._holder.frame_numbers_with_data
        self._classes_dict = {}

        # Layouts
        lyt_mid = QHBoxLayout()
        lyt_video = QVBoxLayout()
        lyt_video.setAlignment(Qt.AlignTop)
        lyt_video.addWidget(self._frm_video)
        lyt_video.addLayout(self._build_step_buttons())
        lyt_mid.addLayout(lyt_video)
        lyt_mid.addWidget(self._build_classification_box())
        lyt_btm = QHBoxLayout()
        lyt_main = QVBoxLayout()
        lyt_main.setMargin(10)
        lyt_main.setSpacing(10)
        lyt_main.setAlignment(Qt.AlignTop)
        lyt_main.addLayout(lyt_mid)
        lyt_main.addLayout(lyt_btm)

        widget = QWidget()
        widget.setLayout(lyt_main)

        self.setCentralWidget(widget)
        self.setWindowTitle(APP_NAME + " - " + self.STR_WIN_TITLE)
        self.update()
        self._show_next_frame()

    def _create_video_frame(self):
        self._frm_video = QLabel()  # Video frame
        self._frm_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._frm_video.setFrameShape(QFrame.Box)
        self._frm_video.setAlignment(Qt.AlignTop)
        self._frm_video.setFixedWidth(640)
        self._frm_video.setFixedHeight(480)
        self._frm_video.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self._frm_video.setScaledContents(True)

    # MEDIA BUTTON CONTROLS

    def _build_step_buttons(self):
        """
        Build the media controls to handle the VideoCapture / GroundTruth
        holders.
        """
        lyt_overall = QVBoxLayout()
        lyt_media = QHBoxLayout()
        self._lcd_frame = QLCDNumber(self)
        self._sld_pos = QWVideoPositionSlider(Qt.Horizontal, self, self.pause,
                                              self.unpause)
        self._sld_pos.setRange(0, self._frames_num_total)
        media_btn_n = 30

        btn_skip_back = QPushButton("<<")
        btn_step_back = QPushButton("<")
        self._btn_play = QPushButton(self.STR_BTN_PLAY)
        btn_step_forward = QPushButton(">")
        btn_skip_forward = QPushButton(">>")

        self.connect(btn_skip_back, SIGNAL(self.STR_CLK), self._skip_back)
        self.connect(btn_step_back, SIGNAL(self.STR_CLK), self._step_back)
        self.connect(self._btn_play, SIGNAL(self.STR_CLK), self._toggle_play)
        self.connect(btn_step_forward, SIGNAL(self.STR_CLK),
                     self._step_forward)
        self.connect(btn_skip_forward, SIGNAL(self.STR_CLK),
                     self._skip_forward)

        buttons = [btn_skip_back, btn_step_back, self._btn_play,
                   btn_step_forward, btn_skip_forward]

        for btn in buttons:
            btn.setFixedWidth(media_btn_n + (media_btn_n * 2))
            btn.setFixedHeight(media_btn_n)
            lyt_media.addWidget(btn)

        lyt_media.addWidget(self._lcd_frame)
        lyt_overall.addWidget(self._sld_pos)
        lyt_overall.addLayout(lyt_media)

        return lyt_overall

    def __find_in_range(self, rang, default):
        candidate = None
        rang = [x for x in rang if x >= 0]
        # Note, the above was changed. Initially it was assigning to range,
        # so the comprehension was being ignored
        # @todo: check this hasn't messed things up.

        for x in rang:
            if x in self._frames_with_data:
                candidate = x
                break

        next_frame = candidate if candidate is not None else default
        self._set_frame(next_frame)

    def _skip_back(self):
        """
        Skips backward in time to either the next frameskip interval,
        or a ground truth. Whichever comes first.
        """
        curr = self._frame_num
        skip = self._frameskip
        mini = curr - skip
        rang = range(curr - 1, mini, -1)
        self.__find_in_range(rang, mini)

    def _skip_forward(self):
        """
        Skips forward in time to either the next frameskip interval,
        or a ground truth. Whichever comes first.
        """
        curr = self._frame_num
        skip = self._frameskip
        maxi = curr + skip
        rang = range(curr + 1, maxi)
        self.__find_in_range(rang, maxi)

    def _step_back(self):
        """
        Steps back a single frame
        """
        if self.at_start:
            return

        self._set_frame(self._frame_num - 1)

    def _toggle_play(self):
        if not self._playing:
            self._playback_timer = QTimer(self)
            self._playback_timer.timeout.connect(self._step_forward)
            self._playback_timer.start(self._playback_speed)
            self.unpause()
        else:
            self._playback_timer.stop()
            self.pause()
    
    def pause(self):
        self._playing = False
        self._btn_play.setText(self.STR_BTN_PLAY)

    def unpause(self):
        self._playing = True
        self._btn_play.setText(self.STR_BTN_PAUSE)

    def _step_forward(self):
        """
        Steps forward a single frame
        """
        if self.at_end:
            return

        self._set_frame(self._frame_num + 1)

    # END MEDIA BUTTON CONTROLS

    def _build_classification_box(self):
        """
        This deals with the right-hand groupbox & frame, specifically
        all controls relating to the marking of ground truth are either defined,
        or called here.
        """
        grp_classification = QGroupBox("Frame Classification")
        grp_classification.setAlignment(Qt.AlignTop)
        self._txt_desc = QTextEdit()
        self._txt_note = QTextEdit()
        self._cmb_classifications = QComboBox(self)
        self._cmb_classifications.setFixedWidth(350)
        self._cmb_classifications.currentIndexChanged[str].connect(self._update_text)

        # Add them in order:
        for key in sorted(CLASSES_CCW.keys()):
            line = str(key) + ": " + CLASSES_CCW[key][STR_CCW_TITLE]
            self._cmb_classifications.addItem(line)
            self._classes_dict[key] = line

        # Group Buttons
        self._btn_next = QPushButton("Commit Classification")
        self._btn_delt = QPushButton("&Delete This Classification")
        self._btn_delt.setEnabled(False)
        self._btn_exit = QPushButton("&Save and Exit")
        self.connect(self._btn_next, SIGNAL(self.STR_CLK), self._commit_frame)
        self.connect(self._btn_exit, SIGNAL(self.STR_CLK), self._save_and_exit)
        self.connect(self._btn_delt, SIGNAL(self.STR_CLK), self._delete_frame)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        hbox = QHBoxLayout()

        vbox.addWidget(self._cmb_classifications)
        vbox.addWidget(QLabel("Class Description:"))
        vbox.addWidget(self._txt_desc)
        vbox.addWidget(QLabel("Areas of Interest:"))
        vbox.addWidget(self._txt_note)
        vbox.addWidget(self._btn_next)
        hbox.addWidget(self._btn_delt)
        hbox.addWidget(self._btn_exit)
        vbox.addLayout(hbox)
        grp_classification.setLayout(vbox)

        # Final bindings
        self._sld_pos.valueChanged.connect(self._set_frame)

        return grp_classification

    # QT DATA EXTRACTION

    def _get_selected_class(self, text):
        """
        Goes through the ComboBox of classes, extracting the numeric ID which
        precedes the colon on the current line.
        """
        parts = text.split(':')
        return int(parts[0])

    def _update_text(self, text):
        if len(text) == 0:
            return False

        klass = self._get_selected_class(text)
        descr = CLASSES_CCW[klass]
        self._txt_desc.setText(descr[STR_CCW_DESC])

    def _set_current_class(self, klass):
        target = self._classes_dict[klass]
        id_ref = self._cmb_classifications.findText(target)
        self._cmb_classifications.setCurrentIndex(id_ref)

    # STEPPING & SAVING

    def update(self):
        frame_has_data = self._holder.has_truth_for(self._frame_num)
        self._btn_delt.setEnabled(frame_has_data)

        update_text = ""

        if frame_has_data:
            key_notes, key_class = STR_TRUTH_NOTES, STR_TRUTH_KLASS
            truth = self._holder.get_truth_for(self._frame_num)
            notes, klass = truth[key_notes], truth[key_class]

            if notes is not None:
                update_text = notes

            self._set_current_class(klass)
        
        self._txt_note.setText(update_text)

    def _set_frame(self, frame_num):
        if frame_num >= 0 and frame_num <= self._frames_num_total:
            self._holder.converter.set_current_frame(frame_num)
            self.update()
            self._show_next_frame()

    def _mark_frame_with_notes(self, frame):
        """
        Keeps an internal register of the frames that have notes marked out.
        """
        self._frames_with_data.append(self._frame_num)
        self._frames_with_data = list(set(self._frames_with_data))

    def _rem_frame_from_notes(self, frame_num):
        """
        Removes a frame from the register for having notes.
        """
        self._frames_with_data = [x for x in self._frames_with_data if x != frame_num]

    def _commit_frame(self):
        """
        In this method, we record all the data internally before invoking
        the call to display the next frame, and reinitialise the notes area
        """
        notes = self._txt_note.toPlainText()
        line = str(self._cmb_classifications.currentText())
        klass = self._get_selected_class(line)

        self._holder.add_truth(self._frame_num, klass, notes)
        self.update()
        self._mark_frame_with_notes(self._frame_num)

    def _delete_frame(self):
        """
        This can only get called if the current frame is one that has truth recorded.
        If so, delete it and update.
        """
        self._holder.remove_truth_for(self._frame_num)
        self.update()
        self._rem_frame_from_notes(self._frame_num)

    def _save(self):
        """
        Save the current state of the annotations. Irrespective if complete or not.
        """
        file_path = self._holder.save_file()
        QMessageBox.information(self, 'File Saved', "Saved to\n" + file_path)

    def _save_and_exit(self):
        """
        Performs as _save, but kills the window too
        @todo: prompt if this should be marked as complete or not
        """
        response = QMessageBox.question(self, 'Mark as Complete?',
                                        'Would you like to mark this session as complete?',
                                        QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes,
                                        QMessageBox.No)

        if response == QMessageBox.Cancel:
            return
        elif response == QMessageBox.Yes:
            self._holder.set_complete()

        self._save()
        self.hide()
        self.close()

    def _show_next_frame(self):
        """
        Gets the next frame and renders it to the QLabel widget
        """
        self._frame_num = self._holder.frame_number
        frame = self._holder.next_frame

        self._frm_video.setPixmap(frame)
        self._lcd_frame.display(self._frame_num)
        self._sld_pos.setValue(self._frame_num)

    @property
    def at_start(self):
        return self._frame_num == 0

    @property
    def at_end(self):
        return self._frame_num == self._frames_num_total
