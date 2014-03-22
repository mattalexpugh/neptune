__author__ = 'matt'

from PyQt4.QtGui import QSlider


class QWVideoPositionSlider(QSlider):
    """
    Extension of the QSlider class that automatically executes a
    function on click / release
    """

    def __init__(self, style, parent, press_func, release_func):
        super(QWVideoPositionSlider, self).__init__(style, parent)
        self.parent = parent
        self.press_func = press_func
        self.release_func = release_func

    def mousePressEvent(self, event):
        self.press_func()
        QSlider.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.release_func()
        QSlider.mouseReleaseEvent(self, event)
