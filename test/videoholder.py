import unittest
import app.paths as p
from structures.video import create_capture
from core.gui.util.containers import VideoFrameConverter


__author__ = 'matt'


class ValidWindow(unittest.TestCase):

    def setUp(self):
        path_gp = p.get_path(p.KEY_GOPRO)
        self.fp = path_gp + "Site 2.MP4"
        self.cap = create_capture(self.fp)
        self.conv = VideoFrameConverter(self.cap)

    def test_site_2(self):
        patches = self.conv.get_frame_mxn_patches_generator(20, 20)
        assert True


if __name__ == "__main__":
    unittest.main()