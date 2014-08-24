__author__ = 'matt'

from cv2 import cvtColor, COLOR_BGR2RGB, cv
import numpy as np
from math import ceil
from PyQt4.QtGui import QImage, QPixmap


def convert_single_frame(frame):
    try:
        h, w = frame.shape[:2]
        img = QImage(frame, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        return img
    except:
        return None


class VideoFrameConverter():
    """
    This wraps around an OpenCV VideoCapture object, converting a frame
    to a QPixmap image for use in a QLabel. Bridges the gap between cv2
    & PyQt4
    """

    def __init__(self, capture):
        self.capture = capture
        self._current_frame = np.array([])
        self.num_frames = capture.get(cv.CV_CAP_PROP_FRAME_COUNT)

    def capture_next_frame(self, returnFrame=False):
        ret, read_frame = self.capture.read()

        if ret:
            self._current_frame = cvtColor(read_frame, COLOR_BGR2RGB)

        if returnFrame:
            return self._current_frame

    def convert_frame(self):
        frame = self.current_frame

        try:
            h, w = frame.shape[:2]
            self.previous_frame = frame
            img = QImage(frame, w, h, QImage.Format_RGB888)
            return QPixmap.fromImage(img)
        except:
            return None

    def get_frame_mxn_patches_list(self, m, n):
        return [(x, y) for x, y in self.get_frame_mxn_patches_generator(m, n)]

    def get_frame_mxn_patches_generator(self, m, n):
        """
        Returns a patched segmentation of the current frame as a generator:

            m x n  (m = cols, n = rows)
        """

        frame = self.current_frame
        dimensions = frame.shape

        m_in_x = dimensions[1] / m
        n_in_y = dimensions[0] / n

        for x in range(m_in_x - 1):
            x1_tmp = x * m
            x2_tmp = (x + 1) * m

            for y in range(n_in_y - 1):
                y1_tmp = y * n
                y2_tmp = (y + 1) * n

                window = frame[y1_tmp:y2_tmp, x1_tmp:x2_tmp]
                window_dimensions = tuple(window.shape[:2])

                if window_dimensions == (m, n):
                    # Ignore those on the boundaries smaller than mxn
                    id_str = "{}-{}-{}-{}".format(
                        x1_tmp, x2_tmp,
                        y1_tmp, y2_tmp
                    )

                    yield window, id_str

    def get_next_frame(self):
        self.capture_next_frame()
        return self.convert_frame()

    def set_current_frame(self, i):
        self.capture.set(cv.CV_CAP_PROP_POS_FRAMES, ceil(i))

    @property
    def frame_number(self):
        return self.capture.get(cv.CV_CAP_PROP_POS_FRAMES) - 1

    @property
    def current_frame(self):
        if self._current_frame.shape[0] == 0:
            self.capture_next_frame()

        return self._current_frame


## TESTING STUFF
if __name__ == '__main__':
    from structures.video import create_capture
    from algorithms.features.wavelets import gabor_2d_with_kernel, full_frame_kernels
    import app.paths as p
    import os
    import cv2
    from numpy import histogram, interp

    i_max = 255
    i_max_half = i_max / 2
    i_valid_range = 50
    i_valid_max = i_max_half + i_valid_range
    i_valid_min = i_max_half - i_valid_range

    def is_in_target_range(patch):
        mean = np.mean(patch)
        v_min = mean > i_valid_min
        v_max = mean < i_valid_max

        return v_min and v_max

    def get_video_name_stub():
        video_name = "Site 10.MP4"
        return ''.join(video_name.split('.')[:-1])

    def get_test_patches(video_name="Site 10.MP4", window=75, frame_num=10000):
        path_gp = p.get_path(p.KEY_GOPRO)
        fp2 = path_gp + video_name
        cap = create_capture(fp2)
        conv = VideoFrameConverter(cap)
        conv.set_current_frame(frame_num)

        return conv.get_frame_mxn_patches_list(window, window), conv.current_frame

    def get_gs_patches_in_range(tg_patches=None):
        tg_patches = get_test_patches() if tg_patches is None else tg_patches
        tg_gs_patches = [(cv2.cvtColor(x[0], cv2.COLOR_BGR2GRAY), x[1]) for x in tg_patches]

        return [x for x in tg_gs_patches if is_in_target_range(x[0])]

    def histeq(im,nbr_bins=256):

       #get image histogram
       imhist, bins = histogram(im.flatten(),nbr_bins,normed=True)
       cdf = imhist.cumsum() #cumulative distribution function
       cdf = 255 * cdf / cdf[-1] #normalize

       #use linear interpolation of cdf to find new pixel values
       im2 = interp(im.flatten(),bins[:-1],cdf)

       return im2.reshape(im.shape)

    kl2fr = {
        16: [6000, 17200, 20900],
        18: [19500, 21300],
        24: [30000, 30100, 31000],
        27: [24000, 25000]
    }

    video_name = "Site 10.MP4"
    window = 50
    fpstub = '/Users/matt/Downloads/' if p.IS_MAC else '/home/matt/downloads/'

    for klass in kl2fr.keys():
        for frame_num in kl2fr[klass]:
            video_name_stub = get_video_name_stub()
            patches, frame = get_test_patches(video_name, window, frame_num)
            gs_patches = get_gs_patches_in_range(patches)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            printed_frames = []

            for patch, descriptor in gs_patches:

                #if not is_in_target_range(patch):
                #    continue

                mean = np.mean(patch)

                base_stub = fpstub + 'windows/{0}/{2}x{2}/{5}_{1}/'.format(
                    video_name_stub, frame_num, window, descriptor, mean, klass
                )

                fp = base_stub + 'm{}_{}.png'.format(mean, descriptor)
                histeqfp = base_stub + 'm_histeq{}_{}.png'.format(mean, descriptor)
                histeqpatch = cv2.equalizeHist(patch)

                if not os.path.exists(os.path.dirname(fp)):
                    os.makedirs(os.path.dirname(fp))

                gfp = base_stub + 'gab{}_{}.png'.format(mean, descriptor)
                geqfp = base_stub + 'gab_histeq{}_{}.png'.format(mean, descriptor)



                gpatch = gabor_2d_with_kernel(patch, full_frame_kernels[0])

                for k in full_frame_kernels[1:]:
                    gpatch += gabor_2d_with_kernel(patch, k)

                gabhisteqpatch = histeq(gpatch)

                retval = cv2.imwrite(fp, patch)
                retval = cv2.imwrite(histeqfp, histeqpatch)
                retval = cv2.imwrite(gfp, gpatch)
                retval = cv2.imwrite(geqfp, gabhisteqpatch)

                if frame_num not in printed_frames:
                    fp = base_stub + "frame_{}.png".format(frame_num)
                    cv2.imwrite(fp, frame)

                    eqed = cv2.equalizeHist(frame)
                    eqfp = base_stub + "frameeq_{}.png".format(frame_num)
                    cv2.imwrite(eqfp, eqed)

                    printed_frames.append(frame_num)