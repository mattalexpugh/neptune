__author__ = 'matt'

import cv2
import numpy as np


from algorithms.correction.colour import retinex_with_adjust


def cv2_calc_hist_gray(frame):
    """
    Converts the frame to grayscale from 3-channel colour
    before deriving the histogram based on 8-bit resolution
    """

    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    return np.int32(np.around([x[0] for x in hist]))


def cv2_calc_hist_bgr(frame):
    """
    Returns an ndarray of n_samples * n_channels (3) as a
    representation of independent histograms ont he B,G,R
    channels of a colour image.
    """
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    hists = []

    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([frame], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        hists.append([x[0] for x in hist])

    return np.hstack([hists[0], hists[1], hists[2]])


def cv2_calc_hist_retinex_gray(frame):
    frame = retinex_with_adjust(frame)
    return cv2_calc_hist_gray(frame)


def cv2_calc_hist_retinex_bgr(frame):
    frame = retinex_with_adjust(frame)
    return cv2_calc_hist_bgr(frame)


def np_calc_hist(frame):
    return np.histogram(frame.ravel(), 256, [0, 256])


METHOD_MAP = {
    'cv2_gray_hist': cv2_calc_hist_gray,
    'cv2_calc_hist_bgr': cv2_calc_hist_bgr,
    'cv2_calc_hist_retinex_gray': cv2_calc_hist_retinex_gray,
    'cv2_calc_hist_retinex_bgr': cv2_calc_hist_bgr,
    'np_calc_hist': np_calc_hist
}