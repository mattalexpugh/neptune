import numpy as np
import cv2

from mahotas.features.lbp import lbp


__author__ = 'matt'


## GENERAL WAVELET METHODS


def gabor_2d_kernel(xy=21, sigma=5, theta=0, lmbda=50, psi=90):
    """
    Generates a Gabor function in 2D spatial domain

        xy      Window size
        sigma   Gaussian enveloping
        theta   Rotation of sinusoid
        lmbda   Phase of sinusoid
        psi     I can't remember
    """

    if not xy % 2:
        exit(1)

    theta = theta * np.pi / 180
    psi = psi * np.pi / 180

    xs = np.linspace(-1., 1., xy)
    ys = np.linspace(-1., 1., xy)

    lmbda = np.float(lmbda)
    x, y = np.meshgrid(xs, ys)
    sigma = np.float(sigma) / xy

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = x * np.sin(theta) + y * np.cos(theta)

    term1 = np.exp(-0.5 * (x_theta ** 2 + y_theta ** 2 ) / sigma ** 2)
    term2 = np.cos(2. * np.pi * x_theta / lmbda + psi)

    return np.array(term1 * term2, dtype=np.float32)


def gabor_2d_transform(image, xy, sigma, theta, lmbda, psi):
    """
    Creates a kernel as per input params and directly
    convolves it with image.
    """
    kernel = gabor_2d_kernel(xy, sigma, theta, lmbda, psi)
    return gabor_2d_with_kernel(image, kernel)


def gabor_2d_with_kernel(image, kernel):
    """
    Performs the convolution of image with supplied kernel

    Returns m x n np.float32 array
    """
    return cv2.filter2D(image, cv2.CV_32F, kernel)


## EXPERIMENT SPECIFIC METHODS

rotations = [0, 45, 90, 135]
full_frame_kernels = [gabor_2d_kernel(theta=x) for x in rotations]


def gabor_histo_at_4_rotations(image):
    stacked_histo = []

    for kernel in full_frame_kernels:
        gabor_frame = gabor_2d_with_kernel(image, kernel)
        histo = np.histogram(gabor_frame.ravel(), 256, [0, 256])
        stacked_histo.extend(histo)

    hstack = np.hstack(stacked_histo)

    return hstack


## Gabor composites

def gabor_composite_from_thetas(image, histeq=False):
    """
    Processes textures at the four orientations in kernels and adds
    the impulse repsonses together, creating an overall
    representation of frame texture. Strips any colour information.

    Returns m x n np.float64 matrix of responses (where m x n = dim(image))
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if histeq:
        image = cv2.equalizeHist(image)

    return reduce(lambda x, y: x + y, [cv2.filter2D(image, cv2.CV_32F, k) for k in full_frame_kernels])


## Get LBPs of different radius on the impulse response


def gabor_composite_lbp1(image):
    return lbp(gabor_composite_from_thetas(image), radius=1, points=8)


def gabor_composite_lbp2(image):
    return lbp(gabor_composite_from_thetas(image), radius=2, points=12)


def gabor_composite_lbp3(image):
    return lbp(gabor_composite_from_thetas(image), radius=3, points=16)


## Gabor impulse response histograms


def gabor_composite_histogram(image):
    frame = gabor_composite_from_thetas(image)
    return np.histogram(frame.ravel())


METHOD_MAP = {
    'gabor_histo_4_rotations': gabor_histo_at_4_rotations,
    'gabor_composite_4_rotations': gabor_composite_from_thetas,
    'gabor_lbp1': gabor_composite_lbp1,
    'gabor_lbp2': gabor_composite_lbp2,
    'gabor_lbp3': gabor_composite_lbp3,
    'gabor_composite_histogram': gabor_composite_histogram
}
