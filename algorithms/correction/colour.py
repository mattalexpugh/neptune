import numpy as np
import cv2


__author__ = 'matt'
__doc__ = """
Colour correction algorithms. Ripped from colorcorrect package (wanted to avoid PIL as a requirement)
from https://bitbucket.org/aihara/colorcorrect/
"""

def retinex(nimg, cv2flip=True):
    if cv2flip:
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)

    return nimg.transpose(1, 2, 0).astype(np.uint8)


def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0] ** 2)

    max_r = nimg[0].max()
    max_r2 = max_r ** 2

    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()

    coefficient = np.linalg.solve(np.array([[sum_r2, sum_r], [max_r2, max_r]]),
                                  np.array([sum_g, max_g]))

    nimg[0] = np.minimum((nimg[0] ** 2) * coefficient[0] + nimg[0] * coefficient[1], 255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1] ** 2)
    max_b = nimg[1].max()
    max_b2 = max_r ** 2
    coefficient = np.minimum(np.linalg.solve(np.array([[sum_b2, sum_b], [max_b2, max_b]]),
                                             np.array([sum_g, max_g])), 255)
    nimg[1] = (nimg[1] ** 2) * coefficient[0] + nimg[1] * coefficient[1]

    return nimg.transpose(1, 2, 0).astype(np.uint8)


def retinex_with_adjust(nimg):
    return retinex_adjust(retinex(nimg))


def col_to_bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


METHOD_MAP = {
    'retinex': retinex,
    'retinex_with_adjust': retinex_with_adjust
}
