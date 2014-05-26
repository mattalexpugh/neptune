from mahotas.features.lbp import lbp

__author__ = 'matt'


def lbp_r1(image):
    return lbp(image, radius=1, points=8)


def lbp_r2(image):
    return lbp(image, radius=2, points=12)


def lbp_r3(image):
    return lbp(image, radius=3, points=16)


METHOD_MAP = {
    'lbp1': lbp_r1,
    'lbp2': lbp_r2,
    'lbp3': lbp_r3
}
