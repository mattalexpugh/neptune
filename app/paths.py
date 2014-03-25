__author__ = 'matt'

import platform

KEY_GT = 'gt'
KEY_GOPRO = 'gopro'
KEY_FOCB = 'focb'

INTERNAL_GT_PATH = "data/static/json/gt/"


PATHS = {
    'Darwin': {
        KEY_GT: '/Users/matt/Dropbox/CompSci/MPhil/neptune/' + INTERNAL_GT_PATH,
        KEY_GOPRO: '/Users/matt/Documents/MPhil/Video/GoPro/',
        KEY_FOCB: '/Users/matt/Documents/MPhil/Video/FoCB/'
    },

    'Linux': {
        KEY_GT: '/home/matt/MPhil/neptune/' + INTERNAL_GT_PATH,
        KEY_GOPRO: '/home/matt/documents/MPhil/Videos/GoPro/',
        KEY_FOCB: '/home/matt/documents/MPhil/Videos/FoCB/'
    }
}


def get_path(k, current_platform=None):
    """
    Shortcut utility function to deal with platform independence and just return the
    relevant paths.
    """

    if current_platform is None:
        current_platform = platform.system()

    return PATHS[current_platform][k]
