__author__ = 'matt'

import platform

KEY_GT = 'gt'
KEY_GOPRO = 'bangor'
KEY_FOCB = 'focb'


PATHS = {
    'Darwin': {
        KEY_GT: '/Users/matt/Dropbox/CompSci/MPhil/neptune/',
        KEY_GOPRO: '/Users/matt/Documents/MPhil/Video/GoPro/',
        KEY_FOCB: '/Users/matt/Documents/MPhil/Video/FoCB/'
    },

    'Linux': {
        KEY_GT: '/home/matt/MPhil/neptune/',
        KEY_GOPRO: '/home/matt/Videos/GoPro video from fishers sled/',
        KEY_FOCB: '/home/matt/Videos/focb/'
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
