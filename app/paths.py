__author__ = 'matt'

import platform

KEY_GT = 'gt'
KEY_GOPRO = 'gopro'
KEY_FOCB = 'focb'

__curr_sys = platform.system()

IS_MAC = platform.system() == 'Darwin'
IS_MAC = __curr_sys == 'Darwin'
IS_LIN = __curr_sys == 'Linux'
IS_WIN = __curr_sys == 'Windows'
IS_UNIX = IS_MAC or IS_LIN


INTERNAL_GT_PATH = "data/static/json/gt/" if IS_UNIX else 'data\static\json\gt\\'


PATHS = {
    'Darwin': {
        KEY_GT: '/Users/matt/Dropbox/CompSci/MPhil/neptune/' + INTERNAL_GT_PATH,
        KEY_GOPRO: '/Users/matt/Documents/MPhil/Video/GoPro/',
        KEY_FOCB: '/Users/matt/Documents/MPhil/Video/FoCB/'
    },

    'Windows': {
        KEY_GT: 'C:\Users\Matt\Projects\mphil-neptune\\' + INTERNAL_GT_PATH,
        KEY_GOPRO: 'C:\Users\Matt\Documents\MPhil\Video\GoPro\\',
        KEY_FOCB: 'C:\Users\Matt\Documents\MPhil\Video\FoCB\\'
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
