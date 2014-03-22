import cv2

__author__ = 'matt'


def create_capture(source=0):
    source = str(source).strip()
    chunks = source.split(':')

    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        # Handle drive letters in Windows platforms
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]

    try:
        source = int(source)
    except ValueError:
        pass

    params = dict(s.split('=') for s in chunks[1:])
    cap = cv2.VideoCapture(source)

    if 'size' in params:
        w, h = map(int, params['size'].split('x'))

        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, h)

    if cap is None or not cap.isOpened():
        raise SystemError("Unable to open selected video source" + source)

    return cap
