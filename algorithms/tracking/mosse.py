import numpy as np
import cv2

__doc__ = """
Bolme's MOSSE tracker.

Original tracker based on implementation provided by OpenCV group (BSD License)
"""

### MODULE CONSTANTS ###

eps = 1e-5

### UTIL FUNCTIONS ###

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand() - 0.5) * coef
    c, s = np.cos(ang), np.sin(ang)

    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5) * coef
    c = (w/2, h/2)
    T[:, 2] = c - np.dot(T[:2, :2], c)

    return cv2.warpAffine(a, T, (w, h), borderMode=cv2.BORDER_REFLECT)


def div_spec(A, B):
    Ar, Ai = A[..., 0], A[..., 1]
    Br, Bi = B[..., 0], B[..., 1]
    C = (Ar + 1j * Ai) / (Br + 1j * Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()

    return C


def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0), thickness=1, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), lineType=cv2.CV_AA)


### CLASSES ###

class MOSSETracker(object):

    MIN_PSR = 8.0
    ID = 'MSB'

    def __str__(self):
        return self.get_id()

    def __init__(self, frame, rect):
        x1, y1, x2, y2 = rect # Extract coordinates of the rectangle to be tracked
        w, h = map(cv2.getOptimalDFTSize, [x2 - x1, y2 - y1])
        x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2

        # 0.5 * x \in [w, h] (-1) = the centre point of the region
        self.pos = x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
        self.size = w, h

        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        # Hanning Window
        # http://en.wikipedia.org/wiki/Window_function
        # http://en.wikipedia.org/wiki/Window_function#Hann_.28Hanning.29_window
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)

        g = np.zeros((h, w), np.float32)
        g[h // 2, w // 2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, None, cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)

        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, None, cv2.DFT_COMPLEX_OUTPUT)

            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(A, A, 0, conjB=True)

        self.update_kernel()
        self.update(frame)
        self.id = id(self)

    def update_kernel(self):
        self.H = div_spec(self.H1, self.H2)
        self.H[..., 1] *= -1

    def preprocess(self, img):
        img = np.log(np.float32(img) + 1.0)
        img = (img - img.mean()) / (img.std() + eps)

        return img * self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT),
                             self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval - smean) / (sstd + eps)

        return resp, (mx - w // 2, my - h // 2), psr

    def update(self, frame, rate=0.125, img_override=None):
        (x, y), (w, h) = self.pos, self.size

        if img_override is None:
            self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        else:
            self.last_img = img = img_override

        img = self.preprocess(img)

        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > self.MIN_PSR

        if not self.good:
            return

        self.pos = x + dx, y + dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(A, A, 0, conjB=True)

        self.H1 = self.H1 * (1.0 - rate) + H1 * rate
        self.H2 = self.H2 * (1.0 - rate) + H2 * rate

        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = f.shape

        f = np.roll(f, -h // 2, 0)
        f = np.roll(f, -w // 2, 1)

        kernel = np.uint8((f - f.min()) / f.ptp() * 255)
        resp = self.last_resp
        resp = np.uint8(np.clip(resp / resp.max(), 0, 1) * 255)
        vis = np.vstack([self.last_img, kernel, resp])

        x, y = 5, 10

        draw_str(vis, (x, y), "IMAGE")
        draw_str(vis, (x, y + h), "KERNEL")
        draw_str(vis, (x, y + 2 * h), "FRESP")

        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)
        red = (0, 0, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), red)

        if self.good:  # Tracking successful, pinpoint centre with circle
            cv2.circle(vis, (int(x), int(y)), 2, red, -1)
        else:  # Not successful, cross through the rectangle
            cv2.line(vis, (x1, y1), (x2, y2), red)
            cv2.line(vis, (x2, y1), (x1, y2), red)

        # Writing to the screen here
        draw_str(vis, (x1, y2 + 12), "PSR: %.2f" % self.psr)
        draw_str(vis, (x1, y2 + 24), "ID: " + self.get_id())

    def get_id(self):
        return self.ID + str(self.id)


class HistEqMOSSETracker(MOSSETracker):

    ID = 'HEQ'

    def get_eq_region(self, frame):
        (x, y), (w, h) = self.pos, self.size
        return cv2.equalizeHist(cv2.getRectSubPix(frame, (w, h), (x, y)))

    def update(self, frame, rate=0.125):
        eq_frame = cv2.equalizeHist(frame);
        super(HistEqMOSSETracker, self).update(eq_frame, rate, self.get_eq_region(eq_frame))


class ABMixMOSSETracker(MOSSETracker):

    def __init__(self, frame, rect, ratio=0.5):
        super(ABMixMOSSETracker, self).__init__(frame, rect)
        self.set_ratio(ratio)

    def set_ratio(self, ratio):
        self.ratio = ratio

    def update(self, frame, rate=0.125, img_override=None):
        last_img = self.last_img
        mixed_frame = last_img
        super(ABMixMOSSETracker, self).update(frame, rate, mixed_frame)


class HSVMOSSETracker(MOSSETracker):

    ID = 'HSV'

    pass


class HSIMOSSETracker(MOSSETracker):

    ID = 'HSI'

    pass


class RGBSingleChannelMOSSETracker(MOSSETracker):

    ID = 'RGB'

    pass


class RGBRMOSSETracker(RGBSingleChannelMOSSETracker):

    ID = 'RGB_R'

    pass


class RGBGMOSSETracker(RGBSingleChannelMOSSETracker):

    ID = 'RGB_G'

    pass


class RGBBMOSSETracker(RGBSingleChannelMOSSETracker):

    ID = 'RGB_B'

    pass


TRACKER_TYPES = {
    'MOSSE': MOSSETracker,
    'HistEq': HistEqMOSSETracker,
    'HSI': HSIMOSSETracker,
    'HSV': HSVMOSSETracker,
    'RGB R': RGBRMOSSETracker,
    'RGB G': RGBGMOSSETracker,
    'RGB B': RGBBMOSSETracker
}
