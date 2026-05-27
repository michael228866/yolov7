"""Optional gender classification for detected head crops.

Uses OpenCV's DNN module with the classic Levi-Hassner Caffe gender model.
NOTE: that model was trained on roughly frontal *faces*. On the overhead head
crops typical of a people-counter it is unreliable — treat the label as a
rough guess, and swap in your own model via classify() if you need accuracy.

The ~45 MB model is downloaded on first use (only when --api-url is set). If
the files are missing and cannot be downloaded, classify() returns 'unknown'
so the rest of the pipeline keeps working.
"""

import logging
import urllib.request
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
PROTO_PATH = HERE / 'gender_deploy.prototxt'
MODEL_PATH = HERE / 'gender_net.caffemodel'

PROTO_URL = 'https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt'
MODEL_URL = 'https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel'

GENDER_LABELS = ['Male', 'Female']
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


class GenderClassifier:
    """Classify a head/face crop (BGR ndarray) as 'Male' / 'Female' / 'unknown'."""

    def __init__(self, download=True):
        self.net = None
        if download and not (PROTO_PATH.exists() and MODEL_PATH.exists()):
            self._download()
        if PROTO_PATH.exists() and MODEL_PATH.exists():
            try:
                # Read bytes in Python so non-ASCII paths work: OpenCV's native
                # file readers fail to open Unicode paths on Windows.
                self.net = cv2.dnn.readNetFromCaffe(PROTO_PATH.read_bytes(),
                                                    MODEL_PATH.read_bytes())
                logging.info('Gender model loaded.')
            except cv2.error as e:
                logging.warning(f'Gender model failed to load ({e}); gender will be "unknown".')
        else:
            logging.warning('Gender model files unavailable; gender will be "unknown".')

    @staticmethod
    def _download():
        for url, path, note in [(PROTO_URL, PROTO_PATH, ''),
                                (MODEL_URL, MODEL_PATH, ' (~45 MB, one-time)')]:
            if path.exists():
                continue
            try:
                logging.info(f'Downloading gender model: {path.name}{note} ...')
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                logging.warning(f'Could not download {path.name} ({e}).')

    def classify(self, crop):
        if self.net is None or crop is None or getattr(crop, 'size', 0) == 0:
            return 'unknown'
        try:
            blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), MODEL_MEAN, swapRB=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            return GENDER_LABELS[int(preds[0].argmax())]
        except cv2.error:
            return 'unknown'
