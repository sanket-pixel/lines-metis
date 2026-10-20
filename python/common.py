"""Shared image preparation, line extraction and sequence selection."""
import hashlib
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ['distance_normalized', 'orientation_fraction']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def images(sequence='freiburg2_pioneer_slam', frames=100, start=0):
    folder = ROOT / 'data/tum' / sequence
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    selected = paths[start:] if frames is None else paths[start:start + frames]
    if (frames is not None and frames < 1) or start < 0 or not selected:
        raise ValueError('Choose a positive frame count and a valid start index')
    return selected


def preprocess(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(gray[None, None].astype(np.float32) / 255.)


def extract(outputs):
    raise NotImplementedError('Line extraction is unavailable until the C++ extractor is implemented')


def map_lines(lines, shape):
    height, width = shape[:2]
    return (lines + .5) * np.array([width / 320, height / 240]) - .5
