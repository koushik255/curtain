"""Locate a query image inside a candidate frame (SIFT + RANSAC).

The result is a similarity transform (move, scale, rotate) mapping query
pixels to frame pixels, and the number of matched points that agree with
it. Many agreeing points (dozens to hundreds) mean the query really comes
from that frame; a wrong frame typically gets under ten.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

WORK_SIDE = 640           # images are matched at most this large
RATIO = 0.75              # Lowe's ratio test
MIN_POINTS = 8            # fewer agreeing points: no box
VERIFIED_POINTS = 15      # at least this many: treat the match as confirmed


@dataclass
class Features:
    keypoints: np.ndarray  # [n, 2] in original-image pixels
    descriptors: np.ndarray | None
    size: tuple[int, int]  # (width, height) of the original image


def features(image: np.ndarray) -> Features:
    """SIFT features of a BGR or gray image, keypoints in its own pixels."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, WORK_SIDE / max(h, w))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=2000).detectAndCompute(gray, None)
    points = np.float32([k.pt for k in keypoints]).reshape(-1, 2) / scale
    return Features(points, descriptors, (w, h))


# SIFT on a frame is ~50 ms (most of an alignment); ~0.3 MB each, so the
# cache holds about 300 MB. Frames of the same shot come up again and again.
@lru_cache(maxsize=1024)
def frame_features(path: Path) -> Features:
    return features(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))


def align(query: Features, frame: Features) -> tuple[int, list[float] | None]:
    """(agreeing points, CSS-style matrix [a, b, c, d, e, f] or None).

    The matrix maps query pixels to frame pixels: x' = a x + c y + e,
    y' = b x + d y + f, which is exactly CSS transform: matrix(a, b, c, d, e, f).
    """
    if query.descriptors is None or frame.descriptors is None or len(query.keypoints) < 4 or len(frame.keypoints) < 4:
        return 0, None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query.descriptors, frame.descriptors, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_POINTS:
        return len(good), None
    src = query.keypoints[[m.queryIdx for m in good]]
    dst = frame.keypoints[[m.trainIdx for m in good]]
    tolerance = 4.0 * max(frame.size) / WORK_SIDE  # ~4 px at matching resolution
    matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=tolerance)
    points = int(mask.sum()) if mask is not None else 0
    if matrix is None or points < MIN_POINTS:
        return points, None
    return points, [float(matrix[0, 0]), float(matrix[1, 0]), float(matrix[0, 1]),
                    float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])]


def rerank(candidates: list, alignments: list[tuple]) -> list[int]:
    """Order of candidates: confirmed matches (VERIFIED_POINTS or more) first,
    most agreeing points first; the rest keep their search order. Each
    alignment starts with its agreeing-point count."""
    confirmed = sorted((i for i, a in enumerate(alignments) if a[0] >= VERIFIED_POINTS),
                       key=lambda i: -alignments[i][0])
    return confirmed + [i for i in range(len(candidates)) if i not in confirmed]
