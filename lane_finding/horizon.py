"""Estimating the horizon of a video from its lane lines.

The horizon is the image row where the parallel lane lines meet (their vanishing point). It
moves with the pitch of the car and the slope of the road, so it is a little different in
every video. The bird's-eye view is accurate only if it is built with the right row: with
a wrong one the lane looks like a funnel and the measured radius is off.
"""

from collections.abc import Iterable

import cv2
import numpy as np

from lane_finding.cameras import CameraProfile

# The segments of the two lane lines lean to both sides: dx/dy of the left lines is between
# minus the maximum and minus the minimum, that of the right lines between the two.
MIN_SEGMENT_SLOPE = 0.5
MAX_SEGMENT_SLOPE = 2.6
# Hough transform: votes needed for a line, the shortest segment and the largest gap in a
# segment (the last two as fractions of the image width).
HOUGH_VOTES = 30
MIN_SEGMENT_LENGTH_FRACTION = 0.06
MAX_SEGMENT_GAP_FRACTION = 0.03
# Only the longest segments of each side are used.
MAX_SEGMENTS_PER_SIDE = 40
# A horizon further from the usual one of the camera than this fraction of the image height is
# not believed.
MAX_HORIZON_DEVIATION_FRACTION = 0.1
# Frames with a usable estimate needed for the estimate of a video.
MIN_FRAMES = 5
# While a video is processed, every this many frames is used (about 4 per second at 25 fps).
SAMPLE_EVERY_FRAMES = 6


def vanishing_point_row(binary: np.ndarray, camera: CameraProfile) -> float | None:
    """Estimate the row of the vanishing point of the lane lines in one frame.

    The row is the median of the intersections of every left-leaning segment with every
    right-leaning segment of the binary image, so a few stray segments do not matter.

    Args:
        binary: Binary lane-pixel image of a frame in the camera view (0 and 255).
        camera: Profile of the camera that took the frame.

    Returns:
        The row, or ``None`` if the image shows no segment on one of the two sides.
    """
    width = binary.shape[1]
    road = binary.copy()
    road[: int(camera.horizon_y)] = 0  # no road above the horizon
    road[int(camera.bottom_y) :] = 0  # the car hood
    segments = cv2.HoughLinesP(
        road,
        1,
        np.pi / 180,
        HOUGH_VOTES,
        minLineLength=int(MIN_SEGMENT_LENGTH_FRACTION * width),
        maxLineGap=int(MAX_SEGMENT_GAP_FRACTION * width),
    )
    if segments is None:
        return None
    x1, y1, x2, y2 = segments.reshape(-1, 4).T.astype(np.float64)
    sloped = y1 != y2
    x1, y1, x2, y2 = x1[sloped], y1[sloped], x2[sloped], y2[sloped]
    slope = (x2 - x1) / (y2 - y1)
    length = np.hypot(x2 - x1, y2 - y1)
    sides = []
    for side in (
        (slope > -MAX_SEGMENT_SLOPE) & (slope < -MIN_SEGMENT_SLOPE),
        (slope > MIN_SEGMENT_SLOPE) & (slope < MAX_SEGMENT_SLOPE),
    ):
        longest = np.flatnonzero(side)[np.argsort(-length[side])][:MAX_SEGMENTS_PER_SIDE]
        if longest.size == 0:
            return None
        sides.append((x1[longest], y1[longest], slope[longest]))
    (xa, ya, sa), (xb, yb, sb) = sides
    # The segments are the lines x = x1 + slope * (y - y1); every pair intersects in a row.
    rows = (xb[None, :] - xa[:, None] + sa[:, None] * ya[:, None] - sb[None, :] * yb[None, :]) / (
        sa[:, None] - sb[None, :]
    )
    return float(np.median(rows))


class HorizonEstimator:
    """Estimates the horizon of a video while it is processed, from the frames seen so far.

    Every ``sample_every``-th frame gives a vanishing point row; rows too far from the usual
    horizon of the camera are not believed. The estimate is the median of all believed rows
    so far, available once ``MIN_FRAMES`` of them were collected.
    """

    def __init__(self, sample_every: int = SAMPLE_EVERY_FRAMES) -> None:
        """Create an estimator that has not seen any frame.

        Args:
            sample_every: Distance in frames between the frames that are used.
        """
        self.sample_every = sample_every
        self._frames = 0
        self._rows: list[float] = []

    def update(self, binary: np.ndarray, camera: CameraProfile) -> float | None:
        """Take the next frame of the video into account.

        Args:
            binary: Binary lane-pixel image of the frame in the camera view (0 and 255).
            camera: Profile of the camera that took the video.

        Returns:
            The horizon row estimated from the frames so far, or ``None`` while fewer than
            ``MIN_FRAMES`` frames gave a believable row.
        """
        if self._frames % self.sample_every == 0:
            row = vanishing_point_row(binary, camera)
            limit = MAX_HORIZON_DEVIATION_FRACTION * camera.size[1]
            if row is not None and abs(row - camera.horizon_y) <= limit:
                self._rows.append(row)
        self._frames += 1
        return float(np.median(self._rows)) if len(self._rows) >= MIN_FRAMES else None


def estimate_horizon(binaries: Iterable[np.ndarray], camera: CameraProfile) -> float:
    """Estimate the horizon row from the binary lane-pixel images of several frames.

    Args:
        binaries: Binary lane-pixel images of frames in the camera view (0 and 255).
        camera: Profile of the camera that took them.

    Returns:
        The median of the vanishing point rows of the frames. If fewer than ``MIN_FRAMES``
        frames give a believable row, the usual horizon of the camera.
    """
    estimator = HorizonEstimator(sample_every=1)
    estimate = None
    for binary in binaries:
        estimate = estimator.update(binary, camera)
    return camera.horizon_y if estimate is None else estimate
