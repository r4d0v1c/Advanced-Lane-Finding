"""Selecting the pixels of lane markings in a road image.

A lane marking is a painted stripe 10-15 cm wide that is lighter (white paint) or yellower
(yellow paint) than the road surface next to it. Fixed colour thresholds would only work for
the light and the camera they were chosen with, so every pixel is compared with the road on
both of its sides instead:

* how far to the side the road is looked at follows from the camera geometry: the width of
  the lane in pixels is known at every image row, and a marking is a fixed part of it,
* a pixel belongs to a marking when it is lighter (lightness channel) or yellower (blue-yellow
  channel of the Lab colour space) than the road on both sides by clearly more than the
  noise of the image itself, which is measured in every image.

The inner pixels of areas wider than about two markings - sunlit asphalt, vehicles, grass -
have no darker road on both sides and are therefore not selected. Light stripes narrower
than the narrowest marking - seams, cracks, specks of noise - are dropped. Rows without road
(above the horizon, on the car hood) stay empty.
"""

import cv2
import numpy as np

from lane_finding.cameras import LANE_WIDTH_METERS

# Road standards: lane lines are 10-15 cm wide (US 4-6 inches, Europe 0.10-0.15 m).
MARKING_WIDTH_METERS = 0.15
# The road is looked at this many marking widths to each side, so that markings up to twice
# as wide (wide edge lines, or a marking that crosses a row at a slant in a bend) are found.
COMPARE_DISTANCE_MARKINGS = 2.0
# A marking has to stand out from the road by more than this many standard deviations of
# the road noise of the image (three-sigma rule).
NOISE_SIGMAS = 3.0
# The smallest noise is one grey level, the resolution of an 8-bit image.
MIN_NOISE = 1.0
# The noise is measured on the road ahead: in every row this many lane widths to each side
# of the image centre (the own lane and the neighbouring ones).
ROAD_HALF_WIDTH_LANES = 1.5
# For normally distributed noise the standard deviation is 1.4826 times the median absolute
# deviation; the difference of two pixels has sqrt(2) times the noise of one pixel.
_MEDIAN_DIFFERENCE_TO_SIGMA = 1.4826 / np.sqrt(2)


def _require_bgr(image: np.ndarray) -> None:
    """Check that an array is an 8-bit, 3-channel image.

    Args:
        image: Array to check.

    Raises:
        ValueError: If the array is not an 8-bit image with three channels.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Expected an 8-bit BGR image of shape (height, width, 3), got "
            f"{image.dtype} array of shape {image.shape}"
        )


def compare_distances(lane_widths: np.ndarray) -> np.ndarray:
    """Compute how far to each side the road is looked at in every image row.

    Args:
        lane_widths: Width of the lane in pixels at every row, 0 for rows without road.

    Returns:
        Integer array with the distance in pixels for every row; 0 where no marking can be
        resolved (no road, or the distance would be below one pixel).
    """
    scale = COMPARE_DISTANCE_MARKINGS * MARKING_WIDTH_METERS / LANE_WIDTH_METERS
    return np.rint(scale * np.asarray(lane_widths, dtype=np.float64)).astype(np.intp)


def stands_out(channel: np.ndarray, distances: np.ndarray, lane_widths: np.ndarray) -> np.ndarray:
    """Select the pixels that are higher than the channel on both of their sides.

    A pixel is selected when it exceeds the higher of the two pixels ``distance`` to its
    left and to its right (outside the image counts as the maximum) by more than
    ``NOISE_SIGMAS`` times the noise. The noise is the robust standard deviation of the
    differences between pixels ``distance`` apart on the road ahead.

    Args:
        channel: Single-channel 8-bit image.
        distances: Compare distance for every row (see :func:`compare_distances`).
        lane_widths: Width of the lane in pixels at every row, which marks the road ahead.

    Returns:
        Mask with 255 for the selected pixels and 0 elsewhere.
    """
    height, width = channel.shape
    contrast = np.zeros_like(channel)
    histogram = np.zeros(256, np.int64)
    center = width / 2
    for distance in np.unique(distances[distances > 0]):
        rows = np.flatnonzero(distances == distance)
        top, bottom = rows[0], rows[-1] + 1  # the distance grows with the row
        band = channel[top:bottom]
        left = np.full_like(band, 255)
        left[:, distance:] = band[:, :-distance]
        right = np.full_like(band, 255)
        right[:, :-distance] = band[:, distance:]
        contrast[top:bottom] = cv2.subtract(band, cv2.max(left, right))
        reach = ROAD_HALF_WIDTH_LANES * lane_widths[top:bottom].max()
        first = max(0, int(center - reach))
        last = min(width, int(center + reach))
        road = band[:, first:last]
        differences = cv2.absdiff(road[:, distance:], road[:, :-distance])
        histogram += np.bincount(differences.ravel(), minlength=256)
    total = histogram.sum()
    if total == 0:
        return np.zeros_like(channel)
    median = int(np.searchsorted(np.cumsum(histogram), total / 2))
    noise = max(_MEDIAN_DIFFERENCE_TO_SIGMA * median, MIN_NOISE)
    return cv2.compare(contrast, NOISE_SIGMAS * noise, cv2.CMP_GT)


def threshold_lane_pixels(image: np.ndarray, lane_widths: np.ndarray) -> np.ndarray:
    """Create the binary image of lane-marking pixels from a road image.

    White markings stand out in the lightness, yellow ones in the blue-yellow channel of
    the Lab colour space; both are combined.

    Args:
        image: 8-bit BGR road image, normally already undistorted.
        lane_widths: Width of the lane in pixels at every image row, 0 for the rows without
            road (see :meth:`lane_finding.cameras.CameraProfile.lane_widths`).

    Returns:
        Single-channel uint8 image of the same height and width with 255 for marking
        pixels and 0 elsewhere.

    Raises:
        ValueError: If the input is not an 8-bit BGR image or the number of lane widths
            differs from the image height.
    """
    _require_bgr(image)
    lane_widths = np.asarray(lane_widths, dtype=np.float64)
    if lane_widths.shape != (image.shape[0],):
        raise ValueError(
            f"Expected one lane width per image row ({image.shape[0]}), got {lane_widths.shape}"
        )
    distances = compare_distances(lane_widths)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    white = stands_out(lab[:, :, 0], distances, lane_widths)
    yellow = stands_out(lab[:, :, 2], distances, lane_widths)
    return cv2.bitwise_or(white, yellow)
