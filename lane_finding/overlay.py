"""Drawing the detected lane and its measurements onto the road image.

The lane is drawn in the bird's-eye view, where it is simple (two polynomials), and warped
back onto the camera image.
"""

import cv2
import numpy as np

from lane_finding.lane_fit import LaneFit
from lane_finding.measurements import LaneMeasurements
from lane_finding.perspective import PerspectiveTransform
from lane_finding.tracking import TrackStatus
from lane_finding.visualization import curve_points, draw_text_lines

AREA_COLOR = (0, 200, 0)  # BGR green
AREA_OPACITY = 0.4
LEFT_LINE_COLOR = (0, 0, 255)  # BGR red
RIGHT_LINE_COLOR = (255, 0, 0)  # BGR blue
LINE_OPACITY = 0.9
# Width of the drawn lane lines in the bird's-eye view as a fraction of the image width.
LINE_WIDTH_FRACTION = 0.012
NO_LANE_TEXT = "No plausible lane"
HELD_TEXT = "Lane lines not visible: last lane held"

# Values of the label image the lane is drawn into (0 is the background).
AREA_LABEL = 1
LEFT_LABEL = 2
RIGHT_LABEL = 3


def _lane_labels(fit: LaneFit, size: tuple[int, int]) -> np.ndarray:
    """Draw the lane area and its two boundary lines into a label image.

    Args:
        fit: The lane lines in the bird's-eye view; both must have been found.
        size: ``(width, height)`` of the bird's-eye image.

    Returns:
        uint8 image of that size with ``AREA_LABEL`` between the lines, ``LEFT_LABEL`` and
        ``RIGHT_LABEL`` on the lines and 0 elsewhere.
    """
    width, height = size
    left = curve_points(fit.left, size)
    right = curve_points(fit.right, size)
    thickness = max(1, round(LINE_WIDTH_FRACTION * width))
    labels = np.zeros((height, width), np.uint8)
    cv2.fillPoly(labels, [np.vstack([left, right[::-1]])], AREA_LABEL)
    cv2.polylines(labels, [left.reshape(-1, 1, 2)], False, LEFT_LABEL, thickness)
    cv2.polylines(labels, [right.reshape(-1, 1, 2)], False, RIGHT_LABEL, thickness)
    return labels


def _blend(
    canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], opacity: float
) -> None:
    """Mix a color into the pixels of an image under a mask, in place.

    The mixed value of every channel depends only on the old value, so it is looked up in a
    table of 256 entries; this is much faster than doing the arithmetic on the masked pixels.

    Args:
        canvas: 8-bit BGR image to draw on.
        mask: Boolean mask of the pixels to change.
        color: BGR color to mix in.
        opacity: Share of the color in the result (0 to 1).
    """
    values = np.arange(256, dtype=np.float32)[:, None]
    table = np.rint(values * (1 - opacity) + np.array(color, np.float32) * opacity)
    mixed = cv2.LUT(canvas, table.astype(np.uint8).reshape(1, 256, 3))
    cv2.copyTo(mixed, mask.view(np.uint8), canvas)


def draw_lane_overlay(
    image: np.ndarray,
    transform: PerspectiveTransform,
    fit: LaneFit,
    measurements: LaneMeasurements | None,
    status: TrackStatus = TrackStatus.DETECTED,
) -> np.ndarray:
    """Draw the lane and the measurements on a copy of a road image.

    The lane area is green and its boundaries are red (left) and blue (right). The text at
    the top gives the radius of curvature, the position of the vehicle and the lane width.
    A lane that is held from earlier frames of a video is drawn as well, with a note.
    Without measurements (no plausible lane) only a note is written.

    Args:
        image: 8-bit BGR camera image the lane was found in.
        transform: Perspective transform between the camera view and the bird's-eye view.
        fit: The lane lines in the bird's-eye view.
        measurements: Measurements of the lane made from the fit, or ``None`` if there is no
            plausible lane (then no lane is drawn).
        status: Whether the lane was found in this image or is held from earlier frames.

    Returns:
        BGR copy of the image with the lane drawn on it.
    """
    height, width = image.shape[:2]
    canvas = image.copy()
    if measurements is not None:
        labels = transform.unwarp(_lane_labels(fit, (width, height)), cv2.INTER_NEAREST)
        _blend(canvas, labels == AREA_LABEL, AREA_COLOR, AREA_OPACITY)
        _blend(canvas, labels == LEFT_LABEL, LEFT_LINE_COLOR, LINE_OPACITY)
        _blend(canvas, labels == RIGHT_LABEL, RIGHT_LINE_COLOR, LINE_OPACITY)

    lines = measurements.text_lines() if measurements is not None else [NO_LANE_TEXT]
    if status is TrackStatus.HELD:
        lines.append(HELD_TEXT)
    return draw_text_lines(
        canvas, lines, (30, round(0.09 * height)), scale=1.2 * width / 1280, thickness=2
    )
