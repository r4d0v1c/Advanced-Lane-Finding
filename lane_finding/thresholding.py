"""Color thresholding that turns a road image into a binary lane-pixel mask.

Lane lines are either yellow or white, so two color masks are combined:

* yellow: a hue range with enough saturation and brightness (HSV color space),
* white: high lightness (HLS color space).

Regions that are far wider than any lane line, such as sunlit asphalt, vehicles or
dry grass, are removed from both masks.
"""

import cv2
import numpy as np

# Yellow lane lines in OpenCV HSV space (H: 0-179, S: 0-255, V: 0-255). The minimum
# saturation rejects gray asphalt and concrete, the minimum value still accepts
# lines in shadow.
YELLOW_LOWER = (15, 80, 80)
YELLOW_UPPER = (35, 255, 255)

# White lane lines: minimum HLS lightness (0-255). Lower values start to select
# bright concrete, higher values lose faint dashes.
WHITE_MIN_LIGHTNESS = 200

# Runs at least this wide (as a fraction of the image width) cannot belong to a lane
# line and are removed. In the test material a line crosses a pixel row with at most
# about 3 % of the width, even in sharp curves.
MAX_LINE_RUN_FRACTION = 0.08


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


def yellow_mask(image: np.ndarray) -> np.ndarray:
    """Select the yellow pixels of an image.

    Args:
        image: 8-bit BGR image.

    Returns:
        Mask of the same height and width as the image with 255 for yellow pixels
        and 0 elsewhere.

    Raises:
        ValueError: If the input is not an 8-bit BGR image.
    """
    _require_bgr(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)


def white_mask(image: np.ndarray) -> np.ndarray:
    """Select the bright (white) pixels of an image.

    Args:
        image: 8-bit BGR image.

    Returns:
        Mask of the same height and width as the image with 255 for pixels whose
        HLS lightness is at least ``WHITE_MIN_LIGHTNESS`` and 0 elsewhere.

    Raises:
        ValueError: If the input is not an 8-bit BGR image.
    """
    _require_bgr(image)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return cv2.inRange(hls, (0, WHITE_MIN_LIGHTNESS, 0), (179, 255, 255))


def remove_wide_runs(mask: np.ndarray, max_width: int) -> np.ndarray:
    """Remove the pixels that lie on a horizontal run of ``max_width`` or more.

    This is a morphological opening with a one-pixel-high kernel: the erosion marks
    the first pixel of every run of at least ``max_width`` pixels and the dilation
    spreads each mark back over the run. The kernel anchors are set explicitly
    (a centered anchor is not exact for even widths) and pixels outside the image
    count as 0, so a narrow run at the image border is not mistaken for a wide one.

    Args:
        mask: Binary mask with values 0 and 255.
        max_width: Smallest run length in pixels that is removed.

    Returns:
        Copy of the mask without the wide runs.
    """
    kernel = np.ones((1, max_width), np.uint8)
    border = {"borderType": cv2.BORDER_CONSTANT, "borderValue": 0}
    run_starts = cv2.erode(mask, kernel, anchor=(0, 0), **border)
    wide_runs = cv2.dilate(run_starts, kernel, anchor=(max_width - 1, 0), **border)
    return cv2.subtract(mask, wide_runs)


def threshold_lane_pixels(image: np.ndarray) -> np.ndarray:
    """Create the binary image of lane-line pixels from a road image.

    Combines the yellow and the white mask after removing the regions that are too
    wide to be a lane line. Works for any resolution because all limits are either
    color values or fractions of the image width.

    Args:
        image: 8-bit BGR road image, normally already undistorted.

    Returns:
        Single-channel uint8 image of the same height and width with 255 for lane
        pixels and 0 elsewhere.

    Raises:
        ValueError: If the input is not an 8-bit BGR image.
    """
    _require_bgr(image)
    max_width = max(2, round(MAX_LINE_RUN_FRACTION * image.shape[1]))
    yellow = remove_wide_runs(yellow_mask(image), max_width)
    white = remove_wide_runs(white_mask(image), max_width)
    return cv2.bitwise_or(yellow, white)
