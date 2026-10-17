"""Helpers for inspecting intermediate results of the pipeline."""

from collections.abc import Sequence

import cv2
import numpy as np

from lane_finding.lane_fit import LaneFit

TITLE_COLOR = (0, 0, 255)  # BGR red
LEFT_PIXEL_COLOR = (0, 0, 255)  # BGR red
RIGHT_PIXEL_COLOR = (255, 0, 0)  # BGR blue
CURVE_COLOR = (0, 255, 0)  # BGR green
RESCUED_CURVE_COLOR = (0, 255, 255)  # BGR yellow: a curve derived from the other line
CURVE_THICKNESS = 4
TEXT_COLOR = (0, 255, 255)  # BGR yellow


def side_by_side(
    panels: Sequence[np.ndarray], titles: Sequence[str], scale: float = 1.0
) -> np.ndarray:
    """Place images in a row and label each one with a title.

    Args:
        panels: 8-bit images of the same height, grayscale or BGR.
        titles: One title per image.
        scale: Factor by which every image is resized before it is placed.

    Returns:
        BGR image containing all panels next to each other.

    Raises:
        ValueError: If the number of titles differs from the number of images or if
            the images do not have the same height.
    """
    if len(panels) != len(titles):
        raise ValueError(f"Got {len(panels)} images but {len(titles)} titles")
    if len({panel.shape[0] for panel in panels}) != 1:
        raise ValueError("All images must have the same height")

    labelled = []
    for panel, title in zip(panels, titles):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(panel, title, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, TITLE_COLOR, 3)
        labelled.append(panel)
    return np.hstack(labelled)


def curve_points(coefficients: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Sample a lane curve at every row of a bird's-eye image, for drawing it.

    Args:
        coefficients: ``(A, B, C)`` of the curve ``x = A*y**2 + B*y + C`` in pixels.
        size: ``(width, height)`` of the image. Columns far outside the image are clipped,
            so that a wild curve cannot break the drawing.

    Returns:
        Integer array of shape ``(height, 2)`` with the ``(column, row)`` of each point.
    """
    width, height = size
    rows = np.arange(height)
    columns = np.clip(np.polyval(coefficients, rows), -width, 2 * width)
    return np.column_stack([columns, rows]).round().astype(np.int32)


def draw_lane_fit(image: np.ndarray, fit: LaneFit) -> np.ndarray:
    """Draw the pixels and the curves of a lane fit on a copy of an image.

    The pixels of the left line are red, those of the right line blue. The curves are
    green; a curve that was derived from the other line (it has no pixels) is yellow.

    Args:
        image: Bird's-eye image to draw on, grayscale or BGR. A binary image shows the
            pixels that were not used as white.
        fit: Result of the lane line detection in an image of the same size.

    Returns:
        BGR copy of the image with the lane fit drawn on it.
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    height, width = canvas.shape[:2]
    for (rows, columns), color in (
        (fit.left_pixels, LEFT_PIXEL_COLOR),
        (fit.right_pixels, RIGHT_PIXEL_COLOR),
    ):
        canvas[rows, columns] = color

    for coefficients, pixels in ((fit.left, fit.left_pixels), (fit.right, fit.right_pixels)):
        if coefficients is None:
            continue
        points = curve_points(coefficients, (width, height))
        color = CURVE_COLOR if pixels[0].size else RESCUED_CURVE_COLOR
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, color, CURVE_THICKNESS)
    return canvas


def draw_text_lines(
    image: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int],
    *,
    scale: float = 1.0,
    thickness: int = 2,
) -> np.ndarray:
    """Write lines of text below each other on a copy of an image.

    Args:
        image: Image to write on, grayscale or BGR.
        lines: The lines of text.
        origin: ``(x, y)`` of the bottom-left corner of the first line.
        scale: Font scale; the line spacing follows it.
        thickness: Thickness of the strokes.

    Returns:
        BGR copy of the image with the text on it.
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    x, y = origin
    for index, line in enumerate(lines):
        position = (x, y + round(index * 40 * scale))
        cv2.putText(canvas, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, thickness)
    return canvas
