"""Synthetic bird's-eye binary scenes with known lane lines for the tests."""

from collections.abc import Sequence

import numpy as np


def curve(size: tuple[int, int], bottom_fraction: float, slope: float, shift: float) -> np.ndarray:
    """Create the coefficients of a lane line ``x = A*y**2 + B*y + C``.

    Args:
        size: ``(width, height)`` of the image.
        bottom_fraction: Column of the line in the bottom row as a fraction of the width.
        slope: Pixels the line moves sideways per row at the bottom row.
        shift: Sideways shift caused by the curvature over the image height, as a fraction
            of the width (so that the same value describes the same bend at any size).

    Returns:
        The coefficients ``(A, B, C)``.
    """
    width, height = size
    k = shift * width / height**2
    bottom = bottom_fraction * width
    return np.array([k, slope - 2 * k * height, bottom - slope * height + k * height**2])


def lane(size: tuple[int, int], slope: float = 0.0, shift: float = 0.0):
    """Create the two parallel lines of a lane in the usual place of the bird's-eye view.

    Args:
        size: ``(width, height)`` of the image.
        slope: Slope of the lines at the bottom row (pixels per row).
        shift: Sideways shift of the lines over the image height (fraction of the width).

    Returns:
        The coefficients of the left and the right line.
    """
    return curve(size, 0.25, slope, shift), curve(size, 0.75, slope, shift)


def draw_lane(
    size: tuple[int, int],
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    right_dashed: bool = False,
    left_hidden: Sequence[tuple[int, int]] = (),
    right_hidden: Sequence[tuple[int, int]] = (),
    blocks: Sequence[tuple[int, int, int, int]] = (),
) -> np.ndarray:
    """Draw lane lines (and noise) into a bird's-eye binary image.

    Args:
        size: ``(width, height)`` of the image.
        left: Coefficients of the left line, or ``None`` to leave it out.
        right: Coefficients of the right line, or ``None`` to leave it out.
        right_dashed: Draw the right line as dashes (period and dash length scale with
            the image height, like a real dashed line seen from above).
        left_hidden: Row ranges ``(first, last)`` where the left line is not drawn.
        right_hidden: Row ranges where the right line is not drawn.
        blocks: Noise rectangles ``(row, column, height, width)`` that are filled.

    Returns:
        Binary image with the values 0 and 255.
    """
    width, height = size
    image = np.zeros((height, width), np.uint8)
    rows = np.arange(height)
    half = round(0.01 * width)
    for coefficients, dashed, hidden in (
        (left, False, left_hidden),
        (right, right_dashed, right_hidden),
    ):
        if coefficients is None:
            continue
        visible = np.ones(height, bool)
        if dashed:
            period, dash = 0.48 * height, 0.14 * height
            visible = ((height - rows + 0.2 * period) % period) < dash
        for first, last in hidden:
            visible[first:last] = False
        columns = np.polyval(coefficients, rows).astype(int)
        for row in rows[visible]:
            image[row, max(columns[row] - half, 0) : max(columns[row] + half, 0)] = 255
    for row, column, block_height, block_width in blocks:
        image[row : row + block_height, column : column + block_width] = 255
    return image


def max_error(estimate: np.ndarray, truth: np.ndarray, size: tuple[int, int]) -> float:
    """Compute the largest column difference between two curves over most of the image.

    Args:
        estimate: Coefficients of the estimated curve.
        truth: Coefficients of the true curve.
        size: ``(width, height)`` of the image.

    Returns:
        The largest absolute difference in pixels.
    """
    rows = np.linspace(0.1 * size[1], 0.95 * size[1], 9)
    return float(np.max(np.abs(np.polyval(estimate, rows) - np.polyval(truth, rows))))
