"""Lane line detection in the bird's-eye binary image and polynomial fitting.

Each lane line is described by a second-degree polynomial ``x = A*y**2 + B*y + C`` where
``y`` is the image row and ``x`` the column. The lines are found in four steps:

1. A column histogram of the lower half of the image shows where each line starts. It is
   searched only in the two corridors around the columns the lane is mapped to.
2. Sliding windows follow each line upwards and collect its pixels. A window that sees too
   few pixels (a gap between dashes) keeps moving in the direction the lines were going.
3. A robust fit (Tukey bisquare weights) ignores noise pixels that were collected as well.
4. How the lines are fitted depends on how well each of them is supported by pixels: two
   strong lines are fitted independently, a weak line is fitted together with the other one
   (shared curvature) and a missing line is derived from the other one by the lane width.

"""

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from lane_finding.perspective import LANE_MARGIN_FRACTION, LANE_WIDTH_FRACTION

# A set of pixels: the rows and the columns of the selected pixels.
Pixels = tuple[np.ndarray, np.ndarray]

# Search for the line pixels.
NUM_WINDOWS = 9
# Half width of a window as a fraction of the image width. Wider windows collect more noise.
WINDOW_MARGIN_FRACTION = 0.08
# A window with at least this many pixels (fraction of the image area) is centered on them.
MIN_WINDOW_PIXELS_FRACTION = 0.00005
# The histogram is smoothed with a box as wide as this fraction of the image width.
HISTOGRAM_SMOOTHING_FRACTION = 0.02
# The line starts are searched this far (fraction of the width) from the lane columns.
BASE_SEARCH_HALF_WIDTH = 0.18
# Half width of the band around the previous curves in which a line is searched in a video
# (fraction of the width). Lane lines are about 3 % wide and move little between frames.
SEARCH_MARGIN_FRACTION = 0.06

# How well a line must be supported by pixels. A line is present with at least this many
# pixels (fraction of the image area) spread over this fraction of the image height.
MIN_LINE_PIXELS_FRACTION = 0.0004
MIN_LINE_EXTENT_FRACTION = 0.25
# A strong line has more pixels, spans half of the height and touches at least this many of
# the height bins. Clean dashed lines are strong, faint or partly hidden lines are weak.
STRONG_LINE_PIXELS_FRACTION = 0.0025
STRONG_LINE_EXTENT_FRACTION = 0.5
HEIGHT_BINS = 12
STRONG_LINE_MIN_BINS = 5

# Robust fit.
TUKEY_CONSTANT = 4.685
ROBUST_ITERATIONS = 5
# Smallest residual scale in pixels, so that a very clean line does not reject its own pixels.
MIN_RESIDUAL_SCALE = 2.0
MIN_FIT_ROWS = 3
MIN_ROBUST_WEIGHT = 10.0

# Plausibility of a lane. The lane width is checked at several rows, the curve may not shift
# sideways by more than about one image width over the image height (a curve of about 60 m
# radius), and the curves have to stay near the image.
LANE_WIDTH_RANGE = (0.6, 1.5)
MAX_LANE_WIDTH_RATIO = 1.5
MAX_CURVATURE_SHIFT = 1.05
CHECK_ROW_FRACTIONS = np.linspace(0.05, 0.95, 10)
VISIBLE_COLUMN_RANGE = (-0.25, 1.25)

# The pixels of a line that was not found or was derived from the other one.
NO_PIXELS: Pixels = (np.empty(0, np.intp), np.empty(0, np.intp))


class FitMode(Enum):
    """How the polynomials of a lane fit were obtained."""

    INDEPENDENT = "independent"
    SHARED_CURVATURE = "shared curvature"
    RESCUED = "rescued"
    # Not found in this frame: the lane of the previous frames (only in a video).
    TRACKED = "tracked"
    NONE = "none"


@dataclass(frozen=True, eq=False)
class LaneFit:
    """Result of the lane line detection.

    Attributes:
        left: Coefficients ``(A, B, C)`` of ``x = A*y**2 + B*y + C`` of the left line, or
            ``None`` when it was not found.
        right: The same for the right line.
        left_pixels: Rows and columns of the pixels the left polynomial was fitted to.
            Empty when the line was derived from the other one.
        right_pixels: The same for the right line.
        mode: How the polynomials were obtained.
    """

    left: np.ndarray | None
    right: np.ndarray | None
    left_pixels: Pixels
    right_pixels: Pixels
    mode: FitMode

    def is_plausible(self, size: tuple[int, int]) -> bool:
        """Check that the two curves can be the boundaries of a lane.

        Args:
            size: ``(width, height)`` of the image the lane was fitted in.

        Returns:
            ``True`` if both lines exist, the lane width is about the expected one at
            every checked row, the curvature is realistic and the curves stay near the
            image.
        """
        if self.left is None or self.right is None:
            return False
        width, height = size
        shift_scale = height**2 / width
        if max(abs(self.left[0]), abs(self.right[0])) * shift_scale > MAX_CURVATURE_SHIFT:
            return False
        rows = CHECK_ROW_FRACTIONS * height
        left_columns = np.polyval(self.left, rows)
        right_columns = np.polyval(self.right, rows)
        lane_width = right_columns - left_columns
        expected = LANE_WIDTH_FRACTION * width
        return bool(
            np.all(lane_width > LANE_WIDTH_RANGE[0] * expected)
            and np.all(lane_width < LANE_WIDTH_RANGE[1] * expected)
            and lane_width.max() / lane_width.min() < MAX_LANE_WIDTH_RATIO
            and np.all(left_columns > VISIBLE_COLUMN_RANGE[0] * width)
            and np.all(right_columns < VISIBLE_COLUMN_RANGE[1] * width)
        )


def _require_image(binary: np.ndarray) -> None:
    """Check that an array is a single-channel image.

    Args:
        binary: Array to check.

    Raises:
        ValueError: If the array does not have two dimensions.
    """
    if binary.ndim != 2:
        raise ValueError(f"Expected a single-channel image, got an array of shape {binary.shape}")


def find_base_positions(binary: np.ndarray) -> tuple[int, int]:
    """Find the columns where the two lane lines start at the bottom of the image.

    Args:
        binary: Bird's-eye binary image (lane pixels are non-zero).

    Returns:
        Columns of the left and the right line. A corridor without any pixels gives the
        column the lane is nominally mapped to.

    Raises:
        ValueError: If the input is not a single-channel image.
    """
    _require_image(binary)
    height, width = binary.shape
    histogram = np.count_nonzero(binary[height // 2 :], axis=0).astype(np.float32)
    kernel = max(3, int(HISTOGRAM_SMOOTHING_FRACTION * width) | 1)
    histogram = cv2.blur(histogram.reshape(1, -1), (kernel, 1)).ravel()
    bases = []
    for nominal in (LANE_MARGIN_FRACTION, 1 - LANE_MARGIN_FRACTION):
        first = int((nominal - BASE_SEARCH_HALF_WIDTH) * width)
        last = int((nominal + BASE_SEARCH_HALF_WIDTH) * width)
        peak = first + int(np.argmax(histogram[first:last]))
        bases.append(peak if histogram[peak] > 0 else int(nominal * width))
    return bases[0], bases[1]


@dataclass
class _Track:
    """State of the windows that follow one line.

    Attributes:
        center: Column the next window is centered on.
        shift: Sideways drift of the line per window, ``None`` until two windows have
            seen the line.
        shift_age: Number of windows since the shift was updated.
        measured: Mean column of the last window that saw enough pixels.
        gap: Number of windows since that measurement.
        rows: Rows of the pixels collected so far, one array per window.
        columns: Columns of the pixels collected so far, one array per window.
    """

    center: float
    shift: float | None = None
    shift_age: int = 0
    measured: float | None = None
    gap: int = 1
    rows: list[np.ndarray] = field(default_factory=list)
    columns: list[np.ndarray] = field(default_factory=list)


def _gap_shift(track: _Track, other: _Track) -> float:
    """Choose the drift that carries a window over a gap in a line.

    The two lines of a lane run alike, so the more recently measured drift of the two
    lines is used: when one line is solid its drift is known in every window while the
    drift of the dashed line is out of date, which matters in a bend.

    Args:
        track: The line whose window found too few pixels.
        other: The other line.

    Returns:
        The sideways drift per window; 0 if neither line has shown any drift yet.
    """
    candidates = [
        (candidate.shift_age, order, candidate.shift)
        for order, candidate in enumerate((track, other))
        if candidate.shift is not None
    ]
    return min(candidates)[2] if candidates else 0.0


def search_windows(
    binary: np.ndarray,
    bases: tuple[int, int],
    *,
    num_windows: int = NUM_WINDOWS,
    margin_fraction: float = WINDOW_MARGIN_FRACTION,
    min_pixels_fraction: float = MIN_WINDOW_PIXELS_FRACTION,
) -> tuple[Pixels, Pixels]:
    """Collect the pixels of the two lane lines with sliding windows.

    The windows are stacked from the bottom of the image upwards. A window with enough
    pixels is centered on their mean column. A window without enough pixels (a gap between
    dashes) moves on with the drift the lines showed between their last measurements.

    Args:
        binary: Bird's-eye binary image (lane pixels are non-zero).
        bases: Columns where the left and the right line start.
        num_windows: Number of windows per line.
        margin_fraction: Half width of a window as a fraction of the image width.
        min_pixels_fraction: Pixels (fraction of the image area) needed to center a window.

    Returns:
        The rows and columns of the pixels collected for the left and the right line.

    Raises:
        ValueError: If the input is not a single-channel image.
    """
    _require_image(binary)
    height, width = binary.shape
    margin = int(margin_fraction * width)
    min_pixels = max(10, int(min_pixels_fraction * width * height))
    rows, columns = np.nonzero(binary)  # the pixels come row by row, so the rows are sorted
    window_height = height // num_windows
    bounds = [
        (
            np.searchsorted(rows, height - (index + 1) * window_height),
            np.searchsorted(rows, height - index * window_height),
        )
        for index in range(num_windows)
    ]

    tracks = [_Track(center=float(base)) for base in bases]
    for first, last in bounds:
        window_rows, window_columns = rows[first:last], columns[first:last]
        seen = []
        for track in tracks:
            track.shift_age += 1
            inside = np.abs(window_columns - track.center) < margin
            count = np.count_nonzero(inside)
            if count:
                track.rows.append(window_rows[inside])
                track.columns.append(window_columns[inside])
            seen.append(count >= min_pixels)
            if seen[-1]:
                mean = float(window_columns[inside].mean())
                if track.measured is not None:
                    drift = (mean - track.measured) / track.gap
                    track.shift = drift if track.shift is None else 0.5 * (track.shift + drift)
                    track.shift_age = 0
                track.measured, track.gap, track.center = mean, 1, mean
        for index, track in enumerate(tracks):
            if not seen[index]:
                track.gap += 1
                track.center += _gap_shift(track, tracks[1 - index])
            track.center = min(max(track.center, margin), width - margin)

    lines = [
        (np.concatenate(track.rows), np.concatenate(track.columns)) if track.rows else NO_PIXELS
        for track in tracks
    ]
    return lines[0], lines[1]


def _row_means(
    rows: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Summarize the pixels of a line by the mean column of every occupied row.

    Fitting these means weighted by the pixel counts gives the same least-squares result as
    fitting every pixel, with far fewer points.

    Args:
        rows: Rows of the pixels.
        columns: Columns of the pixels.

    Returns:
        The occupied rows, the mean column of each and the number of pixels in each, or
        ``None`` if fewer than ``MIN_FIT_ROWS`` rows are occupied.
    """
    counts = np.bincount(rows)
    occupied = np.flatnonzero(counts)
    if occupied.size < MIN_FIT_ROWS:
        return None
    sums = np.bincount(rows, weights=columns, minlength=counts.size)
    weights = counts[occupied].astype(np.float64)
    return occupied.astype(np.float64), sums[occupied] / weights, weights


def _weighted_least_squares(
    design: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Solve a weighted least-squares problem.

    Args:
        design: Design matrix with one row per observation.
        targets: Observed values.
        weights: Weight of every observation.

    Returns:
        The parameter vector.
    """
    root = np.sqrt(weights)[:, None]
    return np.linalg.lstsq(design * root, targets * root[:, 0], rcond=None)[0]


def _robust_least_squares(
    design: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Solve a weighted least-squares problem that ignores outliers (Tukey bisquare).

    Starting from the ordinary solution, every observation is weighted down according to
    its residual, and observations far from the fit get no weight at all.

    Args:
        design: Design matrix with one row per observation.
        targets: Observed values.
        weights: Weight of every observation (here the number of pixels it stands for).

    Returns:
        The parameter vector.
    """
    solution = _weighted_least_squares(design, targets, weights)
    for _ in range(ROBUST_ITERATIONS):
        residuals = targets - design @ solution
        spread = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        scaled = residuals / (TUKEY_CONSTANT * max(spread, MIN_RESIDUAL_SCALE))
        robust_weights = weights * np.where(np.abs(scaled) < 1, (1 - scaled**2) ** 2, 0.0)
        too_few = np.count_nonzero(robust_weights) < len(solution)
        if robust_weights.sum() < MIN_ROBUST_WEIGHT or too_few:
            break
        solution = _weighted_least_squares(design, targets, robust_weights)
    return solution


def fit_polynomial(rows: np.ndarray, columns: np.ndarray) -> np.ndarray | None:
    """Fit ``x = A*y**2 + B*y + C`` robustly to the pixels of one line.

    Args:
        rows: Rows of the pixels.
        columns: Columns of the pixels.

    Returns:
        The coefficients ``(A, B, C)``, or ``None`` if the pixels occupy fewer than
        ``MIN_FIT_ROWS`` rows.
    """
    data = _row_means(rows, columns)
    if data is None:
        return None
    y, x, weights = data
    design = np.column_stack([y * y, y, np.ones_like(y)])
    return _robust_least_squares(design, x, weights)


def fit_shared_curvature(
    left_pixels: Pixels, right_pixels: Pixels
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit both lines robustly with one common curvature ``A``.

    The lines of a lane are parallel, so they bend alike. Fitting them together keeps the
    curvature of a line that has only a few pixels close to the one of the other line.

    Args:
        left_pixels: Rows and columns of the left line.
        right_pixels: Rows and columns of the right line.

    Returns:
        The coefficients ``(A, B, C)`` of the left and of the right line, or ``None`` if
        one of the lines occupies fewer than ``MIN_FIT_ROWS`` rows.
    """
    left = _row_means(*left_pixels)
    right = _row_means(*right_pixels)
    if left is None or right is None:
        return None
    (left_y, left_x, left_weights), (right_y, right_x, right_weights) = left, right
    y = np.concatenate([left_y, right_y])
    design = np.zeros((y.size, 5))
    design[:, 0] = y * y
    design[: left_y.size, 1] = left_y
    design[: left_y.size, 2] = 1
    design[left_y.size :, 3] = right_y
    design[left_y.size :, 4] = 1
    a, left_b, left_c, right_b, right_c = _robust_least_squares(
        design,
        np.concatenate([left_x, right_x]),
        np.concatenate([left_weights, right_weights]),
    )
    return np.array([a, left_b, left_c]), np.array([a, right_b, right_c])


def _is_line_present(pixels: Pixels, size: tuple[int, int]) -> bool:
    """Check that a pixel set is large and tall enough to be a lane line.

    Args:
        pixels: Rows and columns of the candidate line.
        size: ``(width, height)`` of the image.

    Returns:
        ``True`` if the line has enough pixels spread over enough of the image height.
    """
    rows = pixels[0]
    width, height = size
    return bool(
        rows.size >= MIN_LINE_PIXELS_FRACTION * width * height
        and np.ptp(rows) >= MIN_LINE_EXTENT_FRACTION * height
    )


def _is_line_strong(pixels: Pixels, size: tuple[int, int]) -> bool:
    """Check that a line is well supported along most of its length.

    Args:
        pixels: Rows and columns of the line.
        size: ``(width, height)`` of the image.

    Returns:
        ``True`` if the line has many pixels, spans half of the image height and touches
        several height bins, so that its polynomial is well determined on its own.
    """
    rows = pixels[0]
    width, height = size
    if rows.size < STRONG_LINE_PIXELS_FRACTION * width * height:
        return False
    if np.ptp(rows) < STRONG_LINE_EXTENT_FRACTION * height:
        return False
    bins = rows * HEIGHT_BINS // height
    return int(np.count_nonzero(np.bincount(bins, minlength=HEIGHT_BINS))) >= STRONG_LINE_MIN_BINS


def search_around(
    binary: np.ndarray,
    curves: tuple[np.ndarray, np.ndarray],
    margin_fraction: float = SEARCH_MARGIN_FRACTION,
) -> tuple[Pixels, Pixels]:
    """Collect the pixels of the two lane lines near the curves of a previous frame.

    In a video the lines move only a little from one frame to the next, so it is enough to
    take the pixels in a band around the previous curves. This is faster than the windows,
    it needs no histogram and a gap in a dashed line does not matter because the band
    follows the curve through it.

    Args:
        binary: Bird's-eye binary image (lane pixels are non-zero).
        curves: Coefficients ``(A, B, C)`` of the previous left and right curve.
        margin_fraction: Half width of the band as a fraction of the image width.

    Returns:
        The rows and columns of the pixels collected for the left and the right line.

    Raises:
        ValueError: If the input is not a single-channel image.
    """
    _require_image(binary)
    margin = margin_fraction * binary.shape[1]
    rows, columns = np.nonzero(binary)
    lines = []
    for coefficients in curves:
        inside = np.abs(columns - np.polyval(coefficients, rows)) < margin
        lines.append((rows[inside], columns[inside]))
    return lines[0], lines[1]


def fit_from_pixels(
    left_pixels: Pixels,
    right_pixels: Pixels,
    size: tuple[int, int],
    lane_width: np.ndarray | None = None,
) -> LaneFit:
    """Fit the two lane lines to the pixels collected for them.

    Two strong lines are fitted independently. If a line is only weakly supported, both
    are fitted with a common curvature. If only one line is present, the other one is
    derived from it by the lane width.

    Args:
        left_pixels: Rows and columns collected for the left line.
        right_pixels: Rows and columns collected for the right line.
        size: ``(width, height)`` of the image the pixels come from.
        lane_width: Coefficients of the lane width in pixels, that is of the right curve
            minus the left curve, used to derive a missing line. By default the lane has
            the constant width the bird's-eye view maps a lane to.

    Returns:
        The fitted lines and how they were obtained. Use :meth:`LaneFit.is_plausible` to
        check the result; it may not describe a lane when the pixels show none.
    """
    width = size[0]
    left_present = _is_line_present(left_pixels, size)
    right_present = _is_line_present(right_pixels, size)

    left = right = None
    mode = FitMode.INDEPENDENT
    if (
        left_present
        and right_present
        and not (_is_line_strong(left_pixels, size) and _is_line_strong(right_pixels, size))
    ):
        mode = FitMode.SHARED_CURVATURE
        shared = fit_shared_curvature(left_pixels, right_pixels)
        if shared is not None:
            left, right = shared
    else:
        if left_present:
            left = fit_polynomial(*left_pixels)
        if right_present:
            right = fit_polynomial(*right_pixels)

    if lane_width is None:
        lane_width = np.array([0.0, 0.0, LANE_WIDTH_FRACTION * width])
    if left is not None and right is None:
        right = left + lane_width
        right_pixels = NO_PIXELS
        mode = FitMode.RESCUED
    elif right is not None and left is None:
        left = right - lane_width
        left_pixels = NO_PIXELS
        mode = FitMode.RESCUED
    elif left is None and right is None:
        mode = FitMode.NONE
    return LaneFit(left, right, left_pixels, right_pixels, mode)


def fit_lane(binary: np.ndarray, lane_width: np.ndarray | None = None) -> LaneFit:
    """Find the two lane lines in a bird's-eye binary image and fit their polynomials.

    The lines are searched from scratch: the histogram gives where they start and the
    sliding windows follow them. See :func:`fit_from_pixels` for how the lines are fitted.

    Args:
        binary: Bird's-eye binary image (lane pixels are non-zero).
        lane_width: Coefficients of the lane width in pixels (right curve minus left curve)
            used to derive a missing line, by default a lane of the nominal constant width.

    Returns:
        The fitted lines and how they were obtained. Use :meth:`LaneFit.is_plausible` to
        check the result; it may not describe a lane when the image shows none.

    Raises:
        ValueError: If the input is not a single-channel image.
    """
    _require_image(binary)
    height, width = binary.shape
    left_pixels, right_pixels = search_windows(binary, find_base_positions(binary))
    return fit_from_pixels(left_pixels, right_pixels, (width, height), lane_width)
