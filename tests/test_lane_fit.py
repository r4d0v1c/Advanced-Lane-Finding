"""Tests for lane_finding.lane_fit."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic import curve, draw_lane, lane, max_error

from lane_finding.lane_fit import (
    FitMode,
    LaneFit,
    find_base_positions,
    fit_lane,
    fit_polynomial,
    fit_shared_curvature,
    search_windows,
)

IMAGES_DIR = Path(__file__).resolve().parents[1] / "resources" / "test_images_export"
SIZES = [(1280, 720), (960, 540)]
EMPTY = (np.empty(0, np.intp), np.empty(0, np.intp))


# ------------------------------------------------------------------- find_base_positions


@pytest.mark.parametrize("size", SIZES)
def test_find_base_positions_finds_where_the_lines_start(size):
    """The bases are the columns of the two lines in the lower half."""
    left, right = lane(size, slope=-0.1)
    bases = find_base_positions(draw_lane(size, left, right))
    assert bases[0] == pytest.approx(0.25 * size[0], abs=0.02 * size[0])
    assert bases[1] == pytest.approx(0.75 * size[0], abs=0.02 * size[0])


def test_find_base_positions_ignores_pixels_outside_the_corridors():
    """Large noise at the image border does not attract the bases."""
    size = (1280, 720)
    left, right = lane(size)
    image = draw_lane(size, left, right, blocks=[(400, 0, 300, 70), (400, 1210, 300, 70)])
    bases = find_base_positions(image)
    assert bases[0] == pytest.approx(320, abs=25)
    assert bases[1] == pytest.approx(960, abs=25)


def test_find_base_positions_without_pixels_gives_the_nominal_columns():
    """An empty image gives the columns the lane is mapped to."""
    assert find_base_positions(np.zeros((720, 1280), np.uint8)) == (320, 960)


def test_find_base_positions_rejects_a_color_image():
    """Only single-channel images are accepted."""
    with pytest.raises(ValueError, match="single-channel"):
        find_base_positions(np.zeros((720, 1280, 3), np.uint8))


# ----------------------------------------------------------------------- search_windows


def test_search_windows_follows_curved_lines():
    """The windows collect almost all pixels of strongly curved lines and nothing else."""
    size = (1280, 720)
    left, right = lane(size, slope=-0.15, shift=0.25)
    image = draw_lane(size, left, right)
    bases = find_base_positions(image)

    (left_rows, left_columns), (right_rows, right_columns) = search_windows(image, bases)

    assert left_rows.size + right_rows.size >= 0.97 * np.count_nonzero(image)
    assert np.all(left_columns < right_columns.min())


def test_search_windows_bridge_the_gaps_of_a_dashed_line():
    """A window that sees no dash keeps going in the direction of the line."""
    size = (1280, 720)
    left, right = lane(size, slope=-0.1, shift=0.2)
    image = draw_lane(size, left, right, right_dashed=True)
    right_only = draw_lane(size, None, right, right_dashed=True)

    _, (rows, _) = search_windows(image, find_base_positions(image))

    assert rows.size >= 0.9 * np.count_nonzero(right_only)


def test_search_windows_do_not_collect_noise_far_from_the_lines():
    """Pixels outside the windows are not collected."""
    size = (1280, 720)
    left, right = lane(size)
    image = draw_lane(size, left, right, blocks=[(100, 560, 400, 80)])
    (left_rows, _), (right_rows, _) = search_windows(image, find_base_positions(image))
    assert left_rows.size + right_rows.size == np.count_nonzero(draw_lane(size, left, right))


# --------------------------------------------------------------------- fit_polynomial


def cloud(coefficients: np.ndarray, noise: float, seed: int = 0):
    """Create the pixels of a thick line along a curve.

    Args:
        coefficients: Coefficients of the curve.
        noise: Largest sideways offset of a pixel from the curve.
        seed: Seed of the random generator.

    Returns:
        The rows and columns of the pixels.
    """
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(720), 10)
    columns = np.polyval(coefficients, rows) + rng.uniform(-noise, noise, rows.size)
    return rows, np.round(columns).astype(int)


def test_fit_polynomial_recovers_a_quadratic():
    """The coefficients of a noisy line are recovered."""
    truth = np.array([2e-4, -0.3, 400.0])
    fitted = fit_polynomial(*cloud(truth, noise=10))
    assert fitted == pytest.approx(truth, rel=0.05, abs=1.0)
    assert max_error(fitted, truth, (1280, 720)) < 1.5


def test_fit_polynomial_ignores_outliers():
    """A cluster of stray pixels does not pull the fit away, unlike least squares."""
    truth = np.array([1e-4, -0.2, 300.0])
    rows, columns = cloud(truth, noise=8)
    outlier_rows = np.arange(400, 520)  # a burst of noise in 17 % of the rows
    outlier_columns = np.full(outlier_rows.size, 520)
    all_rows = np.concatenate([rows, np.repeat(outlier_rows, 4)])
    all_columns = np.concatenate([columns, np.repeat(outlier_columns, 4)])

    robust = fit_polynomial(all_rows, all_columns)
    least_squares = np.polyfit(all_rows, all_columns, 2)

    assert max_error(robust, truth, (1280, 720)) < 2.0
    assert max_error(least_squares, truth, (1280, 720)) > 10.0


def test_fit_polynomial_needs_enough_rows():
    """Pixels in fewer than three rows give no fit."""
    assert fit_polynomial(np.array([5, 5, 6, 6]), np.array([10, 11, 10, 11])) is None
    assert fit_polynomial(*EMPTY) is None


# ---------------------------------------------------------------- fit_shared_curvature


def test_fit_shared_curvature_gives_both_lines_the_same_curvature():
    """Two lines with one curvature are recovered, also when one has few rows."""
    left_truth = np.array([2e-4, -0.3, 300.0])
    right_truth = np.array([2e-4, -0.25, 950.0])
    left_rows, left_columns = cloud(left_truth, noise=8, seed=1)
    right_rows, right_columns = cloud(right_truth, noise=8, seed=2)
    keep = right_rows > 450  # only the lower part of the right line is seen

    left, right = fit_shared_curvature(
        (left_rows, left_columns), (right_rows[keep], right_columns[keep])
    )

    assert left[0] == right[0]
    assert max_error(left, left_truth, (1280, 720)) < 2.0
    assert max_error(right, right_truth, (1280, 720)) < 4.0


def test_fit_shared_curvature_needs_both_lines():
    """A line in fewer than three rows gives no fit."""
    rows, columns = cloud(np.array([0.0, 0.0, 300.0]), noise=3)
    assert fit_shared_curvature((rows, columns), EMPTY) is None


# ------------------------------------------------------------------------ is_plausible


def fit_of(left, right) -> LaneFit:
    """Wrap two curves into a lane fit without pixels."""
    return LaneFit(left, right, EMPTY, EMPTY, FitMode.INDEPENDENT)


@pytest.mark.parametrize("size", SIZES)
def test_is_plausible_accepts_a_normal_lane(size):
    """A lane of the expected width, also a bent one, is plausible."""
    for shift in (0.0, 0.3):
        left, right = lane(size, slope=-0.1, shift=shift)
        assert fit_of(left, right).is_plausible(size)


@pytest.mark.parametrize(
    "left, right",
    [
        (None, curve((1280, 720), 0.75, 0, 0)),  # a line is missing
        (curve((1280, 720), 0.75, 0, 0), curve((1280, 720), 0.25, 0, 0)),  # lines crossed
        (curve((1280, 720), 0.35, 0, 0), curve((1280, 720), 0.65, 0, 0)),  # lane too narrow
        (curve((1280, 720), 0.05, 0, 0), curve((1280, 720), 0.95, 0, 0)),  # lane too wide
        (curve((1280, 720), 0.25, 0, 3), curve((1280, 720), 0.75, 0, 3)),  # absurdly curved
        (curve((1280, 720), 0.25, 0, 0), curve((1280, 720), 0.75, 0.9, 0)),  # lines diverge
    ],
    ids=["missing", "crossed", "narrow", "wide", "curved", "diverging"],
)
def test_is_plausible_rejects_impossible_lanes(left, right):
    """Curves that cannot be the boundaries of a lane are rejected."""
    assert not fit_of(left, right).is_plausible((1280, 720))


# --------------------------------------------------------------------------- fit_lane


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("shift", [0.0, 0.1, -0.25], ids=["straight", "right bend", "left bend"])
def test_fit_lane_on_a_clean_lane(size, shift):
    """Two solid lines are fitted independently and accurately."""
    left, right = lane(size, slope=0.05, shift=shift)

    fit = fit_lane(draw_lane(size, left, right))

    assert fit.mode is FitMode.INDEPENDENT
    assert fit.is_plausible(size)
    assert max_error(fit.left, left, size) < 3
    assert max_error(fit.right, right, size) < 3
    assert fit.left_pixels[0].size > 0 and fit.right_pixels[0].size > 0


@pytest.mark.parametrize("size", SIZES)
def test_fit_lane_with_a_dashed_line(size):
    """A dashed right line is found and fitted well, also in a bend."""
    left, right = lane(size, slope=-0.08, shift=0.2)

    fit = fit_lane(draw_lane(size, left, right, right_dashed=True))

    assert fit.is_plausible(size)
    assert max_error(fit.left, left, size) < 3
    assert max_error(fit.right, right, size) < 0.01 * size[0]


@pytest.mark.parametrize("size", SIZES)
def test_fit_lane_derives_a_missing_line(size):
    """A line without any pixels is derived from the other one."""
    left, right = lane(size, slope=0.05, shift=0.15)

    fit = fit_lane(draw_lane(size, left, None))

    assert fit.mode is FitMode.RESCUED
    assert fit.right_pixels[0].size == 0
    assert fit.is_plausible(size)
    assert max_error(fit.right, right, size) < 3


def test_fit_lane_derives_a_missing_left_line():
    """The same holds when the left line is missing."""
    size = (1280, 720)
    left, right = lane(size, slope=0.05, shift=0.15)

    fit = fit_lane(draw_lane(size, None, right))

    assert fit.mode is FitMode.RESCUED
    assert fit.left_pixels[0].size == 0
    assert max_error(fit.left, left, size) < 3


@pytest.mark.parametrize("size", SIZES)
def test_fit_lane_shares_the_curvature_with_a_weakly_seen_line(size):
    """A line seen only in a short part is extended with the curvature of the other line."""
    height = size[1]
    left, right = lane(size, slope=-0.05, shift=0.2)
    hidden = [(0, int(0.62 * height)), (int(0.9 * height), height)]

    fit = fit_lane(draw_lane(size, left, right, right_hidden=hidden))

    assert fit.mode is FitMode.SHARED_CURVATURE
    assert fit.left[0] == fit.right[0]
    assert max_error(fit.right, right, size) < 0.012 * size[0]


@pytest.mark.parametrize("size", SIZES)
def test_fit_lane_ignores_noise_near_the_lines(size):
    """Noise blocks next to the lines do not shift the fit."""
    width, height = size
    left, right = lane(size, slope=0.05, shift=0.1)
    size_block = round(0.05 * width)
    blocks = [
        (int(0.3 * height), int(0.25 * width) + 30, size_block, size_block),
        (int(0.6 * height), int(0.75 * width) - 80, size_block, size_block),
        (int(0.45 * height), int(0.25 * width) - 90, size_block, size_block),
    ]

    fit = fit_lane(draw_lane(size, left, right, blocks=blocks))

    assert fit.is_plausible(size)
    assert max_error(fit.left, left, size) < 0.01 * width
    assert max_error(fit.right, right, size) < 0.01 * width


def test_fit_lane_follows_a_line_with_a_gap():
    """A gap in a solid line in a bend does not lose the line."""
    size = (1280, 720)
    left, right = lane(size, slope=-0.1, shift=0.25)

    fit = fit_lane(draw_lane(size, left, right, left_hidden=[(250, 470)]))

    assert fit.mode is FitMode.INDEPENDENT
    assert max_error(fit.left, left, size) < 3


def test_fit_lane_of_an_empty_image_finds_nothing():
    """An image without lane pixels gives no lane."""
    fit = fit_lane(np.zeros((720, 1280), np.uint8))
    assert fit.mode is FitMode.NONE
    assert fit.left is None and fit.right is None
    assert not fit.is_plausible((1280, 720))


def test_fit_lane_rejects_a_color_image():
    """Only single-channel images are accepted."""
    with pytest.raises(ValueError, match="single-channel"):
        fit_lane(np.zeros((720, 1280, 3), np.uint8))


# ---------------------------------------------------------------------- real images


@pytest.mark.parametrize("name", ["straight_lines1", "straight_lines2"])
def test_fit_lane_on_the_straight_road_images(name, birds_eye_binary):
    """On a straight road the lines are found where the lane is mapped to, without bend."""
    image = cv2.imread(str(IMAGES_DIR / f"{name}.jpg"))
    fit = fit_lane(birds_eye_binary(image))

    assert fit.mode is FitMode.INDEPENDENT
    assert fit.is_plausible((1280, 720))
    assert np.polyval(fit.left, 719) == pytest.approx(320, abs=30)
    assert np.polyval(fit.right, 719) == pytest.approx(960, abs=30)
    assert max(abs(fit.left[0]), abs(fit.right[0])) * 720**2 / 1280 < 0.1


@pytest.mark.parametrize("path", sorted(IMAGES_DIR.glob("*.jpg")), ids=lambda path: path.stem)
def test_fit_lane_is_plausible_on_every_test_image(path, birds_eye_binary):
    """All provided test images give a plausible lane."""
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    fit = fit_lane(birds_eye_binary(image))
    assert fit.is_plausible((width, height))


# The horizon row of every video was measured (see cameras.py): it differs a little.
@pytest.mark.parametrize(
    "video, seconds, horizon",
    [
        ("project_video03.mp4", 5, 422.0),
        ("project_video03.mp4", 20, 422.0),
        ("project_video03.mp4", 32, 422.0),
        ("challenge01.mp4", 7, 420.0),
        ("project_video01.mp4", 4, 304.5),
        ("project_video02.mp4", 12, 310.2),
    ],
)
def test_fit_lane_is_plausible_on_video_frames(
    video, seconds, horizon, birds_eye_binary, read_video_frame
):
    """Frames of the videos with a clear lane give a plausible lane."""
    frame = read_video_frame(video, seconds)
    height, width = frame.shape[:2]
    fit = fit_lane(birds_eye_binary(frame, horizon))
    assert fit.is_plausible((width, height))
