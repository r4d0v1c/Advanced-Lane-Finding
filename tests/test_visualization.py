"""Tests for lane_finding.visualization."""

import numpy as np
import pytest

from lane_finding.lane_fit import FitMode, LaneFit
from lane_finding.visualization import (
    CURVE_COLOR,
    LEFT_PIXEL_COLOR,
    RESCUED_CURVE_COLOR,
    RIGHT_PIXEL_COLOR,
    TEXT_COLOR,
    TEXT_OUTLINE_COLOR,
    TITLE_COLOR,
    curve_points,
    draw_lane_fit,
    draw_text_lines,
    side_by_side,
)


def test_side_by_side_combines_color_and_grayscale_panels():
    """Panels are resized and placed in one BGR row."""
    color = np.zeros((100, 160, 3), np.uint8)
    gray = np.zeros((100, 160), np.uint8)
    result = side_by_side([color, gray], ["Color", "Gray"], scale=0.5)
    assert result.shape == (50, 160, 3)
    assert result.dtype == np.uint8


def test_side_by_side_draws_the_titles():
    """Every panel contains title pixels."""
    black = np.zeros((100, 200, 3), np.uint8)
    result = side_by_side([black, black], ["A", "B"])
    for panel in (result[:, :200], result[:, 200:]):
        assert (panel == TITLE_COLOR).all(axis=-1).any()


def test_side_by_side_does_not_modify_its_input():
    """The input images are left untouched."""
    image = np.zeros((100, 200, 3), np.uint8)
    side_by_side([image], ["Title"])
    assert not image.any()


def test_side_by_side_rejects_mismatching_titles():
    """The number of titles must match the number of images."""
    with pytest.raises(ValueError, match="titles"):
        side_by_side([np.zeros((10, 10), np.uint8)], ["A", "B"])


def test_side_by_side_rejects_images_of_different_heights():
    """Images of different heights cannot be placed in one row."""
    with pytest.raises(ValueError, match="height"):
        side_by_side([np.zeros((10, 10), np.uint8), np.zeros((20, 10), np.uint8)], ["A", "B"])


def test_curve_points_gives_one_point_per_row():
    """Every row of the image has a point at the column of the curve."""
    points = curve_points(np.array([0.0, 0.5, 20.0]), (200, 100))

    assert points.shape == (100, 2)
    assert points.dtype == np.int32
    assert points[:, 1].tolist() == list(range(100))
    assert points[0].tolist() == [20, 0]
    assert points[40].tolist() == [40, 40]


def test_curve_points_clips_absurd_curves():
    """Columns far outside the image are limited to one image width around it."""
    points = curve_points(np.array([1e6, 1e9, -1e12]), (200, 100))
    assert points[:, 0].min() >= -200
    assert points[:, 0].max() <= 400


def make_fit(left=None, right=None, left_pixels=None, right_pixels=None) -> LaneFit:
    """Create a lane fit for the drawing tests.

    Args:
        left: Coefficients of the left curve or ``None``.
        right: Coefficients of the right curve or ``None``.
        left_pixels: ``(rows, columns)`` of the left pixels; none by default.
        right_pixels: ``(rows, columns)`` of the right pixels; none by default.

    Returns:
        The lane fit.
    """
    empty = (np.empty(0, np.intp), np.empty(0, np.intp))
    return LaneFit(
        left,
        right,
        left_pixels or empty,
        right_pixels or empty,
        FitMode.INDEPENDENT,
    )


def test_draw_lane_fit_colors_the_pixels_of_each_line():
    """The left pixels are red, the right pixels blue and the others keep their color."""
    binary = np.zeros((100, 200), np.uint8)
    binary[:, 100] = 255
    fit = make_fit(
        left_pixels=(np.array([10, 11]), np.array([20, 21])),
        right_pixels=(np.array([50]), np.array([150])),
    )

    result = draw_lane_fit(binary, fit)

    assert result.shape == (100, 200, 3)
    assert tuple(result[10, 20]) == LEFT_PIXEL_COLOR
    assert tuple(result[11, 21]) == LEFT_PIXEL_COLOR
    assert tuple(result[50, 150]) == RIGHT_PIXEL_COLOR
    assert tuple(result[70, 100]) == (255, 255, 255)


def test_draw_lane_fit_draws_the_curves():
    """A curve with pixels is green, a curve without pixels (derived) is yellow."""
    image = np.zeros((100, 200, 3), np.uint8)
    fit = make_fit(
        left=np.array([0.0, 0.0, 50.0]),
        right=np.array([0.0, 0.0, 150.0]),
        left_pixels=(np.array([10]), np.array([50])),
    )

    result = draw_lane_fit(image, fit)

    assert tuple(result[60, 50]) == CURVE_COLOR
    assert tuple(result[60, 150]) == RESCUED_CURVE_COLOR


def test_draw_lane_fit_does_not_modify_the_input():
    """The input image is left untouched."""
    image = np.zeros((100, 200, 3), np.uint8)
    fit = make_fit(left=np.array([0.0, 0.0, 50.0]), left_pixels=(np.array([5]), np.array([5])))
    draw_lane_fit(image, fit)
    assert not image.any()


def test_draw_lane_fit_without_curves_draws_only_pixels():
    """A fit that found nothing leaves the image as it is."""
    image = np.zeros((100, 200), np.uint8)
    assert not draw_lane_fit(image, make_fit()).any()


def test_draw_lane_fit_survives_absurd_curves():
    """Curves far outside the image are clipped instead of failing."""
    image = np.zeros((100, 200, 3), np.uint8)
    draw_lane_fit(image, make_fit(left=np.array([1e6, 1e9, -1e12])))


def test_draw_text_lines_writes_the_text_and_keeps_the_input():
    """The text appears on a copy; grayscale input gives a BGR image."""
    image = np.zeros((120, 400), np.uint8)

    result = draw_text_lines(image, ["First", "Second"], (10, 40))

    assert not image.any()
    assert result.shape == (120, 400, 3)
    assert (result == TEXT_COLOR).all(axis=-1).any()


def test_draw_text_lines_outlines_the_text_so_that_it_can_be_read_on_a_bright_background():
    """On a white image the text has a dark outline around its yellow strokes."""
    image = np.full((120, 400, 3), 255, np.uint8)

    result = draw_text_lines(image, ["Text"], (10, 60))

    assert (result == TEXT_COLOR).all(axis=-1).any()
    assert (result == TEXT_OUTLINE_COLOR).all(axis=-1).any()


def test_draw_text_lines_puts_the_lines_below_each_other():
    """The second line starts lower than the first one."""
    image = np.zeros((200, 400, 3), np.uint8)

    result = draw_text_lines(image, ["Line", "Line"], (10, 40))

    rows = np.flatnonzero((result == TEXT_COLOR).all(axis=-1).any(axis=1))
    assert rows.min() < 40 and rows.max() > 60  # the first line ends above row 40
    assert np.any(np.diff(rows) > 1)  # a gap between the two lines
