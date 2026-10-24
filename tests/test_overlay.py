"""Tests for lane_finding.overlay."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_finding.cameras import CAMERAS, CameraProfile
from lane_finding.lane_fit import FitMode, LaneFit
from lane_finding.measurements import LaneMeasurements
from lane_finding.overlay import (
    AREA_COLOR,
    AREA_OPACITY,
    LEFT_LINE_COLOR,
    LINE_OPACITY,
    RIGHT_LINE_COLOR,
    draw_lane_overlay,
)
from lane_finding.perspective import PerspectiveTransform
from lane_finding.pipeline import LanePipeline
from lane_finding.tracking import TrackStatus
from lane_finding.visualization import TEXT_COLOR

IMAGES_DIR = Path(__file__).resolve().parents[1] / "resources" / "test_images_export"
EMPTY = (np.empty(0, np.intp), np.empty(0, np.intp))
GRAY = 100
MEASUREMENTS = LaneMeasurements(curvature=0.0, vehicle_offset=0.1, lane_width=3.7)

CAMERA_IDS = [camera.name for camera in CAMERAS]


def make_scene(
    camera: CameraProfile, left: float = 0.25, right: float = 0.75
) -> tuple[np.ndarray, PerspectiveTransform, LaneFit]:
    """Create a gray road image, its transform and a straight lane in the bird's-eye view.

    Args:
        camera: Camera that gives the image size and the transform.
        left: Column of the left line as a fraction of the width.
        right: Column of the right line as a fraction of the width.

    Returns:
        The image, the transform and the lane fit.
    """
    width, height = camera.size
    image = np.full((height, width, 3), GRAY, np.uint8)
    fit = LaneFit(
        np.array([0.0, 0.0, left * width]),
        np.array([0.0, 0.0, right * width]),
        EMPTY,
        EMPTY,
        FitMode.INDEPENDENT,
    )
    return image, PerspectiveTransform.for_camera(camera), fit


def camera_pixel(transform: PerspectiveTransform, column: float, row: float) -> tuple[int, int]:
    """Find where a point of the bird's-eye view is in the camera view.

    Args:
        transform: Perspective transform of the scene.
        column: Column in the bird's-eye view.
        row: Row in the bird's-eye view.

    Returns:
        ``(column, row)`` in the camera view.
    """
    point = np.float32([[[column, row]]])
    x, y = cv2.perspectiveTransform(point, transform.inverse_matrix)[0, 0]
    return round(float(x)), round(float(y))


def blended(color: tuple[int, int, int], opacity: float) -> np.ndarray:
    """Compute what a gray pixel looks like after a color is mixed into it.

    Args:
        color: BGR color that is mixed in.
        opacity: Share of the color in the result.

    Returns:
        The BGR value.
    """
    return np.array(color) * opacity + GRAY * (1 - opacity)


@pytest.mark.parametrize("camera", CAMERAS, ids=CAMERA_IDS)
def test_the_lane_area_is_green(camera):
    """The road between the lane lines is tinted green."""
    image, transform, fit = make_scene(camera)
    width, height = camera.size
    x, y = camera_pixel(transform, 0.5 * width, 0.5 * height)

    result = draw_lane_overlay(image, transform, fit, MEASUREMENTS)

    assert result[y, x] == pytest.approx(blended(AREA_COLOR, AREA_OPACITY), abs=1)
    assert result[y, x - 10] == pytest.approx(blended(AREA_COLOR, AREA_OPACITY), abs=1)


@pytest.mark.parametrize("camera", CAMERAS, ids=CAMERA_IDS)
def test_the_lane_lines_are_red_on_the_left_and_blue_on_the_right(camera):
    """The boundaries of the lane have their own colors."""
    image, transform, fit = make_scene(camera)
    width, height = camera.size

    result = draw_lane_overlay(image, transform, fit, MEASUREMENTS)

    x, y = camera_pixel(transform, 0.25 * width, 0.5 * height)
    assert result[y, x] == pytest.approx(blended(LEFT_LINE_COLOR, LINE_OPACITY), abs=1)
    x, y = camera_pixel(transform, 0.75 * width, 0.5 * height)
    assert result[y, x] == pytest.approx(blended(RIGHT_LINE_COLOR, LINE_OPACITY), abs=1)


@pytest.mark.parametrize("camera", CAMERAS, ids=CAMERA_IDS)
def test_the_road_outside_the_lane_is_not_changed(camera):
    """Only the lane is drawn: neighbouring lanes and the rows below the view stay as they are."""
    image, transform, fit = make_scene(camera)
    width, height = camera.size
    result = draw_lane_overlay(image, transform, fit, MEASUREMENTS)

    for column in (0.05, 0.15, 0.85, 0.95):
        x, y = camera_pixel(transform, column * width, 0.5 * height)
        assert tuple(result[y, x]) == (GRAY,) * 3
    assert np.array_equal(result[-5:], image[-5:])


def test_the_drawing_follows_the_fitted_lane():
    """A lane that is not in the usual place is drawn where it is."""
    camera = CAMERAS[0]
    image, transform, fit = make_scene(camera, left=0.45, right=0.85)
    width, height = camera.size

    result = draw_lane_overlay(image, transform, fit, MEASUREMENTS)

    inside = camera_pixel(transform, 0.65 * width, 0.5 * height)
    usual = camera_pixel(transform, 0.35 * width, 0.5 * height)
    assert result[inside[1], inside[0]] == pytest.approx(blended(AREA_COLOR, AREA_OPACITY), abs=1)
    assert tuple(result[usual[1], usual[0]]) == (GRAY,) * 3


def test_a_curved_lane_is_drawn_as_a_curve():
    """The area follows the polynomials: the lane center moves with the bend."""
    camera = CAMERAS[0]
    image, transform, _ = make_scene(camera)
    width, height = camera.size
    shift = 0.1 * width / height**2  # the lane moves 0.1 W to the right over the image height
    left = np.array([shift, 0.0, 0.25 * width - shift * height**2 / 2])
    right = left + np.array([0.0, 0.0, 0.5 * width])
    fit = LaneFit(left, right, EMPTY, EMPTY, FitMode.INDEPENDENT)

    result = draw_lane_overlay(image, transform, fit, MEASUREMENTS)

    for row in (0.2 * height, 0.5 * height, 0.8 * height):
        center = np.polyval(0.5 * (left + right), row)
        x, y = camera_pixel(transform, center, row)
        assert result[y, x] == pytest.approx(blended(AREA_COLOR, AREA_OPACITY), abs=1)
        x, y = camera_pixel(transform, center + 0.35 * width, row)
        assert tuple(result[y, x]) == (GRAY,) * 3


def test_the_measurements_are_written_at_the_top():
    """The text is yellow and in the upper left corner, not on the road."""
    image, transform, fit = make_scene(CAMERAS[0])

    result = draw_lane_overlay(image, transform, fit, MEASUREMENTS)

    text_rows = np.flatnonzero((result == TEXT_COLOR).all(axis=-1).any(axis=1))
    assert text_rows.size > 0
    assert text_rows.max() < 0.3 * image.shape[0]


def test_without_measurements_only_a_note_is_written():
    """A lane that is not plausible is not drawn."""
    image, transform, fit = make_scene(CAMERAS[0])
    height = image.shape[0]

    result = draw_lane_overlay(image, transform, fit, None)

    assert (result == TEXT_COLOR).all(axis=-1).any()
    assert np.array_equal(result[int(0.3 * height) :], image[int(0.3 * height) :])


def test_a_held_lane_is_not_drawn():
    """A lane held from earlier frames is not shown: only a note, like without a lane."""
    image, transform, fit = make_scene(CAMERAS[0])
    height = image.shape[0]

    held = draw_lane_overlay(image, transform, fit, MEASUREMENTS, TrackStatus.HELD)
    without = draw_lane_overlay(image, transform, fit, None)

    below_text = slice(int(0.35 * height), None)
    assert np.array_equal(held[below_text], image[below_text])
    assert np.array_equal(without[below_text], image[below_text])
    assert np.count_nonzero((held == TEXT_COLOR).all(axis=-1)) > 0


def test_the_input_image_is_not_modified():
    """The overlay is drawn on a copy."""
    image, transform, fit = make_scene(CAMERAS[0])
    original = image.copy()

    result = draw_lane_overlay(image, transform, fit, MEASUREMENTS)

    assert np.array_equal(image, original)
    assert result is not image
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_absurd_curves_do_not_break_the_drawing():
    """Curves far outside the image are clipped instead of failing."""
    image, transform, _ = make_scene(CAMERAS[0])
    wild = LaneFit(
        np.array([1e6, 1e9, -1e12]),
        np.array([-1e6, 0.0, 1e12]),
        EMPTY,
        EMPTY,
        FitMode.INDEPENDENT,
    )
    assert draw_lane_overlay(image, transform, wild, MEASUREMENTS).shape == image.shape


def test_the_overlay_of_a_real_road_image_covers_the_lane(calibration):
    """The detected lane of a real image is tinted green, the sky is not."""
    result = LanePipeline(calibration).process(cv2.imread(str(IMAGES_DIR / "straight_lines1.jpg")))
    width, height = result.camera.size

    overlay = draw_lane_overlay(result.image, result.transform, result.fit, result.measurements)

    x, y = camera_pixel(result.transform, 0.5 * width, 0.5 * height)
    blue, green, red = (int(value) for value in overlay[y, x])
    assert green - red > 50 and green - blue > 50
    sky = slice(height // 2 - 100, height // 2 - 50)
    assert np.array_equal(overlay[sky], result.image[sky])
