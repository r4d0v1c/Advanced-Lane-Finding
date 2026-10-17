"""Tests for lane_finding.measurements."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic import draw_lane

from lane_finding.cameras import CAMERAS, MAIN_CAMERA, SECOND_CAMERA, CameraProfile
from lane_finding.lane_fit import FitMode, LaneFit, fit_lane
from lane_finding.measurements import (
    CAR_POSITION_FRACTION,
    LANE_WIDTH_METERS,
    LaneMeasurements,
    MetersPerPixel,
    measure_lane,
    meters_per_pixel,
    polynomial_in_meters,
)

IMAGES_DIR = Path(__file__).resolve().parents[1] / "resources" / "test_images_export"
EMPTY = (np.empty(0, np.intp), np.empty(0, np.intp))


def physical_lane(
    camera: CameraProfile,
    scale: MetersPerPixel,
    *,
    curvature: float = 0.0,
    vehicle_offset: float = 0.0,
    heading: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the lane lines of a lane with a known shape and a known vehicle position.

    The lane center is a parabola in meters: ``x(s) = -offset + heading * s + curvature *
    s**2 / 2`` where ``s`` is the distance ahead of the bottom row and ``x`` the lateral
    position relative to the vehicle's axis (positive to the right). The lane lines are
    half a lane width to both sides of it.

    Args:
        camera: Camera profile that gives the image size.
        scale: Meters per pixel of the bird's-eye view.
        curvature: Curvature of the lane in 1/m (positive for a bend to the right).
        vehicle_offset: Position of the vehicle relative to the lane center in meters
            (positive if it is to the right of the center).
        heading: Direction of the lane relative to the vehicle (meters of lateral
            movement per meter ahead).

    Returns:
        The pixel coefficients ``(A, B, C)`` of the left and the right line.
    """
    width, height = camera.size
    bottom = height - 1
    xm, ym = scale.horizontal, scale.vertical
    a = curvature * ym**2 / (2 * xm)
    b = (-heading * ym - curvature * ym**2 * bottom) / xm
    center = (
        CAR_POSITION_FRACTION * width
        + (-vehicle_offset + heading * ym * bottom + curvature * ym**2 * bottom**2 / 2) / xm
    )
    half_lane = LANE_WIDTH_METERS / (2 * xm)
    return np.array([a, b, center - half_lane]), np.array([a, b, center + half_lane])


def exact_fit(left: np.ndarray, right: np.ndarray) -> LaneFit:
    """Wrap two curves into a lane fit without pixels.

    Args:
        left: Coefficients of the left line.
        right: Coefficients of the right line.

    Returns:
        The lane fit.
    """
    return LaneFit(left, right, EMPTY, EMPTY, FitMode.INDEPENDENT)


# ------------------------------------------------------------------- meters_per_pixel


def test_meters_per_pixel_of_the_main_camera():
    """The lane (640 pixels) is 3.7 m wide and the view covers about 29 m ahead."""
    scale = meters_per_pixel(MAIN_CAMERA)
    assert scale.horizontal == pytest.approx(3.7 / 640)
    assert scale.vertical == pytest.approx(0.0408, abs=0.0005)
    assert scale.vertical * MAIN_CAMERA.size[1] == pytest.approx(29.4, abs=0.4)


def test_meters_per_pixel_of_the_second_camera():
    """The second camera looks at a narrower angle, so one pixel is more road."""
    scale = meters_per_pixel(SECOND_CAMERA)
    assert scale.horizontal == pytest.approx(3.7 / 480)
    assert scale.vertical == pytest.approx(0.0718, abs=0.001)
    assert scale.vertical > meters_per_pixel(MAIN_CAMERA).vertical


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_meters_per_pixel_changes_little_with_the_horizon(camera):
    """A different horizon changes the vertical scale slightly, in the expected direction."""
    default = meters_per_pixel(camera).vertical
    lower = meters_per_pixel(camera, camera.horizon_y + 10).vertical
    higher = meters_per_pixel(camera, camera.horizon_y - 10).vertical
    assert lower < default < higher
    assert abs(lower / default - 1) < 0.03
    assert abs(higher / default - 1) < 0.03


def test_meters_per_pixel_rejects_a_horizon_without_road():
    """A horizon at the near edge leaves no road to measure."""
    with pytest.raises(ValueError, match="no road"):
        meters_per_pixel(MAIN_CAMERA, MAIN_CAMERA.bottom_y - MAIN_CAMERA.top_offset)


# --------------------------------------------------------------- polynomial_in_meters


def test_polynomial_in_meters_equals_a_fit_in_meters():
    """Scaling the coefficients is the same as fitting the scaled points."""
    scale = MetersPerPixel(horizontal=0.006, vertical=0.04)
    coefficients = np.array([2e-4, -0.3, 400.0])
    rows = np.linspace(0, 719, 50)
    columns = np.polyval(coefficients, rows)

    refit = np.polyfit(rows * scale.vertical, columns * scale.horizontal, 2)

    assert polynomial_in_meters(coefficients, scale) == pytest.approx(refit, rel=1e-6)


# ---------------------------------------------------------------------- measure_lane


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
@pytest.mark.parametrize(
    "curvature, offset",
    [(1 / 400, 0.3), (-1 / 800, -0.4), (1 / 1500, 0.0), (-1 / 250, 0.1)],
)
def test_measure_lane_recovers_the_curvature_and_the_offset_of_exact_curves(
    camera, curvature, offset
):
    """The conversion to meters is exact for curves of a known shape."""
    scale = meters_per_pixel(camera)
    left, right = physical_lane(
        camera, scale, curvature=curvature, vehicle_offset=offset, heading=0.02
    )

    measured = measure_lane(exact_fit(left, right), camera.size, scale)

    assert measured.curvature == pytest.approx(curvature, rel=2e-3)
    assert measured.radius == pytest.approx(1 / abs(curvature), rel=2e-3)
    assert measured.vehicle_offset == pytest.approx(offset, abs=1e-6)
    assert measured.lane_width == pytest.approx(LANE_WIDTH_METERS, abs=1e-6)


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
@pytest.mark.parametrize(
    "curvature, offset",
    [(1 / 400, 0.3), (-1 / 800, -0.4), (1 / 1500, 0.0)],
)
def test_measure_lane_from_a_drawn_lane(camera, curvature, offset):
    """Detecting a drawn lane and measuring it gives back its radius and position."""
    scale = meters_per_pixel(camera)
    left, right = physical_lane(
        camera, scale, curvature=curvature, vehicle_offset=offset, heading=0.02
    )

    fit = fit_lane(draw_lane(camera.size, left, right, right_dashed=True))
    measured = measure_lane(fit, camera.size, scale)

    assert math.copysign(1, measured.curvature) == math.copysign(1, curvature)
    assert measured.radius == pytest.approx(1 / abs(curvature), rel=0.15)
    assert measured.vehicle_offset == pytest.approx(offset, abs=0.05)
    assert measured.lane_width == pytest.approx(LANE_WIDTH_METERS, abs=0.15)


def test_measure_lane_of_a_straight_lane():
    """A straight lane has a very large radius."""
    scale = meters_per_pixel(MAIN_CAMERA)
    left, right = physical_lane(MAIN_CAMERA, scale, vehicle_offset=-0.2)

    fit = fit_lane(draw_lane(MAIN_CAMERA.size, left, right, right_dashed=True))
    measured = measure_lane(fit, MAIN_CAMERA.size, scale)

    assert measured.radius > 10000
    assert measured.vehicle_offset == pytest.approx(-0.2, abs=0.03)


def test_vehicle_offset_is_positive_to_the_right_of_the_lane_center():
    """A lane center to the left of the vehicle's axis means the vehicle is to the right."""
    scale = meters_per_pixel(MAIN_CAMERA)
    width = MAIN_CAMERA.size[0]
    half_lane = LANE_WIDTH_METERS / 2 / scale.horizontal
    shift = 0.5 / scale.horizontal  # the lane center is 0.5 m to the left of the axis
    left = np.array([0.0, 0.0, width / 2 - shift - half_lane])
    right = np.array([0.0, 0.0, width / 2 - shift + half_lane])

    measured = measure_lane(exact_fit(left, right), MAIN_CAMERA.size, scale)

    assert measured.vehicle_offset == pytest.approx(0.5)
    assert measured.curvature == 0
    assert measured.radius == math.inf


def test_measure_lane_needs_both_lines():
    """Without a lane line there is nothing to measure."""
    scale = meters_per_pixel(MAIN_CAMERA)
    fit = LaneFit(np.array([0.0, 0.0, 300.0]), None, EMPTY, EMPTY, FitMode.NONE)
    assert measure_lane(fit, MAIN_CAMERA.size, scale) is None


def test_radius_is_the_inverse_of_the_curvature():
    """The radius does not depend on the direction of the bend."""
    assert LaneMeasurements(0.002, 0.0, 3.7).radius == pytest.approx(500)
    assert LaneMeasurements(-0.002, 0.0, 3.7).radius == pytest.approx(500)
    assert LaneMeasurements(0.0, 0.0, 3.7).radius == math.inf


# -------------------------------------------------------------------- real images


@pytest.mark.parametrize("name", ["straight_lines1", "straight_lines2"])
def test_straight_road_images_have_a_large_radius_and_a_small_offset(name, birds_eye_binary):
    """On a straight road the radius is large and the vehicle is near the lane center."""
    scale = meters_per_pixel(MAIN_CAMERA)
    image = cv2.imread(str(IMAGES_DIR / f"{name}.jpg"))

    measured = measure_lane(fit_lane(birds_eye_binary(image)), MAIN_CAMERA.size, scale)

    assert measured.radius > 3000
    assert abs(measured.vehicle_offset) < 0.35
    assert measured.lane_width == pytest.approx(LANE_WIDTH_METERS, abs=0.4)


# The horizon row of every video was measured (see cameras.py): it differs a little.
@pytest.mark.parametrize(
    "video, seconds, horizon, camera",
    [
        ("project_video03.mp4", 5, 422.0, MAIN_CAMERA),
        ("project_video03.mp4", 20, 422.0, MAIN_CAMERA),
        ("project_video03.mp4", 32, 422.0, MAIN_CAMERA),
        ("challenge01.mp4", 7, 420.0, MAIN_CAMERA),
        ("project_video02.mp4", 12, 310.2, SECOND_CAMERA),
    ],
)
def test_video_frames_give_realistic_measurements(
    video, seconds, horizon, camera, birds_eye_binary, read_video_frame
):
    """Frames with a clear lane give a realistic radius, offset and lane width."""
    frame = read_video_frame(video, seconds)
    fit = fit_lane(birds_eye_binary(frame, horizon))

    measured = measure_lane(fit, camera.size, meters_per_pixel(camera, horizon))

    assert measured.radius > 200
    assert abs(measured.vehicle_offset) < 0.9
    assert measured.lane_width == pytest.approx(LANE_WIDTH_METERS, abs=0.5)


# ------------------------------------------------------------------------ text_lines


def test_text_lines_describe_a_bend_and_the_position():
    """The text names the radius, the direction and the side of the vehicle."""
    lines = LaneMeasurements(curvature=1 / 850, vehicle_offset=-0.304, lane_width=3.71).text_lines()
    assert lines == [
        "Radius: 850 m (right)",
        "Vehicle: 0.30 m left of center",
        "Lane width: 3.71 m",
    ]


def test_text_lines_show_a_gentle_bend_as_straight():
    """A radius above the limit is shown as straight, a vehicle on the axis as centered."""
    lines = LaneMeasurements(curvature=-1 / 5000, vehicle_offset=0.002, lane_width=3.7).text_lines()
    assert lines[0] == "Radius: straight"
    assert lines[1] == "Vehicle: at the center"
    assert LaneMeasurements(0.0, 0.3, 3.7).text_lines()[1] == "Vehicle: 0.30 m right of center"
    assert "left" in LaneMeasurements(-1 / 900, 0.0, 3.7).text_lines()[0]
