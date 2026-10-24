"""Tests for lane_finding.perspective."""

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pytest

from lane_finding.cameras import CAMERAS, MAIN_CAMERA, SECOND_CAMERA, CameraProfile
from lane_finding.perspective import (
    LANE_MARGIN_FRACTION,
    PerspectiveTransform,
    destination_points,
)
from lane_finding.thresholding import threshold_lane_pixels

IMAGES_DIR = Path(__file__).resolve().parents[1] / "resources" / "test_images_export"


class LaneLineMeasurement(NamedTuple):
    """Position of the two lane lines in a bird's-eye binary image.

    Attributes:
        left_slope: Horizontal pixels per row of the left line (0 for a vertical line).
        right_slope: The same for the right line.
        width_far: Distance between the lines at 45 % of the image height.
        width_near: Distance between the lines at 95 % of the image height.
    """

    left_slope: float
    right_slope: float
    width_far: float
    width_near: float


def measure_lane_lines(binary: np.ndarray, camera: CameraProfile) -> LaneLineMeasurement:
    """Fit a straight line to each lane line of a bird's-eye binary image.

    The lines are searched around the columns the lane is mapped to, in the lower 60 %
    of the image where they are closest to straight even on a bend.

    Args:
        binary: Bird's-eye binary image of the camera's size.
        camera: Camera the image comes from.

    Returns:
        Slopes of both lines and the lane width at two heights.
    """
    width, height = camera.size
    fits = []
    for center in (LANE_MARGIN_FRACTION * width, (1 - LANE_MARGIN_FRACTION) * width):
        first_column = int(center - 0.1 * width)
        last_column = int(center + 0.1 * width)
        rows, columns = [], []
        for row in range(int(0.4 * height), height):
            hits = np.flatnonzero(binary[row, first_column:last_column])
            if hits.size >= 2:
                rows.append(row)
                columns.append(first_column + hits.mean())
        assert len(rows) >= 20, "lane line not found"
        fits.append(np.polyfit(rows, columns, 1))
    (left_slope, left_offset), (right_slope, right_offset) = fits

    def lane_width(row: float) -> float:
        """Compute the distance between the fitted lines at a row."""
        return (right_slope * row + right_offset) - (left_slope * row + left_offset)

    return LaneLineMeasurement(
        left_slope, right_slope, lane_width(0.45 * height), lane_width(0.95 * height)
    )


def test_destination_points_are_the_middle_half_of_the_image():
    """The lane is mapped to a rectangle covering the middle half of the width."""
    expected = [[320, 0], [960, 0], [960, 720], [320, 720]]
    assert destination_points((1280, 720)).tolist() == expected


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_matrices_map_the_source_points_to_the_destination_points_and_back(camera):
    """The transform maps the trapezoid to the rectangle, the inverse maps it back."""
    transform = PerspectiveTransform.for_camera(camera)
    source = transform.source_points.reshape(-1, 1, 2)
    destination = transform.destination_points.reshape(-1, 1, 2)
    forward = cv2.perspectiveTransform(source, transform.matrix)
    backward = cv2.perspectiveTransform(destination, transform.inverse_matrix)
    assert np.allclose(forward, destination, atol=1e-2)
    assert np.allclose(backward, source, atol=1e-2)


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_matrix_and_inverse_matrix_cancel_out(camera):
    """A matrix times its inverse is the identity (up to the scale of a homography)."""
    transform = PerspectiveTransform.for_camera(camera)
    product = transform.matrix @ transform.inverse_matrix
    assert np.allclose(product / product[2, 2], np.eye(3), atol=1e-6)


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_lines_along_the_trapezoid_edges_become_vertical(camera):
    """Lane lines on the trapezoid edges are mapped to vertical lines at the lane columns."""
    transform = PerspectiveTransform.for_camera(camera)
    width, height = camera.size
    top_left, top_right, bottom_right, bottom_left = np.round(transform.source_points).astype(int)
    image = np.zeros((height, width), np.uint8)
    cv2.line(image, tuple(top_left), tuple(bottom_left), 255, 3)
    cv2.line(image, tuple(top_right), tuple(bottom_right), 255, 3)

    warped = transform.warp(image)

    for row in (int(0.05 * height), height // 2, int(0.95 * height)):
        columns = np.arange(width)
        left_half, right_half = warped[row, : width // 2], warped[row, width // 2 :]
        left = np.average(columns[: width // 2], weights=left_half)
        right = np.average(columns[width // 2 :], weights=right_half)
        assert left == pytest.approx(LANE_MARGIN_FRACTION * width, abs=3)
        assert right == pytest.approx((1 - LANE_MARGIN_FRACTION) * width, abs=3)


def test_warp_then_unwarp_restores_the_lane_area():
    """Transforming forth and back keeps the lane stripes of a bird's-eye image."""
    transform = PerspectiveTransform.for_camera(MAIN_CAMERA)
    bird = np.zeros((720, 1280), np.uint8)
    bird[:, 308:332] = 255
    bird[:, 948:972] = 255

    restored = transform.warp(transform.unwarp(bird))

    center = (slice(100, 700), slice(250, 1030))
    assert np.abs(restored[center].astype(int) - bird[center].astype(int)).mean() < 10


@pytest.mark.parametrize("shape", [(720, 1280), (720, 1280, 3)], ids=["gray", "color"])
def test_warp_and_unwarp_keep_shape_and_dtype(shape):
    """The result has the shape and dtype of the input."""
    transform = PerspectiveTransform.for_camera(MAIN_CAMERA)
    image = np.zeros(shape, np.uint8)
    for result in (transform.warp(image), transform.unwarp(image)):
        assert result.shape == shape
        assert result.dtype == np.uint8


def test_nearest_interpolation_keeps_binary_values():
    """Warping a binary image with nearest-neighbour interpolation creates no gray values."""
    transform = PerspectiveTransform.for_camera(MAIN_CAMERA)
    binary = (np.random.default_rng(0).random((720, 1280)) > 0.5).astype(np.uint8) * 255
    assert set(np.unique(transform.warp(binary, cv2.INTER_NEAREST))) <= {0, 255}


def test_warp_and_unwarp_reject_other_sizes():
    """Images of another size cannot be transformed."""
    transform = PerspectiveTransform.for_camera(MAIN_CAMERA)
    image = np.zeros((540, 960, 3), np.uint8)
    with pytest.raises(ValueError, match="960x540"):
        transform.warp(image)
    with pytest.raises(ValueError, match="960x540"):
        transform.unwarp(image)


@pytest.mark.parametrize("name", ["straight_lines1", "straight_lines2"])
def test_straight_road_of_the_main_camera_gives_vertical_parallel_lines(name, calibration):
    """On the images used to derive the geometry the lines are vertical, 640 pixels apart."""
    image = calibration.undistort(cv2.imread(str(IMAGES_DIR / f"{name}.jpg")))
    transform = PerspectiveTransform.for_camera(MAIN_CAMERA)
    mask = threshold_lane_pixels(image, MAIN_CAMERA.lane_widths())
    binary = transform.warp(mask, cv2.INTER_NEAREST)

    measurement = measure_lane_lines(binary, MAIN_CAMERA)

    assert abs(measurement.left_slope) < 0.04  # about 2 degrees
    assert abs(measurement.right_slope) < 0.04
    assert measurement.width_far == pytest.approx(640, abs=30)
    assert measurement.width_near == pytest.approx(640, abs=30)


@pytest.mark.parametrize("name", ["solidYellowLeft", "solidWhiteRight"])
def test_straight_road_of_the_second_camera_gives_vertical_parallel_lines(name):
    """The same holds for the second camera (no undistortion), 480 pixels apart."""
    image = cv2.imread(str(IMAGES_DIR / f"{name}.jpg"))
    transform = PerspectiveTransform.for_camera(SECOND_CAMERA)
    mask = threshold_lane_pixels(image, SECOND_CAMERA.lane_widths())
    binary = transform.warp(mask, cv2.INTER_NEAREST)

    measurement = measure_lane_lines(binary, SECOND_CAMERA)

    assert abs(measurement.left_slope) < 0.05
    assert abs(measurement.right_slope) < 0.05
    assert measurement.width_far == pytest.approx(480, abs=30)
    assert measurement.width_near == pytest.approx(480, abs=30)


@pytest.mark.parametrize(
    "video, seconds, camera",
    [
        ("project_video03.mp4", 5, MAIN_CAMERA),
        ("project_video03.mp4", 20, MAIN_CAMERA),
        ("project_video03.mp4", 32, MAIN_CAMERA),
        ("challenge01.mp4", 7, MAIN_CAMERA),
        ("project_video01.mp4", 4, SECOND_CAMERA),
        ("project_video02.mp4", 5, SECOND_CAMERA),
        ("project_video02.mp4", 12, SECOND_CAMERA),
    ],
)
def test_lane_width_is_right_on_frames_not_used_for_the_geometry(
    video, seconds, camera, calibration, read_video_frame
):
    """On video frames the lane width stays within 12 % of the expected width."""
    frame = read_video_frame(video, seconds)
    if camera.size == calibration.image_size:
        frame = calibration.undistort(frame)
    transform = PerspectiveTransform.for_camera(camera)
    binary = transform.warp(threshold_lane_pixels(frame, camera.lane_widths()), cv2.INTER_NEAREST)

    measurement = measure_lane_lines(binary, camera)

    expected = (1 - 2 * LANE_MARGIN_FRACTION) * camera.size[0]
    assert measurement.width_far == pytest.approx(expected, rel=0.12)
    assert measurement.width_near == pytest.approx(expected, rel=0.12)


@pytest.mark.parametrize("seconds", [1, 12])
def test_measured_horizon_removes_the_funnel_effect_in_challenge02(
    seconds, calibration, read_video_frame
):
    """The camera of challenge02 looks lower; with its own horizon the lines are parallel."""
    frame = calibration.undistort(read_video_frame("challenge02.mp4", seconds))
    binary = threshold_lane_pixels(frame, MAIN_CAMERA.lane_widths())

    def width_ratio(horizon_y: float | None) -> float:
        """Compute the lane width far away divided by the width nearby."""
        transform = PerspectiveTransform.for_camera(MAIN_CAMERA, horizon_y)
        measurement = measure_lane_lines(transform.warp(binary, cv2.INTER_NEAREST), MAIN_CAMERA)
        return measurement.width_far / measurement.width_near

    assert width_ratio(None) < 0.9
    assert width_ratio(444.0) == pytest.approx(1.0, abs=0.06)
