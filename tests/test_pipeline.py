"""Tests for lane_finding.pipeline."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_finding.cameras import MAIN_CAMERA, SECOND_CAMERA
from lane_finding.lane_fit import FitMode
from lane_finding.measurements import LANE_WIDTH_METERS, measure_lane, meters_per_pixel
from lane_finding.pipeline import LanePipeline
from lane_finding.thresholding import threshold_lane_pixels

IMAGES_DIR = Path(__file__).resolve().parents[1] / "resources" / "test_images_export"


def read_test_image(name: str) -> np.ndarray:
    """Read one of the provided road images.

    Args:
        name: File name without the extension.

    Returns:
        BGR image.
    """
    image = cv2.imread(str(IMAGES_DIR / f"{name}.jpg"))
    assert image is not None, f"could not read {name}"
    return image


# --------------------------------------------------------------------------- prepare


def test_prepare_undistorts_the_images_of_the_calibrated_camera(calibration):
    """The image of the main camera is undistorted and its binary image comes from that."""
    image = read_test_image("test1")

    prepared = LanePipeline(calibration).prepare(image)

    undistorted = calibration.undistort(image)
    assert prepared.camera is MAIN_CAMERA
    assert np.array_equal(prepared.image, undistorted)
    assert not np.array_equal(prepared.image, image)
    expected = prepared.transform.warp(threshold_lane_pixels(undistorted), cv2.INTER_NEAREST)
    assert np.array_equal(prepared.binary, expected)


def test_prepare_keeps_the_images_of_the_other_camera_as_they_are(calibration):
    """The calibration does not apply to the second camera."""
    image = read_test_image("solidYellowLeft")

    prepared = LanePipeline(calibration).prepare(image)

    assert prepared.camera is SECOND_CAMERA
    assert np.array_equal(prepared.image, image)
    assert prepared.binary.shape == image.shape[:2]


def test_prepare_does_not_undistort_without_a_calibration():
    """Without a calibration no image is undistorted."""
    image = read_test_image("test1")
    assert np.array_equal(LanePipeline().prepare(image).image, image)


def test_prepare_creates_a_binary_birds_eye_image(calibration):
    """The binary image has the size of the camera image and only the values 0 and 255."""
    prepared = LanePipeline(calibration).prepare(read_test_image("test2"))

    assert prepared.binary.dtype == np.uint8
    assert prepared.binary.shape == (720, 1280)
    assert set(np.unique(prepared.binary)) <= {0, 255}
    assert prepared.binary.any()


def test_prepare_rejects_an_unknown_resolution(calibration):
    """Images of a resolution without a camera profile are refused."""
    with pytest.raises(ValueError, match="No camera"):
        LanePipeline(calibration).prepare(np.zeros((100, 200, 3), np.uint8))


def test_prepare_rejects_an_image_that_is_not_bgr(calibration):
    """A grayscale image cannot be thresholded by color."""
    with pytest.raises(ValueError, match="BGR"):
        LanePipeline(calibration).prepare(np.zeros((720, 1280), np.uint8))


def test_the_transform_of_a_camera_is_created_only_once(calibration):
    """Images of the same camera share one transform; each camera has its own."""
    pipeline = LanePipeline(calibration)

    first = pipeline.prepare(read_test_image("test1"))
    second = pipeline.prepare(read_test_image("test2"))
    other = pipeline.prepare(read_test_image("solidYellowLeft"))

    assert first.transform is second.transform
    assert other.transform is not first.transform


def test_the_horizon_row_changes_the_birds_eye_view(calibration):
    """A given horizon row replaces the default one of the camera."""
    image = read_test_image("test1")

    default = LanePipeline(calibration).prepare(image)
    lowered = LanePipeline(calibration, horizon_y=444).prepare(image)

    assert np.allclose(lowered.transform.source_points, MAIN_CAMERA.source_points(444))
    assert not np.allclose(lowered.transform.source_points, default.transform.source_points)


# --------------------------------------------------------------------------- process


@pytest.mark.parametrize("name", ["straight_lines1", "straight_lines2"])
def test_process_measures_a_straight_road(name, calibration):
    """A straight road has a huge radius and the vehicle is near the lane center."""
    result = LanePipeline(calibration).process(read_test_image(name))

    assert result.plausible
    assert result.fit.mode is FitMode.INDEPENDENT
    assert result.measurements.radius > 3000
    assert abs(result.measurements.vehicle_offset) < 0.35
    assert result.measurements.lane_width == pytest.approx(LANE_WIDTH_METERS, abs=0.4)


def test_process_measures_a_bend_to_the_left(calibration):
    """The bend of test2 has a radius of several hundred meters and is to the left."""
    result = LanePipeline(calibration).process(read_test_image("test2"))

    assert result.plausible
    assert result.measurements.curvature < 0
    assert 500 < result.measurements.radius < 1500


def test_process_returns_the_prepared_image(calibration):
    """The result carries what the lane was found in, so that it can be drawn on."""
    pipeline = LanePipeline(calibration)
    image = read_test_image("test1")

    result = pipeline.process(image)
    prepared = pipeline.prepare(image)

    assert result.camera is MAIN_CAMERA
    assert result.transform is prepared.transform
    assert np.array_equal(result.image, prepared.image)
    assert np.array_equal(result.binary, prepared.binary)


def test_process_measures_with_the_scale_of_the_given_horizon(calibration):
    """The conversion to meters uses the horizon the bird's-eye view was built with."""
    result = LanePipeline(calibration, horizon_y=425).process(read_test_image("test1"))

    assert result.plausible
    expected = measure_lane(result.fit, MAIN_CAMERA.size, meters_per_pixel(MAIN_CAMERA, 425))
    assert result.measurements == expected


def test_process_finds_no_lane_in_an_empty_image(calibration):
    """Without lane lines the result is implausible and has no measurements."""
    result = LanePipeline(calibration).process(np.zeros((720, 1280, 3), np.uint8))

    assert not result.plausible
    assert result.measurements is None
    assert result.fit.mode is FitMode.NONE


@pytest.mark.parametrize(
    "video, seconds, horizon, camera",
    [
        ("project_video03.mp4", 20, 422.0, MAIN_CAMERA),
        ("challenge01.mp4", 7, 420.0, MAIN_CAMERA),
        ("project_video02.mp4", 12, 310.2, SECOND_CAMERA),
    ],
)
def test_process_finds_the_lane_in_video_frames(
    video, seconds, horizon, camera, calibration, read_video_frame
):
    """Frames of both cameras give a plausible lane with a realistic position."""
    result = LanePipeline(calibration, horizon).process(read_video_frame(video, seconds))

    assert result.camera is camera
    assert result.plausible
    assert abs(result.measurements.vehicle_offset) < 0.9
    assert result.measurements.lane_width == pytest.approx(LANE_WIDTH_METERS, abs=0.5)
