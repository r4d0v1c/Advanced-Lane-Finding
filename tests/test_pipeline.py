"""Tests for lane_finding.pipeline."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_finding.cameras import MAIN_CAMERA, SECOND_CAMERA
from lane_finding.horizon import HorizonEstimator
from lane_finding.lane_fit import FitMode
from lane_finding.measurements import LANE_WIDTH_METERS, measure_lane, meters_per_pixel
from lane_finding.pipeline import (
    DASH_PERIOD_METERS,
    SIDE_MARGIN_METERS,
    LanePipeline,
    view_distance_for_bend,
)
from lane_finding.thresholding import threshold_lane_pixels
from lane_finding.tracking import LaneTracker, TrackStatus

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


# ---------------------------------------------------------------------- lane_pixels


def test_lane_pixels_undistorts_and_selects_the_lane_pixels_in_the_camera_view(calibration):
    """The mask belongs to the undistorted image and has its size."""
    image = read_test_image("test1")

    undistorted, mask = LanePipeline(calibration).lane_pixels(image)

    assert np.array_equal(undistorted, calibration.undistort(image))
    assert np.array_equal(mask, threshold_lane_pixels(undistorted, MAIN_CAMERA.lane_widths()))
    assert mask.shape == image.shape[:2]


def test_lane_pixels_of_an_image_of_the_other_camera_are_not_undistorted(calibration):
    """The calibration does not apply to the second camera."""
    image = read_test_image("solidYellowLeft")
    undistorted, mask = LanePipeline(calibration).lane_pixels(image)
    assert np.array_equal(undistorted, image)
    assert mask.shape == image.shape[:2]


# --------------------------------------------------------------------------- prepare


def test_prepare_undistorts_the_images_of_the_calibrated_camera(calibration):
    """The image of the main camera is undistorted and its binary image comes from that."""
    image = read_test_image("test1")

    prepared = LanePipeline(calibration).prepare(image)

    undistorted = calibration.undistort(image)
    assert prepared.camera is MAIN_CAMERA
    assert np.array_equal(prepared.image, undistorted)
    assert not np.array_equal(prepared.image, image)
    mask = threshold_lane_pixels(undistorted, MAIN_CAMERA.lane_widths())
    expected = prepared.transform.warp(mask, cv2.INTER_NEAREST)
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

    assert np.allclose(lowered.transform.source_points, MAIN_CAMERA.region_of_interest(444))
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


@pytest.mark.xfail(
    strict=True,
    reason="a light pavement seam next to the dashed right line stands out like a marking; "
    "in a single image it pulls the radius down to about 450 m",
)
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
    assert result.status is TrackStatus.LOST


def test_process_reports_that_the_lane_was_found(calibration):
    """Without a tracker an image shows the lane or it does not."""
    result = LanePipeline(calibration).process(read_test_image("straight_lines1"))
    assert result.status is TrackStatus.DETECTED


# ------------------------------------------------------------------------- tracking


def test_a_tracker_follows_the_frames_of_a_video(calibration, read_video_frames):
    """The frames of a clear video section are all found and give measurements."""
    frames = read_video_frames("project_video03.mp4", 20, 8)
    pipeline = LanePipeline(calibration, 422.0, LaneTracker())

    results = [pipeline.process(frame) for frame in frames]

    assert all(result.status is TrackStatus.DETECTED for result in results)
    assert all(result.plausible for result in results)
    assert results[-1].fit.mode is not FitMode.TRACKED
    assert abs(results[-1].measurements.vehicle_offset) < 0.9


def test_a_tracker_holds_the_last_lane_while_the_lines_are_not_visible(
    calibration, read_video_frames
):
    """A black frame between road frames keeps the lane and its measurements."""
    frames = read_video_frames("project_video03.mp4", 20, 8)
    pipeline = LanePipeline(calibration, 422.0, LaneTracker())
    results = [pipeline.process(frame) for frame in frames]

    held = pipeline.process(np.zeros_like(frames[0]))
    again = pipeline.process(read_video_frames("project_video03.mp4", 20.4, 1)[0])

    assert held.status is TrackStatus.HELD
    assert held.fit.mode is FitMode.TRACKED
    assert held.measurements == results[-1].measurements
    assert again.status is TrackStatus.DETECTED


def test_without_a_tracker_a_black_frame_loses_the_lane(calibration, read_video_frames):
    """Only the tracker holds the lane."""
    frames = read_video_frames("project_video03.mp4", 20, 3)
    pipeline = LanePipeline(calibration, 422.0)
    for frame in frames:
        pipeline.process(frame)

    result = pipeline.process(np.zeros_like(frames[0]))

    assert result.status is TrackStatus.LOST
    assert result.measurements is None


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


def test_the_horizon_follows_the_frames_of_a_video(calibration, read_video_frames):
    """Starting from the usual horizon, the pipeline adapts it from the frames seen so far."""
    frames = read_video_frames("challenge02.mp4", 7, 150)
    pipeline = LanePipeline(
        calibration, tracker=LaneTracker(), horizon_estimator=HorizonEstimator(sample_every=3)
    )

    first = pipeline.process(frames[0])
    results = [pipeline.process(frame) for frame in frames[1:]]

    assert first.transform.source_points[0][1] == pytest.approx(MAIN_CAMERA.horizon_y + 40)
    assert pipeline.horizon_y == pytest.approx(444, abs=8)
    assert results[-1].transform.source_points[0][1] == pytest.approx(pipeline.horizon_y + 40)
    assert sum(result.status is TrackStatus.DETECTED for result in results[-50:]) >= 45


# ---------------------------------------------------------------------- view in bends


def test_beside_the_lane_there_is_half_a_lane_of_road():
    """The lane is the middle half of the view, so half a lane is left on each side."""
    assert SIDE_MARGIN_METERS == pytest.approx(LANE_WIDTH_METERS / 2)


def test_the_view_reaches_as_far_as_the_lane_stays_in_it():
    """On a circle the lane moves sideways by d**2 / (2 R): it leaves the view at that d."""
    camera = MAIN_CAMERA
    near = camera.focal_height / (camera.bottom_y - camera.horizon_y)

    assert view_distance_for_bend(camera, None, np.inf) == pytest.approx(camera.view_distance)
    assert view_distance_for_bend(camera, None, 1000.0) == pytest.approx(camera.view_distance)
    assert view_distance_for_bend(camera, None, 150.0) == pytest.approx(
        np.sqrt(2 * 150.0 * SIDE_MARGIN_METERS)
    )
    # never shorter than one period of a dashed line beyond the near edge
    assert view_distance_for_bend(camera, None, 20.0) == pytest.approx(near + DASH_PERIOD_METERS)


def test_without_a_tracker_the_view_does_not_change(calibration):
    """A single image has no earlier frame that could tell how sharp the bend is."""
    pipeline = LanePipeline(calibration)
    for name in ("test2", "test3"):
        assert pipeline.process(read_test_image(name)).camera.view_distance == pytest.approx(35.0)


def test_the_view_stays_long_on_a_highway(calibration, read_video_frames):
    """The bends of a highway are gentle enough for the whole view."""
    pipeline = LanePipeline(calibration, 422.0, LaneTracker())
    for frame in read_video_frames("project_video03.mp4", 20, 30):
        result = pipeline.process(frame)
    assert result.camera.view_distance == pytest.approx(35.0)


def test_the_view_gets_shorter_in_a_sharp_bend(calibration, read_video_frames):
    """In the bends of a mountain road the lane leaves a long view sideways."""
    pipeline = LanePipeline(calibration, 418.0, LaneTracker())
    for frame in read_video_frames("challenge03.mp4", 32, 30):
        result = pipeline.process(frame)
    near = MAIN_CAMERA.focal_height / (MAIN_CAMERA.bottom_y - 418.0)
    assert near + DASH_PERIOD_METERS <= result.camera.view_distance < 25.0
    assert result.transform.source_points[0][1] == pytest.approx(
        418.0 + MAIN_CAMERA.focal_height / result.camera.view_distance
    )
