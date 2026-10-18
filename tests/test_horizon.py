"""Tests for lane_finding.horizon."""

import cv2
import numpy as np
import pytest

from lane_finding.cameras import CAMERAS, MAIN_CAMERA, SECOND_CAMERA, CameraProfile
from lane_finding.horizon import (
    MIN_FRAMES,
    HorizonEstimator,
    estimate_horizon,
    vanishing_point_row,
)
from lane_finding.pipeline import LanePipeline

CAMERA_IDS = [camera.name for camera in CAMERAS]


def converging_lines(
    camera: CameraProfile, vanishing_row: float, left: bool = True, right: bool = True
) -> np.ndarray:
    """Draw the lane lines of a lane whose lines meet at a given row, as a binary image.

    Args:
        camera: Camera that gives the image size and the near edge of the road.
        vanishing_row: Row of the vanishing point, which lies in the middle column.
        left: Draw the left line.
        right: Draw the right line.

    Returns:
        Binary camera-view image (0 and 255) of the lines.
    """
    width, height = camera.size
    image = np.zeros((height, width), np.uint8)
    bottom = camera.bottom_y - 5
    top = vanishing_row + 0.4 * (bottom - vanishing_row)
    thickness = max(3, round(0.008 * width))
    for foot, drawn in ((0.2 * width, left), (0.8 * width, right)):
        if drawn:
            # The line runs from its foot on the near edge to the vanishing point.
            top_column = width / 2 + (foot - width / 2) * (top - vanishing_row) / (
                bottom - vanishing_row
            )
            start, end = (round(foot), round(bottom)), (round(top_column), round(top))
            cv2.line(image, start, end, 255, thickness)
    return image


def crossing_segments(image: np.ndarray, rows: tuple[int, int], meeting_row: float) -> None:
    """Draw several left- and right-leaning segments that cross around a given row.

    Args:
        image: Binary image to draw on.
        rows: First and last row of the segments.
        meeting_row: Row where the segments cross.
    """
    center = image.shape[1] / 2
    for offset in (0, 6, 12):
        for sign in (-1, 1):
            # x = center - 2 * sign * (y - meeting_row) + sign * offset: the two families cross
            # at rows between meeting_row and meeting_row + 18.
            start = (round(center - 2 * sign * (rows[0] - meeting_row) + sign * offset), rows[0])
            end = (round(center - 2 * sign * (rows[1] - meeting_row) + sign * offset), rows[1])
            cv2.line(image, start, end, 255, 5)


# ------------------------------------------------------------- vanishing_point_row


@pytest.mark.parametrize("camera", CAMERAS, ids=CAMERA_IDS)
@pytest.mark.parametrize("offset", [-12, 0, 12])
def test_the_vanishing_point_row_is_where_the_lines_meet(camera, offset):
    """Two straight lines that meet at a row give that row (Hough angles are 1 degree apart)."""
    vanishing_row = camera.horizon_y + offset

    row = vanishing_point_row(converging_lines(camera, vanishing_row), camera)

    assert row == pytest.approx(vanishing_row, abs=4)


def test_an_image_without_lines_has_no_vanishing_point():
    """Nothing to intersect."""
    assert vanishing_point_row(np.zeros((720, 1280), np.uint8), MAIN_CAMERA) is None


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_single_line_has_no_vanishing_point(side):
    """One line is no intersection."""
    image = converging_lines(MAIN_CAMERA, 420, left=side == "left", right=side == "right")
    assert vanishing_point_row(image, MAIN_CAMERA) is None


def test_the_car_hood_is_ignored():
    """Lines on the hood below the road do not move the vanishing point."""
    image = converging_lines(MAIN_CAMERA, 420)
    crossing_segments(image, (int(MAIN_CAMERA.bottom_y) + 2, 719), 700)

    assert vanishing_point_row(image, MAIN_CAMERA) == pytest.approx(420, abs=4)


def test_the_sky_is_ignored():
    """Lines above the horizon do not move the vanishing point."""
    image = converging_lines(MAIN_CAMERA, 420)
    crossing_segments(image, (150, int(MAIN_CAMERA.horizon_y) - 5), 250)

    assert vanishing_point_row(image, MAIN_CAMERA) == pytest.approx(420, abs=4)


def test_the_input_image_is_not_modified():
    """The parts above the horizon and below the road are removed from a copy."""
    image = converging_lines(MAIN_CAMERA, 420)
    original = image.copy()
    vanishing_point_row(image, MAIN_CAMERA)
    assert np.array_equal(image, original)


# ------------------------------------------------------------------ estimate_horizon


@pytest.mark.parametrize("camera", CAMERAS, ids=CAMERA_IDS)
def test_the_horizon_of_a_video_is_the_median_of_its_frames(camera):
    """A few frames with a different row do not matter."""
    rows = [camera.horizon_y + offset for offset in (20, 21, 19, 20, 22, 45, 45)]
    frames = [converging_lines(camera, row) for row in rows]

    assert estimate_horizon(frames, camera) == pytest.approx(camera.horizon_y + 21, abs=4)


def test_too_few_frames_give_the_usual_horizon_of_the_camera():
    """With little evidence the default is kept."""
    frames = [converging_lines(MAIN_CAMERA, 440)] * (MIN_FRAMES - 1)
    frames += [np.zeros((720, 1280), np.uint8)] * 10
    assert estimate_horizon(frames, MAIN_CAMERA) == MAIN_CAMERA.horizon_y


def test_an_unbelievable_horizon_is_ignored():
    """A vanishing point far from where it can be is not used."""
    frames = [converging_lines(MAIN_CAMERA, MAIN_CAMERA.horizon_y - 90)] * 10
    assert estimate_horizon(frames, MAIN_CAMERA) == MAIN_CAMERA.horizon_y


def test_no_frames_give_the_usual_horizon_of_the_camera():
    """An empty video."""
    assert estimate_horizon([], SECOND_CAMERA) == SECOND_CAMERA.horizon_y


# ----------------------------------------------------------------- HorizonEstimator


def test_the_running_estimate_waits_for_enough_frames():
    """Until enough frames gave a row, there is no estimate."""
    estimator = HorizonEstimator(sample_every=1)
    frame = converging_lines(MAIN_CAMERA, 440)

    estimates = [estimator.update(frame, MAIN_CAMERA) for _ in range(MIN_FRAMES)]

    assert estimates[:-1] == [None] * (MIN_FRAMES - 1)
    assert estimates[-1] == pytest.approx(440, abs=4)


def test_the_running_estimate_uses_only_every_nth_frame():
    """Frames between the samples do not count."""
    estimator = HorizonEstimator(sample_every=3)
    frame = converging_lines(MAIN_CAMERA, 440)

    estimates = [estimator.update(frame, MAIN_CAMERA) for _ in range(3 * MIN_FRAMES)]

    assert estimates[3 * (MIN_FRAMES - 1) - 1] is None
    assert estimates[3 * (MIN_FRAMES - 1)] == pytest.approx(440, abs=4)


def test_the_running_estimate_follows_only_the_past():
    """The estimate is the median of the frames so far and moves when the road changes."""
    estimator = HorizonEstimator(sample_every=1)
    for _ in range(5):
        early = estimator.update(converging_lines(MAIN_CAMERA, 430), MAIN_CAMERA)
    for _ in range(7):
        later = estimator.update(converging_lines(MAIN_CAMERA, 450), MAIN_CAMERA)

    assert early == pytest.approx(
        vanishing_point_row(converging_lines(MAIN_CAMERA, 430), MAIN_CAMERA)
    )
    assert later == pytest.approx(
        vanishing_point_row(converging_lines(MAIN_CAMERA, 450), MAIN_CAMERA)
    )


# -------------------------------------------------------------------- real videos


# The horizon of each video was measured earlier (see cameras.py): it differs from video to video.
@pytest.mark.parametrize(
    "video, seconds, expected, camera",
    [
        ("project_video03.mp4", range(2, 40, 3), 422, MAIN_CAMERA),
        ("challenge02.mp4", np.arange(0.5, 16, 0.5), 444, MAIN_CAMERA),
        ("project_video02.mp4", range(2, 26, 2), 310, SECOND_CAMERA),
    ],
)
def test_the_horizon_of_real_videos(
    video, seconds, expected, camera, calibration, read_video_frame
):
    """Frames spread over a video give the horizon of that video."""
    pipeline = LanePipeline(calibration)
    masks = (pipeline.lane_pixels(read_video_frame(video, second))[1] for second in seconds)

    assert estimate_horizon(masks, camera) == pytest.approx(expected, abs=6)
