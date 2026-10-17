"""Tests for lane_finding.cameras."""

import pytest

from lane_finding.cameras import CAMERAS, MAIN_CAMERA, SECOND_CAMERA, camera_for_size


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_source_points_form_a_symmetric_lane_trapezoid(camera):
    """The trapezoid is symmetric, one lane wide and bounded by the far and near row."""
    top_left, top_right, bottom_right, bottom_left = camera.source_points()
    width = camera.size[0]
    assert top_left[0] + top_right[0] == pytest.approx(width)
    assert bottom_left[0] + bottom_right[0] == pytest.approx(width)
    far_row = camera.horizon_y + camera.top_offset
    assert top_left[1] == top_right[1] == pytest.approx(far_row)
    assert bottom_left[1] == bottom_right[1] == pytest.approx(camera.bottom_y)
    for left, right in ((top_left, top_right), (bottom_left, bottom_right)):
        lane_width = camera.lane_width_per_row * (left[1] - camera.horizon_y)
        assert right[0] - left[0] == pytest.approx(lane_width, rel=1e-4)


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_source_points_lie_inside_the_image(camera):
    """All corners of the trapezoid are inside the camera image."""
    width, height = camera.size
    for x, y in camera.source_points():
        assert 0 <= x <= width
        assert 0 <= y <= height


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_lower_horizon_moves_the_far_edge_down_and_narrows_the_near_edge(camera):
    """A lower horizon shifts the far row and shrinks the lane below, the near row stays."""
    default = camera.source_points()
    lower = camera.source_points(horizon_y=camera.horizon_y + 20)
    assert lower[0][1] == pytest.approx(default[0][1] + 20)
    assert lower[2][1] == pytest.approx(default[2][1])
    assert lower[3][0] > default[3][0]
    assert lower[2][0] < default[2][0]


def test_horizon_without_road_below_it_is_rejected():
    """A horizon at the near edge leaves no road to look at."""
    with pytest.raises(ValueError, match="no road"):
        MAIN_CAMERA.source_points(horizon_y=MAIN_CAMERA.bottom_y)


def test_camera_for_size_returns_the_matching_camera():
    """The camera is recognized by the image resolution."""
    assert camera_for_size(1280, 720) is MAIN_CAMERA
    assert camera_for_size(960, 540) is SECOND_CAMERA


def test_camera_for_size_rejects_unknown_resolution():
    """An unknown resolution raises an error that names the known ones."""
    with pytest.raises(ValueError, match="1280x720, 960x540"):
        camera_for_size(640, 480)
