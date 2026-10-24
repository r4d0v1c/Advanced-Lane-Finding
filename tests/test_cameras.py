"""Tests for lane_finding.cameras."""

import pytest

from lane_finding.cameras import CAMERAS, MAIN_CAMERA, SECOND_CAMERA, camera_for_size


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_the_region_of_interest_is_a_symmetric_lane_trapezoid(camera):
    """The trapezoid is symmetric, one lane wide and bounded by the far and near row."""
    top_left, top_right, bottom_right, bottom_left = camera.region_of_interest()
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
def test_the_region_of_interest_lies_inside_the_image(camera):
    """All corners of the trapezoid are inside the camera image."""
    width, height = camera.size
    for x, y in camera.region_of_interest():
        assert 0 <= x <= width
        assert 0 <= y <= height


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_lower_horizon_moves_the_far_edge_down_and_narrows_the_near_edge(camera):
    """A lower horizon shifts the far row and shrinks the lane below, the near row stays."""
    default = camera.region_of_interest()
    lower = camera.region_of_interest(horizon_y=camera.horizon_y + 20)
    assert lower[0][1] == pytest.approx(default[0][1] + 20)
    assert lower[2][1] == pytest.approx(default[2][1])
    assert lower[3][0] > default[3][0]
    assert lower[2][0] < default[2][0]


def test_horizon_without_road_below_it_is_rejected():
    """A horizon at the near edge leaves no road to look at."""
    with pytest.raises(ValueError, match="no road"):
        MAIN_CAMERA.region_of_interest(horizon_y=MAIN_CAMERA.bottom_y)


@pytest.mark.parametrize("camera", CAMERAS, ids=lambda camera: camera.name)
def test_lane_widths_grow_from_the_horizon_and_stop_at_the_hood(camera):
    """Every road row has the lane width of the geometry; rows without road have 0."""
    widths = camera.lane_widths()
    horizon = camera.horizon_y + 10
    lowered = camera.lane_widths(horizon)

    assert widths.shape == (camera.size[1],)
    row = int(camera.bottom_y)
    assert widths[row] == pytest.approx(camera.lane_width_per_row * (row - camera.horizon_y))
    assert lowered[row] == pytest.approx(camera.lane_width_per_row * (row - horizon))
    assert not widths[: int(camera.horizon_y) + 1].any()
    assert not widths[row + 1 :].any()


def test_the_view_distance_can_be_changed():
    """The far edge of the view is about 35 m ahead and can be moved."""
    assert MAIN_CAMERA.view_distance == pytest.approx(35.0)
    shorter = MAIN_CAMERA.with_view_distance(20.0)
    assert shorter.view_distance == pytest.approx(20.0)
    assert shorter.horizon_y == MAIN_CAMERA.horizon_y
    assert shorter.bottom_y == MAIN_CAMERA.bottom_y


def test_camera_for_size_returns_the_matching_camera():
    """The camera is recognized by the image resolution."""
    assert camera_for_size(1280, 720) is MAIN_CAMERA
    assert camera_for_size(960, 540) is SECOND_CAMERA


def test_camera_for_size_rejects_unknown_resolution():
    """An unknown resolution raises an error that names the known ones."""
    with pytest.raises(ValueError, match="1280x720, 960x540"):
        camera_for_size(640, 480)
