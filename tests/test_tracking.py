"""Tests for lane_finding.tracking."""

import cv2
import numpy as np
import pytest
from synthetic import curve, draw_lane, lane, max_error

from lane_finding import tracking
from lane_finding.cameras import MAIN_CAMERA
from lane_finding.lane_fit import FitMode, fit_lane
from lane_finding.perspective import PerspectiveTransform
from lane_finding.tracking import (
    CONFIRMING_FRAMES,
    HOLD_FRAMES,
    MAX_MISSED_FRAMES,
    LaneTracker,
    TrackStatus,
)

SIZES = [(1280, 720), (960, 540)]
DETECTED, HELD, LOST = TrackStatus.DETECTED, TrackStatus.HELD, TrackStatus.LOST


def clean(size, shift=0.1, **options):
    """Draw a clean lane with a slight bend.

    Args:
        size: ``(width, height)`` of the bird's-eye image.
        shift: Sideways shift of the lines over the image height (fraction of the width).
        **options: Passed on to :func:`synthetic.draw_lane`.

    Returns:
        The bird's-eye binary image.
    """
    left, right = lane(size, slope=-0.05, shift=shift)
    return draw_lane(size, left, right, **options)


def empty(size):
    """Create a bird's-eye image without any lane pixel.

    Args:
        size: ``(width, height)`` of the bird's-eye image.

    Returns:
        The black image.
    """
    return np.zeros((size[1], size[0]), np.uint8)


def run(tracker, images):
    """Feed images to a tracker one after another.

    Args:
        tracker: The tracker.
        images: Bird's-eye binary images.

    Returns:
        The tracked lane of every image.
    """
    return [tracker.update(image) for image in images]


def center_at_bottom(fit, height):
    """Compute the column of the lane center at the bottom row.

    Args:
        fit: A lane fit with both curves.
        height: Height of the bird's-eye image.

    Returns:
        The column.
    """
    return float(np.polyval(0.5 * (fit.left + fit.right), height - 1))


# ----------------------------------------------------------------- finding the lane


@pytest.mark.parametrize("size", SIZES)
def test_the_first_clear_lane_is_detected_at_once(size):
    """A lane that two good lines show is accepted in the first frame."""
    left, right = lane(size, slope=-0.05, shift=0.1)

    tracked = LaneTracker().update(clean(size))

    assert tracked.status is DETECTED
    assert tracked.fit.mode is FitMode.INDEPENDENT
    assert tracked.fit.left_pixels[0].size > 0
    assert max_error(tracked.fit.left, left, size) < 3
    assert max_error(tracked.fit.right, right, size) < 3


def test_without_lane_lines_the_lane_is_lost():
    """Before any lane was found there is nothing to hold."""
    tracked = LaneTracker().update(empty((1280, 720)))
    assert tracked.status is LOST
    assert tracked.fit.mode is FitMode.NONE


def test_a_weakly_seen_first_lane_has_to_be_confirmed():
    """Only one line is visible: it takes several agreeing frames to trust the lane."""
    size = (1280, 720)
    left, _ = lane(size, slope=-0.05, shift=0.1)
    images = [draw_lane(size, left, None)] * (CONFIRMING_FRAMES + 1)

    statuses = [tracked.status for tracked in run(LaneTracker(), images)]

    assert statuses == [LOST] * (CONFIRMING_FRAMES - 1) + [DETECTED, DETECTED]


def test_a_color_image_is_rejected():
    """Only single-channel images are accepted."""
    with pytest.raises(ValueError, match="single-channel"):
        LaneTracker().update(np.zeros((720, 1280, 3), np.uint8))


# ---------------------------------------------------------------------- smoothing


@pytest.mark.parametrize("size", SIZES)
def test_the_lane_is_smoother_than_the_detections(size):
    """Random jitter of the detections is reduced a lot."""
    rng = np.random.default_rng(1)
    left, right = lane(size, slope=-0.05, shift=0.1)
    images = []
    for jitter in rng.normal(0, 0.006 * size[0], 80):
        shifted = np.array([0.0, 0.0, jitter])
        images.append(draw_lane(size, left + shifted, right + shifted))
    height = size[1]

    tracked = run(LaneTracker(), images)

    assert all(frame.status is DETECTED for frame in tracked)
    raw = [center_at_bottom(fit_lane(image), height) for image in images]
    smooth = [center_at_bottom(frame.fit, height) for frame in tracked]
    assert np.std(smooth[10:]) < 0.5 * np.std(raw[10:])


def test_the_lane_follows_a_lane_that_moves():
    """A steady sideways movement is followed with a small lag and is never rejected."""
    size = (1280, 720)
    left, right = lane(size, slope=-0.05, shift=0.1)
    images = []
    for frame in range(40):
        moved = np.array([0.0, 0.0, 2.0 * frame])
        images.append(draw_lane(size, left + moved, right + moved))

    tracked = run(LaneTracker(), images)

    assert all(frame.status is DETECTED for frame in tracked)
    error = center_at_bottom(tracked[-1].fit, 720) - (center_at_bottom(fit_lane(images[-1]), 720))
    assert -15 < error < 0  # behind the movement, by about 8 pixels


# ------------------------------------------------------------- holding and finding again


@pytest.mark.parametrize("size", SIZES)
def test_the_last_lane_is_held_while_no_lines_are_visible(size):
    """The lane stays as it was for a while, then it is lost."""
    tracker = LaneTracker()
    last = run(tracker, [clean(size)] * 10)[-1]

    after = run(tracker, [empty(size)] * (HOLD_FRAMES + 1))

    assert [frame.status for frame in after] == [HELD] * HOLD_FRAMES + [LOST]
    for frame in after[:HOLD_FRAMES]:
        assert frame.fit.mode is FitMode.TRACKED
        assert frame.fit.left_pixels[0].size == 0
        assert np.array_equal(frame.fit.left, last.fit.left)
        assert np.array_equal(frame.fit.right, last.fit.right)


def test_a_short_gap_is_bridged_and_the_lane_is_found_again_at_once():
    """Lines that come back while the old lane still guides the search need no confirmation."""
    size = (1280, 720)
    images = [clean(size)] * 10 + [empty(size)] * (MAX_MISSED_FRAMES - 1) + [clean(size)] * 2

    statuses = [frame.status for frame in run(LaneTracker(), images)]

    assert statuses[10:] == [HELD] * (MAX_MISSED_FRAMES - 1) + [DETECTED, DETECTED]


def test_after_a_long_gap_the_lane_has_to_be_confirmed():
    """Several frames in a row have to agree before a lane is taken again."""
    size = (1280, 720)
    images = [clean(size)] * 10 + [empty(size)] * MAX_MISSED_FRAMES + [clean(size)] * 5

    statuses = [frame.status for frame in run(LaneTracker(), images)]

    assert statuses[10 + MAX_MISSED_FRAMES :] == [HELD] * (CONFIRMING_FRAMES - 1) + [DETECTED] * (
        5 - CONFIRMING_FRAMES + 1
    )


def test_a_lost_lane_is_found_again_when_it_has_been_confirmed():
    """After the hold has run out a lane is also taken after several agreeing frames."""
    size = (1280, 720)
    images = [clean(size)] * 5 + [empty(size)] * (HOLD_FRAMES + 3) + [clean(size)] * 4

    tracked = run(LaneTracker(), images)

    start = 5 + HOLD_FRAMES + 3
    confirmed = 4 - CONFIRMING_FRAMES + 1
    assert [frame.status for frame in tracked[start:]] == (
        [LOST] * (CONFIRMING_FRAMES - 1) + [DETECTED] * confirmed
    )


def test_a_weak_lane_is_not_confirmed_while_the_last_lane_is_held():
    """During the hold only a lane made of two well supported lines replaces the last one."""
    size = (1280, 720)
    _, right = lane(size, slope=-0.05, shift=0.1)
    only_right = draw_lane(size, None, right)
    images = [clean(size)] * 10 + [empty(size)] * MAX_MISSED_FRAMES + [only_right] * 10

    tracked = run(LaneTracker(), images)

    assert all(frame.status is HELD for frame in tracked[10:])


# ---------------------------------------------------------------- checking the results


def test_a_sudden_jump_of_the_lane_is_rejected():
    """The lane cannot move by an eighth of the image width from one frame to the next."""
    size = (1280, 720)
    left, right = lane(size, slope=-0.05, shift=0.1)
    jumped = np.array([0.0, 0.0, 0.12 * size[0]])
    tracker = LaneTracker()
    before = run(tracker, [clean(size)] * 10)[-1]

    odd = tracker.update(draw_lane(size, left + jumped, right + jumped))
    after = tracker.update(clean(size))

    assert odd.status is HELD
    assert np.array_equal(odd.fit.left, before.fit.left)
    assert after.status is DETECTED
    assert max_error(after.fit.left, left, size) < 3


def test_a_lasting_change_is_accepted_once_it_is_confirmed():
    """If the new position stays, it becomes the lane after the confirmation."""
    size = (1280, 720)
    left, right = lane(size, slope=-0.05, shift=0.1)
    moved = np.array([0.0, 0.0, 0.12 * size[0]])
    new_lane = draw_lane(size, left + moved, right + moved)
    images = [clean(size)] * 10 + [new_lane] * 12

    tracked = run(LaneTracker(), images)

    held = MAX_MISSED_FRAMES + CONFIRMING_FRAMES - 1
    assert [frame.status for frame in tracked[10:]] == [HELD] * held + [DETECTED] * (12 - held)
    assert max_error(tracked[-1].fit.left, left + moved, size) < 3
    assert max_error(tracked[-1].fit.right, right + moved, size) < 3


def test_a_lane_of_another_width_is_rejected(monkeypatch):
    """A plausible lane that is much wider than the tracked one is not a continuation of it."""
    # Without the check of the position only the width can tell the lanes apart.
    monkeypatch.setattr(tracking, "MAX_FRAME_SHIFT_FRACTION", 10.0)
    size = (1280, 720)
    wide = draw_lane(size, curve(size, 0.15, -0.05, 0.1), curve(size, 0.85, -0.05, 0.1))
    tracker = LaneTracker()
    run(tracker, [clean(size)] * 10)

    assert tracker.update(wide).status is HELD


def test_a_missing_line_is_derived_with_the_width_the_lane_had():
    """A lane that is wider than nominal keeps its width when one line disappears."""
    size = (1280, 720)
    left = curve(size, 0.2, -0.05, 0.1)
    right = curve(size, 0.8, -0.05, 0.1)
    tracker = LaneTracker()
    run(tracker, [draw_lane(size, left, right)] * 10)

    tracked = run(tracker, [draw_lane(size, None, right)] * 5)

    assert all(frame.status is DETECTED for frame in tracked)
    assert tracked[-1].fit.mode is FitMode.RESCUED
    assert max_error(tracked[-1].fit.left, left, size) < 6


def test_a_wrongly_derived_line_is_replaced_as_soon_as_both_lines_are_seen():
    """A guessed line does not confirm itself: the real line takes its place at once."""
    size = (1280, 720)
    left = curve(size, 0.2, -0.05, 0.1)
    right = curve(size, 0.8, -0.05, 0.1)  # the lane is wider than the nominal one
    tracker = LaneTracker()
    first = run(tracker, [draw_lane(size, left, None)] * 10)[-1]
    assert first.fit.mode is FitMode.RESCUED
    assert max_error(first.fit.right, right, size) > 0.05 * size[0]  # derived too narrow

    tracked = tracker.update(draw_lane(size, left, right))

    assert tracked.status is DETECTED
    assert tracked.fit.mode is FitMode.INDEPENDENT
    assert max_error(tracked.fit.right, right, size) < 3


def test_a_lane_of_another_width_is_rejected_while_the_width_is_fresh():
    """Right after both lines were seen a line at another width is not taken."""
    size = (1280, 720)
    left = curve(size, 0.2, -0.05, 0.1)
    right = curve(size, 0.8, -0.05, 0.1)
    tracker = LaneTracker()
    run(tracker, [draw_lane(size, left, right)] * 10)
    run(tracker, [draw_lane(size, left, None)] * (MAX_MISSED_FRAMES - 1))

    tracked = tracker.update(draw_lane(size, left, curve(size, 0.9, -0.05, 0.1)))

    assert tracked.fit.mode is FitMode.RESCUED  # the right line is derived with the old width
    assert max_error(tracked.fit.right, right, size) < 6


def test_an_old_width_does_not_keep_out_the_lines_seen_now():
    """A width measured long ago (perhaps in another view) does not veto two seen lines."""
    size = (1280, 720)
    left = curve(size, 0.2, -0.05, 0.1)
    wider_right = curve(size, 0.9, -0.05, 0.1)
    tracker = LaneTracker()
    run(tracker, [draw_lane(size, left, curve(size, 0.8, -0.05, 0.1))] * 10)
    run(tracker, [draw_lane(size, left, None)] * (MAX_MISSED_FRAMES + 1))

    tracked = tracker.update(draw_lane(size, left, wider_right))

    assert tracked.status is DETECTED
    assert max_error(tracked.fit.right, wider_right, size) < 3


# ------------------------------------------------------------------ another bird's-eye view


def test_the_lane_is_moved_into_a_new_view_when_the_horizon_changes():
    """The tracked lane is the same road line, expressed in the view of the new horizon."""
    size = MAIN_CAMERA.size
    old = PerspectiveTransform.for_camera(MAIN_CAMERA, 420)
    new = PerspectiveTransform.for_camera(MAIN_CAMERA, 425)
    left, right = lane(size, slope=-0.05, shift=0.1)
    tracker = LaneTracker()
    run(tracker, [draw_lane(size, left, right)] * 5)
    held_before = tracker.update(empty(size)).fit

    tracker.reproject(old, new)
    held_after = tracker.update(empty(size)).fit

    for before, after in (
        (held_before.left, held_after.left),
        (held_before.right, held_after.right),
    ):
        rows = np.linspace(0, size[1] - 1, 8)
        points = np.float32(np.column_stack([np.polyval(before, rows), rows]))[None]
        expected = cv2.perspectiveTransform(points, new.matrix @ old.inverse_matrix)[0]
        assert np.polyval(after, expected[:, 1]) == pytest.approx(expected[:, 0], abs=1)


def test_reprojecting_an_empty_tracker_does_nothing():
    """Without a lane there is nothing to move."""
    tracker = LaneTracker()
    view = PerspectiveTransform.for_camera(MAIN_CAMERA)
    tracker.reproject(view, view)
    assert tracker.update(empty(MAIN_CAMERA.size)).status is LOST
