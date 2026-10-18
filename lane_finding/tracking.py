"""Following the lane through the frames of a video.

A single frame can show the lane badly: the lines may be hidden by the shadow of a bridge,
outshone by sunlit asphalt or just dashed. The tracker therefore remembers the lane of the
previous frames and uses it in five ways:

* the lines are searched near their previous position instead of from scratch,
* a result is accepted only if it is a lane that is near the previous one and about as
  wide (the lane cannot jump sideways from one frame to the next),
* the accepted lanes are smoothed over the frames,
* while no lane is found the last one is held for a while,
* after a longer loss a new lane is accepted only when several frames in a row agree on it.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from lane_finding.lane_fit import (
    NO_PIXELS,
    FitMode,
    LaneFit,
    fit_from_pixels,
    fit_lane,
    search_around,
)
from lane_finding.perspective import PerspectiveTransform

# Weight of the newest frame in the smoothed lane (exponential moving average): the lane
# lags about (1 - weight) / weight = 4 frames behind a change.
SMOOTHING_WEIGHT = 0.2
# The largest sideways move of a lane line between two frames (fraction of the image width).
# A lane change moves the lane by about 0.1 % of the width per frame.
MAX_FRAME_SHIFT_FRACTION = 0.035
# The largest relative change of the lane width between two frames.
MAX_WIDTH_CHANGE = 0.12
# The rows (fractions of the image height) at which two lanes are compared.
COMPARE_ROW_FRACTIONS = (0.95, 0.5)
# After this many frames in a row without an accepted lane the previous lane no longer guides
# the search and a new lane has to be confirmed.
MAX_MISSED_FRAMES = 5
# For this many frames without an accepted lane the last lane is still shown (about 1.5 s).
HOLD_FRAMES = 45
# A new lane is confirmed when this many frames in a row agree on it.
CONFIRMING_FRAMES = 3


class TrackStatus(Enum):
    """What the tracker knows about the lane in a frame."""

    DETECTED = "detected"
    HELD = "held"
    LOST = "lost"


@dataclass(frozen=True, eq=False)
class TrackedLane:
    """The lane in one frame of a video.

    Attributes:
        fit: The smoothed lane lines in the bird's-eye view. When the lane was found in this
            frame it carries the pixels of the detection; when it is held it has the mode
            ``FitMode.TRACKED`` and no pixels. When the lane is lost it is the detection
            of this frame, which is not a plausible lane.
        status: Whether the lane was found in this frame, is held from earlier frames or
            is lost.
    """

    fit: LaneFit
    status: TrackStatus


def _positions(curves: tuple[np.ndarray, np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """Compute the columns of two lane curves at the rows where lanes are compared.

    Args:
        curves: Coefficients ``(A, B, C)`` of the left and the right curve.
        size: ``(width, height)`` of the bird's-eye image.

    Returns:
        Array of shape ``(2, len(COMPARE_ROW_FRACTIONS))`` with the columns of the left and
        of the right curve.
    """
    rows = np.array(COMPARE_ROW_FRACTIONS) * size[1]
    return np.array([np.polyval(curve, rows) for curve in curves])


def _is_close(
    new: tuple[np.ndarray, np.ndarray], old: tuple[np.ndarray, np.ndarray], size: tuple[int, int]
) -> bool:
    """Check that two lanes are in about the same place.

    Args:
        new: Coefficients of the left and the right curve of the first lane.
        old: The same for the second lane.
        size: ``(width, height)`` of the bird's-eye image.

    Returns:
        ``True`` if no lane line moved sideways by more than ``MAX_FRAME_SHIFT_FRACTION`` of
        the image width at the compared rows.
    """
    shift = np.abs(_positions(new, size) - _positions(old, size)).max()
    return bool(shift <= MAX_FRAME_SHIFT_FRACTION * size[0])


def _has_similar_width(
    new: tuple[np.ndarray, np.ndarray], old: tuple[np.ndarray, np.ndarray], size: tuple[int, int]
) -> bool:
    """Check that two lanes are about equally wide.

    Args:
        new: Coefficients of the left and the right curve of the first lane.
        old: The same for the second lane.
        size: ``(width, height)`` of the bird's-eye image.

    Returns:
        ``True`` if the width differs by less than ``MAX_WIDTH_CHANGE`` at the compared rows.
    """
    new_left, new_right = _positions(new, size)
    old_left, old_right = _positions(old, size)
    change = (new_right - new_left) / (old_right - old_left)
    return bool(np.all(np.abs(change - 1) < MAX_WIDTH_CHANGE))


def _reproject_curve(
    coefficients: np.ndarray, old: PerspectiveTransform, new: PerspectiveTransform
) -> np.ndarray:
    """Express a curve of one bird's-eye view in another view of the same camera.

    Args:
        coefficients: ``(A, B, C)`` of the curve in the old view.
        old: Transform of the old view.
        new: Transform of the new view.

    Returns:
        The coefficients of the same road line in the new view (fitted to 20 points).
    """
    rows = np.linspace(0, old.size[1] - 1, 20)
    points = np.column_stack([np.polyval(coefficients, rows), rows]).astype(np.float32)
    moved = cv2.perspectiveTransform(points[None], new.matrix @ old.inverse_matrix)[0]
    return np.polyfit(moved[:, 1], moved[:, 0], 2)


class LaneTracker:
    """Follows the lane of one video from frame to frame.

    The tracker is in one of two situations. While it follows the lane, every frame is
    searched near the previous lane and a result is accepted if it is plausible, near the
    previous lane and about as wide. After ``MAX_MISSED_FRAMES`` frames without an accepted
    lane the tracker gives the previous lane up as a guide: every frame is searched from
    scratch and a lane is accepted only when ``CONFIRMING_FRAMES`` frames in a row agree on
    it. While the last lane is still held, such a lane must also be made of two well
    supported lines, because a weak result is more likely to be a mistake.
    """

    def __init__(self) -> None:
        """Create a tracker that has not seen any frame."""
        # The smoothed left and right curve; None before the first lane was accepted.
        self._lane: tuple[np.ndarray, np.ndarray] | None = None
        # Frames in a row without an accepted lane.
        self._missed = 0
        # The latest candidate for a new lane and the number of frames that agree on it.
        self._candidate: tuple[np.ndarray, np.ndarray] | None = None
        self._agreeing = 0

    @property
    def _following(self) -> bool:
        """Whether the previous lane still guides the search."""
        return self._lane is not None and self._missed < MAX_MISSED_FRAMES

    @property
    def _holding(self) -> bool:
        """Whether the last lane is still shown although it was not found again."""
        return self._lane is not None and self._missed < HOLD_FRAMES

    def update(self, binary: np.ndarray) -> TrackedLane:
        """Find the lane in the next frame and update the tracked lane.

        Args:
            binary: Bird's-eye binary image of the frame (lane pixels are non-zero).

        Returns:
            The smoothed lane and whether it was found in this frame, is held from earlier
            frames or is lost.

        Raises:
            ValueError: If the input is not a single-channel image.
        """
        size = (binary.shape[1], binary.shape[0])
        lane_width = None if self._lane is None else self._lane[1] - self._lane[0]
        if self._following:
            detection = fit_from_pixels(*search_around(binary, self._lane), size, lane_width)
            if self._agrees(detection, size):
                return self._accept(detection)
        detection = fit_lane(binary, lane_width)
        if self._following:
            if self._agrees(detection, size):
                return self._accept(detection)
        elif self._confirms(detection, size):
            return self._accept(detection, reset=True)
        return self._reject(detection)

    def reproject(self, old: PerspectiveTransform, new: PerspectiveTransform) -> None:
        """Move the remembered lane into a new bird's-eye view, e.g. after the horizon moved.

        Args:
            old: Transform of the view the lane was tracked in.
            new: Transform of the view the next frames come in.
        """
        if self._lane is not None:
            self._lane = tuple(_reproject_curve(curve, old, new) for curve in self._lane)
        if self._candidate is not None:
            self._candidate = tuple(_reproject_curve(curve, old, new) for curve in self._candidate)

    def _agrees(self, fit: LaneFit, size: tuple[int, int]) -> bool:
        """Check that a detection continues the tracked lane.

        Args:
            fit: Detection in the current frame.
            size: ``(width, height)`` of the bird's-eye image.

        Returns:
            ``True`` if it is a plausible lane that is near the tracked lane and about as
            wide.
        """
        if not fit.is_plausible(size):
            return False
        curves = (fit.left, fit.right)
        return _is_close(curves, self._lane, size) and _has_similar_width(curves, self._lane, size)

    def _confirms(self, fit: LaneFit, size: tuple[int, int]) -> bool:
        """Check whether a detection completes a run of frames that agree on a new lane.

        The first lane of a video is confirmed by one frame if both of its lines are well
        supported.

        Args:
            fit: Detection in the current frame.
            size: ``(width, height)`` of the bird's-eye image.

        Returns:
            ``True`` if enough frames in a row, this one included, show about the same
            plausible lane.
        """
        trusted = fit.is_plausible(size) and (not self._holding or fit.mode is FitMode.INDEPENDENT)
        if not trusted:
            self._candidate, self._agreeing = None, 0
            return False
        curves = (fit.left, fit.right)
        if self._candidate is not None and _is_close(curves, self._candidate, size):
            self._agreeing += 1
        else:
            self._agreeing = 1
        self._candidate = curves
        first_lane = self._lane is None and fit.mode is FitMode.INDEPENDENT
        return self._agreeing >= (1 if first_lane else CONFIRMING_FRAMES)

    def _accept(self, fit: LaneFit, reset: bool = False) -> TrackedLane:
        """Take a detection into the tracked lane.

        Args:
            fit: The accepted detection.
            reset: Start from the detection instead of smoothing it into the tracked lane.

        Returns:
            The smoothed lane, marked as detected.
        """
        curves = (fit.left, fit.right)
        if self._lane is None or reset:
            self._lane = curves
        else:
            self._lane = tuple(
                SMOOTHING_WEIGHT * new + (1 - SMOOTHING_WEIGHT) * old
                for new, old in zip(curves, self._lane, strict=True)
            )
        self._missed = 0
        self._candidate, self._agreeing = None, 0
        smoothed = LaneFit(*self._lane, fit.left_pixels, fit.right_pixels, fit.mode)
        return TrackedLane(smoothed, TrackStatus.DETECTED)

    def _reject(self, detection: LaneFit) -> TrackedLane:
        """Handle a frame in which no acceptable lane was found.

        Args:
            detection: The detection in the current frame.

        Returns:
            The held lane if it is still held, otherwise the detection marked as lost.
        """
        if self._lane is not None:
            self._missed += 1
            if self._missed <= HOLD_FRAMES:
                held = LaneFit(*self._lane, NO_PIXELS, NO_PIXELS, FitMode.TRACKED)
                return TrackedLane(held, TrackStatus.HELD)
        return TrackedLane(detection, TrackStatus.LOST)
