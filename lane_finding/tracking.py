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

A line that was derived from the other one (it had no pixels) is a guess, not an
observation. The tracker remembers which lines were really seen and compares only those, and
it takes the lane width for deriving a line only from frames in which both lines were well
supported; otherwise a wrong guess would keep confirming itself.
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
# For this many frames without an accepted lane the last lane is still kept (about 1.5 s): it
# guides the search and gives the measurements, but it is not drawn as a found lane.
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
    new: tuple[np.ndarray, np.ndarray],
    old: tuple[np.ndarray, np.ndarray],
    size: tuple[int, int],
    lines: tuple[bool, bool] = (True, True),
) -> bool:
    """Check that two lanes are in about the same place.

    Args:
        new: Coefficients of the left and the right curve of the first lane.
        old: The same for the second lane.
        size: ``(width, height)`` of the bird's-eye image.
        lines: Which of the two lines (left, right) are compared.

    Returns:
        ``True`` if none of the compared lines moved sideways by more than
        ``MAX_FRAME_SHIFT_FRACTION`` of the image width at the compared rows.
    """
    shift = np.abs(_positions(new, size) - _positions(old, size))[list(lines)]
    return bool(np.all(shift <= MAX_FRAME_SHIFT_FRACTION * size[0]))


def _seen_lines(fit: LaneFit) -> tuple[bool, bool]:
    """Tell which lines of a detection rest on pixels of the frame.

    Args:
        fit: A detection.

    Returns:
        For the left and the right line, whether it was fitted to pixels (and not derived
        from the other line).
    """
    return fit.left_pixels[0].size > 0, fit.right_pixels[0].size > 0


def _support(fit: LaneFit) -> tuple[int, bool, int]:
    """Rank a detection by how much of it was observed.

    Args:
        fit: A detection.

    Returns:
        A key that is larger for more seen lines, then for two independently fitted lines,
        then for more pixels.
    """
    pixels = fit.left_pixels[0].size + fit.right_pixels[0].size
    return sum(_seen_lines(fit)), fit.mode is FitMode.INDEPENDENT, pixels


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
        # Whether the left and the right curve were seen in the last accepted frame.
        self._seen = (False, False)
        # The smoothed lane of the frames in which both lines were well supported (fitted
        # independently): its width is used to derive a missing line and to check a new
        # lane. None before such a frame.
        self._measured: tuple[np.ndarray, np.ndarray] | None = None
        # Frames since the measured lane was last updated.
        self._measured_age = 0
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
        """Whether the last lane is still kept although it was not found again."""
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
        self._measured_age += 1
        lane_width = None if self._measured is None else self._measured[1] - self._measured[0]
        candidates = []
        if self._following:
            near = fit_from_pixels(*search_around(binary, self._lane), size, lane_width)
            if near.mode is FitMode.INDEPENDENT and self._agrees(near, size):
                return self._accept(near, size)
            candidates.append(near)
        # The search near the previous lane did not see both lines: a line that is not where
        # the lane expects it (because it was derived wrongly) can only be found from scratch.
        detection = fit_lane(binary, lane_width)
        if self._following:
            candidates.append(detection)
            agreeing = [fit for fit in candidates if self._agrees(fit, size)]
            if agreeing:
                return self._accept(max(agreeing, key=_support), size)
        elif self._confirms(detection, size):
            return self._accept(detection, size, reset=True)
        return self._reject(detection)

    def reproject(self, old: PerspectiveTransform, new: PerspectiveTransform) -> None:
        """Move the remembered lane into a new bird's-eye view, e.g. after the horizon moved.

        Args:
            old: Transform of the view the lane was tracked in.
            new: Transform of the view the next frames come in.
        """
        if self._lane is not None:
            self._lane = tuple(_reproject_curve(curve, old, new) for curve in self._lane)
        if self._measured is not None:
            self._measured = tuple(_reproject_curve(curve, old, new) for curve in self._measured)
        if self._candidate is not None:
            self._candidate = tuple(_reproject_curve(curve, old, new) for curve in self._candidate)

    def _agrees(self, fit: LaneFit, size: tuple[int, int]) -> bool:
        """Check that a detection continues the tracked lane.

        Args:
            fit: Detection in the current frame.
            size: ``(width, height)`` of the bird's-eye image.

        Returns:
            ``True`` if it is a plausible lane whose lines are near the tracked ones and,
            when both of its lines were seen, about as wide as the measured lane. Only the
            lines seen both in the detection and in the tracked lane are compared (all of
            them if there is no such line): a derived line is no evidence of where the line is.
            The width is compared only while it was measured within the last
            ``MAX_MISSED_FRAMES`` frames, like the previous lane guides the search only that
            long: an old width may have been measured in a wrong view of the road.
        """
        if not fit.is_plausible(size):
            return False
        curves = (fit.left, fit.right)
        seen = _seen_lines(fit)
        compared = (seen[0] and self._seen[0], seen[1] and self._seen[1])
        if not any(compared):
            compared = (True, True)
        if not _is_close(curves, self._lane, size, compared):
            return False
        if not all(seen) or self._measured is None or self._measured_age > MAX_MISSED_FRAMES:
            return True  # a derived line has the measured width by construction
        return _has_similar_width(curves, self._measured, size)

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

    def _accept(self, fit: LaneFit, size: tuple[int, int], reset: bool = False) -> TrackedLane:
        """Take a detection into the tracked lane.

        Args:
            fit: The accepted detection.
            size: ``(width, height)`` of the bird's-eye image.
            reset: Start from the detection instead of smoothing it into the tracked lane.

        Returns:
            The smoothed lane, marked as detected.
        """
        curves = (fit.left, fit.right)
        seen = _seen_lines(fit)
        well_supported = fit.mode is FitMode.INDEPENDENT
        if self._lane is None or reset:
            self._lane = curves
        else:
            # A line that was only derived and is now well supported somewhere else replaces
            # the wrong guess at once; everything else is smoothed.
            replaced = [
                well_supported
                and not self._seen[index]
                and not _is_close(curves, self._lane, size, (index == 0, index == 1))
                for index in (0, 1)
            ]
            self._lane = tuple(
                new if replace else SMOOTHING_WEIGHT * new + (1 - SMOOTHING_WEIGHT) * old
                for new, old, replace in zip(curves, self._lane, replaced, strict=True)
            )
        self._seen = seen
        if well_supported:
            self._measured, self._measured_age = self._lane, 0
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
