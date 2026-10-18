"""The complete lane finding pipeline for a road image or the frames of a video.

The steps are:

1. remove the lens distortion (only for the camera the calibration was made for),
2. select the lane-line pixels by color (yellow and white),
3. warp the binary image to the bird's-eye view,
4. find the lane lines and fit a polynomial to each of them,
5. measure the curvature of the lane and the position of the vehicle in meters.

The camera is recognized by the resolution of the image, because each camera has its own
road geometry (see :mod:`lane_finding.cameras`). For the frames of a video a tracker can be
given, which follows the lane from frame to frame in the fourth step, and a horizon
estimator, which adapts the bird's-eye view to the horizon of the video from the frames seen
so far.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from lane_finding.calibration import CameraCalibration
from lane_finding.cameras import CameraProfile, camera_for_size
from lane_finding.horizon import HorizonEstimator
from lane_finding.lane_fit import LaneFit, fit_lane
from lane_finding.measurements import LaneMeasurements, measure_lane, meters_per_pixel
from lane_finding.perspective import PerspectiveTransform
from lane_finding.thresholding import threshold_lane_pixels
from lane_finding.tracking import LaneTracker, TrackStatus

# The bird's-eye view is rebuilt when the estimated horizon moved by at least this many rows.
HORIZON_UPDATE_ROWS = 1.0


@dataclass(frozen=True, eq=False)
class PreparedImage:
    """A road image prepared for the search of the lane lines.

    Attributes:
        image: The camera image with the lens distortion removed (unchanged for a camera
            without calibration).
        binary: Bird's-eye binary image of the lane-line pixels (values 0 and 255).
        transform: Transform between the camera view and the bird's-eye view.
        camera: Profile of the camera that took the image.
    """

    image: np.ndarray
    binary: np.ndarray
    transform: PerspectiveTransform
    camera: CameraProfile


@dataclass(frozen=True, eq=False)
class LaneResult(PreparedImage):
    """Everything the pipeline found in a road image.

    Attributes:
        fit: The lane lines in the bird's-eye view. With a tracker these are the smoothed
            lines, or the held ones when the lane was not found in this frame.
        measurements: Curvature, vehicle position and lane width in meters; ``None`` if
            there is no plausible lane.
        status: Whether the lane was found in this image, is held from earlier frames of a
            video or is lost. Without a tracker it is found or lost.
    """

    fit: LaneFit
    measurements: LaneMeasurements | None
    status: TrackStatus

    @property
    def plausible(self) -> bool:
        """Whether there is a plausible lane (found in this image or held)."""
        return self.measurements is not None


class LanePipeline:
    """Runs all steps from a camera image to the measured lane.

    The perspective transform of each camera is created once and reused, so processing
    many images only costs the processing itself.
    """

    def __init__(
        self,
        calibration: CameraCalibration | None = None,
        horizon_y: float | None = None,
        tracker: LaneTracker | None = None,
        horizon_estimator: HorizonEstimator | None = None,
    ) -> None:
        """Create a pipeline.

        Args:
            calibration: Calibration of the main camera, used to remove the lens distortion
                of its images. Without it no image is undistorted.
            horizon_y: Row of the horizon to build the bird's-eye view with, instead of the
                default one of each camera (it differs a little between videos).
            tracker: Follows the lane through the frames of a video. Without it every image
                is processed on its own. A tracker belongs to one video.
            horizon_estimator: Estimates the horizon of a video from the frames processed so
                far; ``horizon_y`` then changes as the estimate changes. It belongs to one
                video.
        """
        self.calibration = calibration
        self.horizon_y = horizon_y
        self.tracker = tracker
        self.horizon_estimator = horizon_estimator
        self._transforms: dict[tuple[int, int], tuple[float | None, PerspectiveTransform]] = {}

    def _transform_for(self, camera: CameraProfile) -> PerspectiveTransform:
        """Get the perspective transform of a camera for the current horizon.

        Args:
            camera: Profile of the camera.

        Returns:
            The transform for images of the camera's resolution, created again only when the
            horizon changed.
        """
        cached = self._transforms.get(camera.size)
        if cached is None or cached[0] != self.horizon_y:
            transform = PerspectiveTransform.for_camera(camera, self.horizon_y)
            self._transforms[camera.size] = (self.horizon_y, transform)
            return transform
        return cached[1]

    def _follow_horizon(self, lane_mask: np.ndarray, camera: CameraProfile) -> None:
        """Update the horizon from the estimator and move the tracked lane into the new view.

        Args:
            lane_mask: Binary lane-pixel image of the current frame in the camera view.
            camera: Profile of the camera.
        """
        estimate = self.horizon_estimator.update(lane_mask, camera)
        current = camera.horizon_y if self.horizon_y is None else self.horizon_y
        if estimate is None or abs(estimate - current) < HORIZON_UPDATE_ROWS:
            return
        old = self._transform_for(camera)
        self.horizon_y = estimate
        if self.tracker is not None:
            self.tracker.reproject(old, self._transform_for(camera))

    def lane_pixels(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove the distortion and select the lane-line pixels in the camera view.

        Args:
            image: 8-bit BGR road image.

        Returns:
            The undistorted image and the binary image of its lane-line pixels (0 and 255).

        Raises:
            ValueError: If the image is not an 8-bit BGR image.
        """
        height, width = image.shape[:2]
        if self.calibration is not None and (width, height) == self.calibration.image_size:
            image = self.calibration.undistort(image)
        return image, threshold_lane_pixels(image)

    def prepare(self, image: np.ndarray) -> PreparedImage:
        """Remove the distortion and create the bird's-eye binary image.

        Args:
            image: 8-bit BGR road image of a known camera.

        Returns:
            The undistorted image and its bird's-eye binary image.

        Raises:
            ValueError: If the resolution of the image belongs to no known camera or the
                image is not an 8-bit BGR image.
        """
        height, width = image.shape[:2]
        camera = camera_for_size(width, height)
        undistorted, lane_mask = self.lane_pixels(image)
        if self.horizon_estimator is not None:
            self._follow_horizon(lane_mask, camera)
        transform = self._transform_for(camera)
        binary = transform.warp(lane_mask, cv2.INTER_NEAREST)
        return PreparedImage(undistorted, binary, transform, camera)

    def process(self, image: np.ndarray) -> LaneResult:
        """Find the lane in a road image and measure it.

        Args:
            image: 8-bit BGR road image of a known camera.

        Returns:
            The prepared image, the lane lines and, if there is a plausible lane, its
            measurements.

        Raises:
            ValueError: If the resolution of the image belongs to no known camera or the
                image is not an 8-bit BGR image.
        """
        prepared = self.prepare(image)
        size = prepared.camera.size
        if self.tracker is None:
            fit = fit_lane(prepared.binary)
            status = TrackStatus.DETECTED if fit.is_plausible(size) else TrackStatus.LOST
        else:
            tracked = self.tracker.update(prepared.binary)
            fit, status = tracked.fit, tracked.status
        measurements = None
        if status is not TrackStatus.LOST:
            measurements = measure_lane(
                fit, size, meters_per_pixel(prepared.camera, self.horizon_y)
            )
        return LaneResult(
            prepared.image,
            prepared.binary,
            prepared.transform,
            prepared.camera,
            fit,
            measurements,
            status,
        )
