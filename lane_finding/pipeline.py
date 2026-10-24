"""The complete lane finding pipeline for a road image or the frames of a video.

The steps are:

1. remove the lens distortion (only for the camera the calibration was made for),
2. select the lane-line pixels (lighter or yellower than the road beside them),
3. warp the binary image to the bird's-eye view,
4. find the lane lines and fit a polynomial to each of them,
5. measure the curvature of the lane and the position of the vehicle in meters.

The camera is recognized by the resolution of the image, because each camera has its own
road geometry (see :mod:`lane_finding.cameras`). For the frames of a video a tracker can be
given, which follows the lane from frame to frame in the fourth step, and a horizon
estimator, which adapts the bird's-eye view to the horizon of the video from the frames seen
so far. With a tracker the view also reaches less far ahead in sharp bends, where the lane
would otherwise leave it sideways.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from lane_finding.calibration import CameraCalibration
from lane_finding.cameras import LANE_WIDTH_METERS, CameraProfile, camera_for_size
from lane_finding.horizon import HorizonEstimator
from lane_finding.lane_fit import LaneFit, fit_lane
from lane_finding.measurements import LaneMeasurements, measure_lane, meters_per_pixel
from lane_finding.perspective import LANE_WIDTH_FRACTION, PerspectiveTransform
from lane_finding.thresholding import threshold_lane_pixels
from lane_finding.tracking import LaneTracker, TrackStatus

# The bird's-eye view is rebuilt when its horizon or its far edge moved by at least this many
# rows.
VIEW_UPDATE_ROWS = 1.0
# Road standard: a dashed lane line repeats every 14.63 m (12 ft dash and 36 ft gap). The
# view has to show at least one whole period, or a dashed line may have no dash in it.
DASH_PERIOD_METERS = 14.63
# The lane fills the middle of the bird's-eye view; beside each line there is this much road
# (half a lane when the lane is half of the view).
SIDE_MARGIN_METERS = (1 - LANE_WIDTH_FRACTION) / 2 / LANE_WIDTH_FRACTION * LANE_WIDTH_METERS


def view_distance_for_bend(camera: CameraProfile, horizon_y: float | None, radius: float) -> float:
    """Compute how far ahead the bird's-eye view should reach in a bend.

    On a circle of radius ``R`` the road moves sideways by about ``d**2 / (2 R)`` within the
    distance ``d``, so the lane lines leave the view sideways beyond
    ``d = sqrt(2 R * SIDE_MARGIN_METERS)``; further rows only show the roadside. The view
    is not made shorter than the near edge plus one period of a dashed line, nor longer
    than the camera's usual view.

    Args:
        camera: Profile of the camera with its usual view.
        horizon_y: Row of the horizon, ``None`` for the camera's default one.
        radius: Radius of the lane in meters (``inf`` for a straight road).

    Returns:
        The distance of the far edge of the view from the camera in meters.
    """
    horizon = camera.horizon_y if horizon_y is None else horizon_y
    near = camera.focal_height / (camera.bottom_y - horizon)
    reach = np.sqrt(2 * radius * SIDE_MARGIN_METERS)
    return float(min(camera.view_distance, max(near + DASH_PERIOD_METERS, reach)))


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
        # The camera profiles with the current far edge of the view, by image size.
        self._cameras: dict[tuple[int, int], CameraProfile] = {}
        self._transforms: dict[tuple[int, int], tuple[tuple, PerspectiveTransform]] = {}

    def _camera_for(self, width: int, height: int) -> CameraProfile:
        """Get the profile of a camera with the current far edge of its view.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            The camera profile.

        Raises:
            ValueError: If no known camera has this resolution.
        """
        if (width, height) not in self._cameras:
            self._cameras[(width, height)] = camera_for_size(width, height)
        return self._cameras[(width, height)]

    def _transform_for(self, camera: CameraProfile) -> PerspectiveTransform:
        """Get the perspective transform of a camera for the current horizon.

        Args:
            camera: Profile of the camera.

        Returns:
            The transform for images of the camera's resolution, created again only when the
            horizon or the far edge of the view changed.
        """
        key = (self.horizon_y, camera.top_offset)
        cached = self._transforms.get(camera.size)
        if cached is None or cached[0] != key:
            transform = PerspectiveTransform.for_camera(camera, self.horizon_y)
            self._transforms[camera.size] = (key, transform)
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
        if estimate is None or abs(estimate - current) < VIEW_UPDATE_ROWS:
            return
        old = self._transform_for(camera)
        self.horizon_y = estimate
        if self.tracker is not None:
            self.tracker.reproject(old, self._transform_for(camera))

    def _follow_bend(self, camera: CameraProfile, radius: float) -> None:
        """Adapt the far edge of the view to the bend and move the tracked lane into it.

        Only the frames seen so far decide: the radius is that of the tracked lane.

        Args:
            camera: Profile of the camera with the current view.
            radius: Radius of the tracked lane in meters.
        """
        usual = camera_for_size(*camera.size)
        updated = camera.with_view_distance(view_distance_for_bend(usual, self.horizon_y, radius))
        if abs(updated.top_offset - camera.top_offset) < VIEW_UPDATE_ROWS:
            return
        old = self._transform_for(camera)
        self._cameras[camera.size] = updated
        self.tracker.reproject(old, self._transform_for(updated))

    def lane_pixels(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove the distortion and select the lane-line pixels in the camera view.

        The markings are looked for on the road between the current horizon and the car
        hood, with the lane width the camera geometry gives at every row.

        Args:
            image: 8-bit BGR road image of a known camera.

        Returns:
            The undistorted image and the binary image of its lane-line pixels (0 and 255).

        Raises:
            ValueError: If the resolution of the image belongs to no known camera or the
                image is not an 8-bit BGR image.
        """
        height, width = image.shape[:2]
        camera = camera_for_size(width, height)
        if self.calibration is not None and (width, height) == self.calibration.image_size:
            image = self.calibration.undistort(image)
        return image, threshold_lane_pixels(image, camera.lane_widths(self.horizon_y))

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
        camera = self._camera_for(width, height)
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
            if self.tracker is not None:
                self._follow_bend(prepared.camera, measurements.radius)
        return LaneResult(
            prepared.image,
            prepared.binary,
            prepared.transform,
            prepared.camera,
            fit,
            measurements,
            status,
        )
