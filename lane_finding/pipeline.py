"""The complete lane finding pipeline for one road image.

The steps are:

1. remove the lens distortion (only for the camera the calibration was made for),
2. select the lane-line pixels by color (yellow and white),
3. warp the binary image to the bird's-eye view,
4. find the lane lines and fit a polynomial to each of them,
5. measure the curvature of the lane and the position of the vehicle in meters.

The camera is recognized by the resolution of the image, because each camera has its own
road geometry (see :mod:`lane_finding.cameras`).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from lane_finding.calibration import CameraCalibration
from lane_finding.cameras import CameraProfile, camera_for_size
from lane_finding.lane_fit import LaneFit, fit_lane
from lane_finding.measurements import LaneMeasurements, measure_lane, meters_per_pixel
from lane_finding.perspective import PerspectiveTransform
from lane_finding.thresholding import threshold_lane_pixels


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
        fit: The lane lines in the bird's-eye view.
        measurements: Curvature, vehicle position and lane width in meters; ``None`` if
            the lane lines do not describe a plausible lane.
    """

    fit: LaneFit
    measurements: LaneMeasurements | None

    @property
    def plausible(self) -> bool:
        """Whether a plausible lane was found."""
        return self.measurements is not None


class LanePipeline:
    """Runs all steps from a camera image to the measured lane.

    The perspective transform of each camera is created once and reused, so processing
    many images only costs the processing itself.
    """

    def __init__(
        self, calibration: CameraCalibration | None = None, horizon_y: float | None = None
    ) -> None:
        """Create a pipeline.

        Args:
            calibration: Calibration of the main camera, used to remove the lens distortion
                of its images. Without it no image is undistorted.
            horizon_y: Row of the horizon to build the bird's-eye view with, instead of the
                default one of each camera (it differs a little between videos).
        """
        self.calibration = calibration
        self.horizon_y = horizon_y
        self._transforms: dict[tuple[int, int], PerspectiveTransform] = {}

    def _transform_for(self, camera: CameraProfile) -> PerspectiveTransform:
        """Get the perspective transform of a camera, creating it on first use.

        Args:
            camera: Profile of the camera.

        Returns:
            The transform for images of the camera's resolution.
        """
        if camera.size not in self._transforms:
            self._transforms[camera.size] = PerspectiveTransform.for_camera(camera, self.horizon_y)
        return self._transforms[camera.size]

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
        if self.calibration is not None and (width, height) == self.calibration.image_size:
            image = self.calibration.undistort(image)
        transform = self._transform_for(camera)
        binary = transform.warp(threshold_lane_pixels(image), cv2.INTER_NEAREST)
        return PreparedImage(image, binary, transform, camera)

    def process(self, image: np.ndarray) -> LaneResult:
        """Find the lane in a road image and measure it.

        Args:
            image: 8-bit BGR road image of a known camera.

        Returns:
            The prepared image, the lane lines and, if the lane is plausible, its
            measurements.

        Raises:
            ValueError: If the resolution of the image belongs to no known camera or the
                image is not an 8-bit BGR image.
        """
        prepared = self.prepare(image)
        fit = fit_lane(prepared.binary)
        measurements = None
        if fit.is_plausible(prepared.camera.size):
            scale = meters_per_pixel(prepared.camera, self.horizon_y)
            measurements = measure_lane(fit, prepared.camera.size, scale)
        return LaneResult(
            prepared.image, prepared.binary, prepared.transform, prepared.camera, fit, measurements
        )
