"""Perspective transform between the camera view and a bird's-eye view of the road.

In the camera view the lane lines converge towards the horizon. After the transform
the road is seen from above: straight lane lines are vertical and parallel, and a
curve is an ordinary curve that can be fitted with a polynomial.
"""

import cv2
import numpy as np

from lane_finding.cameras import CameraProfile

# The lane is mapped to the middle half of the bird's-eye image; the margins on both
# sides keep the lane lines visible when the car is off-center or the road bends.
LANE_MARGIN_FRACTION = 0.25
# The lane therefore spans this fraction of the image width.
LANE_WIDTH_FRACTION = 1 - 2 * LANE_MARGIN_FRACTION


def destination_points(size: tuple[int, int]) -> np.ndarray:
    """Compute the rectangle of the bird's-eye view that the lane is mapped to.

    Args:
        size: ``(width, height)`` of the image.

    Returns:
        Float32 array of shape ``(4, 2)`` with the corners in the order top-left,
        top-right, bottom-right, bottom-left.
    """
    width, height = size
    left = LANE_MARGIN_FRACTION * width
    right = width - left
    return np.float32([[left, 0], [right, 0], [right, height], [left, height]])


class PerspectiveTransform:
    """Perspective transform of a fixed image size and its inverse.

    The matrices are computed once, so applying the transform to every frame of a
    video only costs the warp itself. Source and result images have the same size.

    Attributes:
        size: ``(width, height)`` of the images the transform works on.
        source_points: Trapezoid on the road in the camera view.
        destination_points: Rectangle in the bird's-eye view it is mapped to.
        matrix: ``3x3`` matrix from the camera view to the bird's-eye view.
        inverse_matrix: ``3x3`` matrix from the bird's-eye view back to the camera view.
    """

    def __init__(
        self,
        source_points: np.ndarray,
        destination_points: np.ndarray,
        size: tuple[int, int],
    ) -> None:
        """Compute the transform from four point pairs.

        Args:
            source_points: Float32 array of shape ``(4, 2)`` in the camera view.
            destination_points: Float32 array of shape ``(4, 2)``, the same corners in
                the bird's-eye view.
            size: ``(width, height)`` of the images.
        """
        self.size = size
        self.source_points = source_points
        self.destination_points = destination_points
        self.matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(destination_points, source_points)

    @classmethod
    def for_camera(
        cls, camera: CameraProfile, horizon_y: float | None = None
    ) -> "PerspectiveTransform":
        """Create the transform of a known camera.

        Args:
            camera: Camera profile that describes the road geometry.
            horizon_y: Row of the horizon to use instead of the default one of the
                camera.

        Returns:
            The transform for images of the camera's resolution.
        """
        region = camera.region_of_interest(horizon_y)
        return cls(region, destination_points(camera.size), camera.size)

    def warp(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Transform a camera image to the bird's-eye view.

        Args:
            image: Image of the transform's size.
            interpolation: OpenCV interpolation flag. Use ``cv2.INTER_NEAREST`` for
                binary images so that they keep the values 0 and 255.

        Returns:
            Bird's-eye view with the same size, dtype and channels as the input.

        Raises:
            ValueError: If the image does not have the size of the transform.
        """
        self._check_size(image)
        return cv2.warpPerspective(image, self.matrix, self.size, flags=interpolation)

    def unwarp(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Transform a bird's-eye image back to the camera view.

        Args:
            image: Bird's-eye image of the transform's size.
            interpolation: OpenCV interpolation flag.

        Returns:
            Image in the camera view with the same size, dtype and channels as the input.

        Raises:
            ValueError: If the image does not have the size of the transform.
        """
        self._check_size(image)
        return cv2.warpPerspective(image, self.inverse_matrix, self.size, flags=interpolation)

    def _check_size(self, image: np.ndarray) -> None:
        """Check that an image has the size of the transform.

        Args:
            image: Image to check.

        Raises:
            ValueError: If the image size differs from the size of the transform.
        """
        height, width = image.shape[:2]
        if (width, height) != self.size:
            raise ValueError(
                f"Image size {width}x{height} does not match the transform size "
                f"{self.size[0]}x{self.size[1]}"
            )
