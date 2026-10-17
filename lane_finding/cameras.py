"""Cameras that appear in the test material and the road geometry of each of them.

The geometry was measured on straight road sections (``straight_lines1/2`` for the
main camera, ``solid*`` images and videos for the second camera). On a flat road the
parallel lane lines meet at the horizon and the lane width grows linearly with the
distance from the horizon in the image:

    lane width in pixels = lane_width_per_row * (row - horizon_y)

The lane width per row is about 3 for both cameras, which means that both are mounted
at the same height. The vanishing point lies in the middle column of the image.

The same model gives the ground distance of an image row, which turns the bird's-eye view
into a map with a known scale:

    distance in metres = focal_height / (row - horizon_y)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    """Road geometry seen by one camera.

    Attributes:
        name: Human-readable name.
        size: ``(width, height)`` of the camera images in pixels.
        horizon_y: Row of the vanishing point of the lane lines in a typical frame.
            It moves with the pitch of the car and the slope of the road.
        lane_width_per_row: Growth of the lane width in pixels per image row below
            the horizon.
        top_offset: Rows below the horizon where the far edge of the bird's-eye view
            lies (about 35 m ahead). Smaller values look further.
        bottom_y: Row of the near edge of the bird's-eye view, just above the car hood
            or the bottom of the image.
        focal_height: Focal length (in pixels) times the height of the camera above the
            road (in metres). It was calibrated with the dashed lane line, whose dashes
            repeat every 14.63 m (12 ft dash and 36 ft gap), and agrees with the focal
            length of the chessboard calibration within a few percent for the main
            camera. The distances are accurate to about 5 %, mostly because they depend
            on the horizon row.
    """

    name: str
    size: tuple[int, int]
    horizon_y: float
    lane_width_per_row: float
    top_offset: float
    bottom_y: float
    focal_height: float

    def source_points(self, horizon_y: float | None = None) -> np.ndarray:
        """Compute the road trapezoid that is mapped to the bird's-eye view.

        The trapezoid is one lane wide, symmetric around the middle column of the
        image and its edges run along the lane lines towards the horizon.

        Args:
            horizon_y: Row of the horizon to use instead of the default one of the
                camera, for example when it was measured for a particular video.

        Returns:
            Float32 array of shape ``(4, 2)`` with the corners in the order top-left,
            top-right, bottom-right, bottom-left.

        Raises:
            ValueError: If the horizon is so low that no road remains between the
                far and the near edge.
        """
        horizon = self.horizon_y if horizon_y is None else horizon_y
        top_y = horizon + self.top_offset
        if top_y >= self.bottom_y:
            raise ValueError(
                f"Horizon at row {horizon:.0f} leaves no road above row {self.bottom_y:.0f}"
            )
        center_x = self.size[0] / 2
        top_half = 0.5 * self.lane_width_per_row * (top_y - horizon)
        bottom_half = 0.5 * self.lane_width_per_row * (self.bottom_y - horizon)
        return np.float32(
            [
                [center_x - top_half, top_y],
                [center_x + top_half, top_y],
                [center_x + bottom_half, self.bottom_y],
                [center_x - bottom_half, self.bottom_y],
            ]
        )


# The camera of the chessboard images: calibration applies, the car hood is visible
# from row 672 on.
MAIN_CAMERA = CameraProfile(
    name="1280x720 camera",
    size=(1280, 720),
    horizon_y=420.0,
    lane_width_per_row=2.97,
    top_offset=40.0,
    bottom_y=668.0,
    focal_height=1400.0,
)

# A different camera (no hood visible) without calibration.
SECOND_CAMERA = CameraProfile(
    name="960x540 camera",
    size=(960, 540),
    horizon_y=308.0,
    lane_width_per_row=3.01,
    top_offset=30.0,
    bottom_y=535.0,
    focal_height=1340.0,
)

CAMERAS = (MAIN_CAMERA, SECOND_CAMERA)


def camera_for_size(width: int, height: int) -> CameraProfile:
    """Find the camera that produces images of a given resolution.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The matching camera profile.

    Raises:
        ValueError: If no known camera has this resolution.
    """
    for camera in CAMERAS:
        if camera.size == (width, height):
            return camera
    known = ", ".join(f"{camera.size[0]}x{camera.size[1]}" for camera in CAMERAS)
    raise ValueError(f"No camera is known for {width}x{height} images (known: {known})")
