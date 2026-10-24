"""Lane curvature and vehicle position in meters.

The bird's-eye view is a map of the road, so a length in pixels can be turned into meters
with two scales:

* horizontal: the lane is 3.7 m wide and spans ``LANE_WIDTH_FRACTION`` of the image width,
* vertical: the ground distance of the near and the far edge of the view follows from the
  camera geometry (``CameraProfile.focal_height``).

Assumptions: the lane is exactly ``LANE_WIDTH_METERS`` wide, the road is flat, and the
vehicle's axis is the middle column of the bird's-eye view (the camera sits on the
vehicle's axis and looks straight ahead).
"""

import math
from dataclasses import dataclass

import numpy as np

from lane_finding.cameras import LANE_WIDTH_METERS, CameraProfile
from lane_finding.lane_fit import LaneFit
from lane_finding.perspective import LANE_WIDTH_FRACTION

# The vehicle's axis, as a fraction of the image width.
CAR_POSITION_FRACTION = 0.5
# A radius above this is shown as "straight": the curvature of such a gentle bend is smaller
# than the error of its measurement (about 0.1 per km).
STRAIGHT_RADIUS_METERS = 3000.0


@dataclass(frozen=True)
class MetersPerPixel:
    """Scale of the bird's-eye view.

    Attributes:
        horizontal: Meters per pixel along the image columns.
        vertical: Meters per pixel along the image rows.
    """

    horizontal: float
    vertical: float


@dataclass(frozen=True)
class LaneMeasurements:
    """Shape of the lane and position of the vehicle in it, in meters.

    Attributes:
        curvature: Curvature of the lane center in 1/m at the near edge of the view;
            positive for a bend to the right, negative to the left, zero if straight.
        vehicle_offset: Lateral position of the vehicle relative to the lane center;
            positive if the vehicle is to the right of the center.
        lane_width: Distance between the two lane lines at the near edge of the view.
    """

    curvature: float
    vehicle_offset: float
    lane_width: float

    @property
    def radius(self) -> float:
        """Radius of curvature of the lane center in meters (infinite if straight)."""
        return math.inf if self.curvature == 0 else 1 / abs(self.curvature)

    def text_lines(self) -> list[str]:
        """Describe the measurements in words for a display.

        Returns:
            Three lines: the radius and the direction of the bend ("straight" above
            ``STRAIGHT_RADIUS_METERS``), the position of the vehicle and the lane width.
        """
        if self.radius > STRAIGHT_RADIUS_METERS:
            bend = "Radius: straight"
        else:
            direction = "right" if self.curvature > 0 else "left"
            bend = f"Radius: {self.radius:.0f} m ({direction})"
        if abs(self.vehicle_offset) < 0.005:
            position = "Vehicle: at the center"
        else:
            side = "right" if self.vehicle_offset > 0 else "left"
            position = f"Vehicle: {abs(self.vehicle_offset):.2f} m {side} of center"
        return [bend, position, f"Lane width: {self.lane_width:.2f} m"]


def meters_per_pixel(camera: CameraProfile, horizon_y: float | None = None) -> MetersPerPixel:
    """Compute the scale of the bird's-eye view of a camera.

    Args:
        camera: Camera profile that describes the road geometry.
        horizon_y: Row of the horizon the bird's-eye view was built with, if it differs
            from the default one of the camera.

    Returns:
        The meters per pixel along the columns and along the rows.

    Raises:
        ValueError: If the horizon is so low that no road remains between the far and the
            near edge.
    """
    horizon = camera.horizon_y if horizon_y is None else horizon_y
    if camera.bottom_y - horizon <= camera.top_offset:
        raise ValueError(
            f"Horizon at row {horizon:.0f} leaves no road above row {camera.bottom_y:.0f}"
        )
    far = camera.focal_height / camera.top_offset
    near = camera.focal_height / (camera.bottom_y - horizon)
    width, height = camera.size
    return MetersPerPixel(
        horizontal=LANE_WIDTH_METERS / (LANE_WIDTH_FRACTION * width),
        vertical=(far - near) / height,
    )


def polynomial_in_meters(coefficients: np.ndarray, scale: MetersPerPixel) -> np.ndarray:
    """Convert a polynomial from pixels to meters.

    The polynomial ``x = A*y**2 + B*y + C`` in pixels becomes the polynomial of the same
    curve in meters, with both the column and the row measured in meters.

    Args:
        coefficients: ``(A, B, C)`` in pixels.
        scale: Meters per pixel of the bird's-eye view.

    Returns:
        ``(A, B, C)`` in meters.
    """
    a, b, c = coefficients
    return np.array(
        [
            a * scale.horizontal / scale.vertical**2,
            b * scale.horizontal / scale.vertical,
            c * scale.horizontal,
        ]
    )


def measure_lane(
    fit: LaneFit, size: tuple[int, int], scale: MetersPerPixel
) -> LaneMeasurements | None:
    """Measure the curvature of the lane and the position of the vehicle.

    The curvature is that of the lane center (the mean of the two polynomials) and the
    position is measured at the bottom row, the closest to the vehicle.

    Args:
        fit: Result of the lane line detection.
        size: ``(width, height)`` of the bird's-eye image.
        scale: Meters per pixel of the bird's-eye view.

    Returns:
        The measurements, or ``None`` if one of the lane lines was not found.
    """
    if fit.left is None or fit.right is None:
        return None
    width, height = size
    bottom_row = height - 1
    center = 0.5 * (fit.left + fit.right)
    a, b, _ = polynomial_in_meters(center, scale)
    slope = 2 * a * bottom_row * scale.vertical + b
    curvature = 2 * a / (1 + slope**2) ** 1.5
    offset = (CAR_POSITION_FRACTION * width - np.polyval(center, bottom_row)) * scale.horizontal
    lane_width = (np.polyval(fit.right, bottom_row) - np.polyval(fit.left, bottom_row)) * (
        scale.horizontal
    )
    return LaneMeasurements(float(curvature), float(offset), float(lane_width))
