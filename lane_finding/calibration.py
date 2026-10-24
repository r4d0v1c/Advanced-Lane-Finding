"""Camera calibration from chessboard images and lens distortion removal."""

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Number of inner chessboard corners (columns, rows) of the calibration target.
DEFAULT_PATTERN_SIZE = (9, 6)

# OpenCV recommends at least 10 views of the target for a reliable calibration.
MIN_CALIBRATION_IMAGES = 10

# The third radial coefficient (k3) is fixed to zero.
_CALIBRATION_FLAGS = cv2.CALIB_FIX_K3


@dataclass(frozen=True)
class ChessboardObservations:
    """Chessboard corners detected in a set of calibration images.

    Attributes:
        image_points: One ``(N, 1, 2)`` float32 array of corner pixel coordinates
            per usable image, where ``N = columns * rows``.
        image_size: ``(width, height)`` shared by all usable images.
        pattern_size: Inner corners ``(columns, rows)`` of the chessboard.
        used: Paths of the images the corners were taken from.
        skipped: Maps each skipped image path to the reason it was skipped.
    """

    image_points: list[np.ndarray]
    image_size: tuple[int, int]
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE
    used: list[Path] = field(default_factory=list)
    skipped: dict[Path, str] = field(default_factory=dict)


@dataclass(eq=False)
class CameraCalibration:
    """Intrinsic camera parameters together with a cached undistortion remap.

    Attributes:
        camera_matrix: ``3x3`` intrinsic matrix.
        dist_coeffs: ``(1, 5)`` distortion coefficients ``(k1, k2, p1, p2, k3)``.
        image_size: ``(width, height)`` of the images the camera was calibrated on.
        rms_error: Overall RMS reprojection error of the calibration in pixels
            (NaN when unknown).
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms_error: float = float("nan")
    _maps: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from an image.

        The remap tables are computed once on first use and reused afterwards, which
        is considerably faster than ``cv2.undistort`` when processing video frames.

        Args:
            image: Grayscale or BGR image with the calibration resolution.

        Returns:
            Undistorted image with the same shape and dtype as the input.

        Raises:
            ValueError: If the image resolution differs from the calibration one.
        """
        height, width = image.shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(
                f"Image resolution {width}x{height} does not match the calibration "
                f"resolution {self.image_size[0]}x{self.image_size[1]}"
            )
        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.camera_matrix,
                self.image_size,
                cv2.CV_16SC2,
            )
        return cv2.remap(image, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    def save(self, path: Path | str) -> None:
        """Save the calibration to a ``.npz`` file, creating parent folders.

        Args:
            path: Destination file path (should end with ``.npz``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            image_size=np.array(self.image_size),
            rms_error=self.rms_error,
        )

    @classmethod
    def load(cls, path: Path | str) -> "CameraCalibration":
        """Load a calibration written by :meth:`save`.

        Args:
            path: Path of the ``.npz`` file.

        Returns:
            The restored calibration.
        """
        with np.load(path) as data:
            width, height = data["image_size"]
            return cls(
                camera_matrix=data["camera_matrix"],
                dist_coeffs=data["dist_coeffs"],
                image_size=(int(width), int(height)),
                rms_error=float(data["rms_error"]),
            )


def build_object_points(pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE) -> np.ndarray:
    """Build the 3-D coordinates of the chessboard's inner corners.

    Args:
        pattern_size: Inner corners ``(columns, rows)`` of the chessboard.

    Returns:
        Float32 array of shape ``(columns * rows, 3)``.
    """
    columns, rows = pattern_size
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return points


def find_chessboard_corners(
    gray: np.ndarray, pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE
) -> np.ndarray | None:
    """Detect the inner corners of a chessboard with sub-pixel accuracy.

    Args:
        gray: 8-bit single-channel image.
        pattern_size: Inner corners ``(columns, rows)`` of the chessboard.

    Returns:
        Float32 array of shape ``(columns * rows, 1, 2)`` with the corner pixel
        coordinates, or ``None`` if the complete pattern was not found.
    """
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern_size, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    if not found:
        return None

    return corners.reshape(-1, 1, 2).astype(np.float32, copy=False)


def collect_observations(
    image_paths: Sequence[Path | str],
    pattern_size: tuple[int, int] = DEFAULT_PATTERN_SIZE,
) -> ChessboardObservations:
    """Detect the chessboard in every image and keep the usable detections.

    Args:
        image_paths: Paths of the calibration images.
        pattern_size: Inner corners ``(columns, rows)`` of the chessboard.

    Returns:
        The corners of all usable images together with the skipped ones.

    Raises:
        ValueError: If the chessboard is not found in any image.
    """
    detections: dict[Path, tuple[tuple[int, int], np.ndarray]] = {}
    skipped: dict[Path, str] = {}

    for path in map(Path, image_paths):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            skipped[path] = "image could not be read"
            continue
        corners = find_chessboard_corners(gray, pattern_size)
        if corners is None:
            skipped[path] = "chessboard pattern not found"
            continue
        height, width = gray.shape
        detections[path] = ((width, height), corners)

    if not detections:
        raise ValueError("Chessboard pattern was not found in any calibration image")

    image_size = Counter(size for size, _ in detections.values()).most_common(1)[0][0]
    used: list[Path] = []
    image_points: list[np.ndarray] = []
    for path, (size, corners) in detections.items():
        if size == image_size:
            used.append(path)
            image_points.append(corners)
        else:
            skipped[path] = (
                f"resolution {size[0]}x{size[1]} differs from {image_size[0]}x{image_size[1]}"
            )

    for path, reason in skipped.items():
        logger.warning("Skipping %s: %s", path.name, reason)
    logger.info(
        "Chessboard found in %d of %d images (resolution %dx%d)",
        len(used),
        len(used) + len(skipped),
        *image_size,
    )
    return ChessboardObservations(image_points, image_size, pattern_size, used, skipped)


def calibrate(observations: ChessboardObservations) -> CameraCalibration:
    """Estimate the camera matrix and distortion coefficients.

    Args:
        observations: Chessboard corners from :func:`collect_observations`.

    Returns:
        The calibrated camera.

    Raises:
        ValueError: If fewer than ``MIN_CALIBRATION_IMAGES`` images are available.
    """
    count = len(observations.image_points)
    if count < MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Calibration needs at least {MIN_CALIBRATION_IMAGES} chessboard images, got {count}"
        )
    object_points = build_object_points(observations.pattern_size)
    rms_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        [object_points] * count,
        observations.image_points,
        observations.image_size,
        None,
        None,
        flags=_CALIBRATION_FLAGS,
    )
    return CameraCalibration(camera_matrix, dist_coeffs, observations.image_size, float(rms_error))
