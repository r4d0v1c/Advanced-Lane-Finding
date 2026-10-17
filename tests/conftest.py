"""Fixtures shared by several test modules."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_finding.calibration import (
    CameraCalibration,
    ChessboardObservations,
    calibrate,
    collect_observations,
)
from lane_finding.cameras import camera_for_size
from lane_finding.perspective import PerspectiveTransform
from lane_finding.thresholding import threshold_lane_pixels

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = REPO_ROOT / "resources" / "camera_cal_export"
VIDEOS_DIR = REPO_ROOT / "resources" / "test_videos_export"


@pytest.fixture(scope="session")
def observations() -> ChessboardObservations:
    """Detect the chessboard once in all real calibration images.

    Returns:
        Observations of the repository's calibration images.
    """
    return collect_observations(sorted(CALIBRATION_DIR.glob("*.jpg")))


@pytest.fixture(scope="session")
def calibration(observations: ChessboardObservations) -> CameraCalibration:
    """Calibrate once on the real calibration images.

    Args:
        observations: Detected corners of the calibration images.

    Returns:
        The calibrated camera.
    """
    return calibrate(observations)


@pytest.fixture(scope="session")
def read_video_frame() -> Callable[[str, float], np.ndarray]:
    """Provide a function that reads one frame of a test video.

    Returns:
        Function taking the video file name and the position in seconds and
        returning the BGR frame.
    """

    def read(name: str, seconds: float) -> np.ndarray:
        """Read one frame of a test video.

        Args:
            name: File name inside the test video folder.
            seconds: Position of the frame in the video.

        Returns:
            BGR frame.
        """
        capture = cv2.VideoCapture(str(VIDEOS_DIR / name))
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        found, frame = capture.read()
        capture.release()
        assert found, f"could not read {name} at {seconds} s"
        return frame

    return read


@pytest.fixture(scope="session")
def birds_eye_binary(calibration: CameraCalibration) -> Callable[..., np.ndarray]:
    """Provide a function that turns a road image into the bird's-eye binary image.

    Args:
        calibration: Calibration of the main camera.

    Returns:
        Function taking a BGR road image of a known camera and optionally the row of the
        horizon, and returning the bird's-eye binary image the lane fit works on.
    """

    def convert(image: np.ndarray, horizon_y: float | None = None) -> np.ndarray:
        """Undistort (main camera), threshold and warp a road image.

        Args:
            image: BGR road image.
            horizon_y: Row of the horizon to use instead of the default one of the camera.

        Returns:
            Bird's-eye binary image.
        """
        height, width = image.shape[:2]
        camera = camera_for_size(width, height)
        if (width, height) == calibration.image_size:
            image = calibration.undistort(image)
        transform = PerspectiveTransform.for_camera(camera, horizon_y)
        return transform.warp(threshold_lane_pixels(image), cv2.INTER_NEAREST)

    return convert
