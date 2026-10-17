"""Tests for lane_finding.calibration."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_finding.calibration import (
    CameraCalibration,
    ChessboardObservations,
    build_object_points,
    calibrate,
    collect_observations,
    find_chessboard_corners,
)

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "resources" / "camera_cal_export"


def line_fit_rms(points: np.ndarray) -> float:
    """Compute the RMS orthogonal distance of 2-D points from their best-fit line.

    Args:
        points: Array of shape ``(N, 2)``.

    Returns:
        RMS distance in the unit of the input coordinates.
    """
    singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return float(singular_values[-1] / np.sqrt(len(points)))


def grid_straightness(corners: np.ndarray, pattern_size: tuple[int, int]) -> float:
    """Measure how straight the rows and columns of a detected corner grid are.

    A chessboard is planar, so its rows and columns project to straight lines unless
    the lens distorts them; the value is therefore a direct distortion measure.

    Args:
        corners: Corner coordinates of shape ``(columns * rows, 1, 2)``.
        pattern_size: Inner corners ``(columns, rows)``.

    Returns:
        Mean line-fit RMS over all rows and columns, in pixels.
    """
    columns, rows = pattern_size
    grid = corners.reshape(rows, columns, 2)
    residuals = [line_fit_rms(row) for row in grid] + [
        line_fit_rms(col) for col in grid.transpose(1, 0, 2)
    ]
    return float(np.mean(residuals))


def synthetic_observations(
    camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: tuple[int, int], views: int
) -> ChessboardObservations:
    """Project a chessboard into randomly posed virtual views of a known camera.

    Args:
        camera_matrix: Ground-truth ``3x3`` intrinsic matrix.
        dist_coeffs: Ground-truth distortion coefficients.
        image_size: ``(width, height)`` of the virtual images.
        views: Number of views to generate.

    Returns:
        Noise-free observations of all views.
    """
    rng = np.random.default_rng(0)
    object_points = build_object_points()
    board_center = object_points.mean(axis=0)
    image_points = []
    for _ in range(views):
        rotation = rng.uniform(-0.5, 0.5, 3)
        translation = np.array([rng.uniform(-3, 3), rng.uniform(-2, 2), rng.uniform(14, 24)])
        # Rotate around the board center so the board stays in front of the camera.
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        translation = translation - rotation_matrix @ board_center
        projected, _ = cv2.projectPoints(
            object_points, rotation, translation, camera_matrix, dist_coeffs
        )
        image_points.append(projected.astype(np.float32))
    return ChessboardObservations(image_points, image_size)


def test_object_points_layout():
    """Object points form a unit grid in the z=0 plane, ordered row by row."""
    points = build_object_points((9, 6))
    assert points.shape == (54, 3)
    assert points.dtype == np.float32
    assert np.array_equal(points[0], [0, 0, 0])
    assert np.array_equal(points[1], [1, 0, 0])
    assert np.array_equal(points[9], [0, 1, 0])
    assert np.array_equal(points[-1], [8, 5, 0])
    assert not points[:, 2].any()


def test_find_corners_on_complete_board():
    """A fully visible 9x6 board yields 54 corners inside the image."""
    gray = cv2.imread(str(CALIBRATION_DIR / "calibration2.jpg"), cv2.IMREAD_GRAYSCALE)
    corners = find_chessboard_corners(gray)
    assert corners is not None
    assert corners.shape == (54, 1, 2)
    assert corners.dtype == np.float32
    assert (corners[..., 0] >= 0).all() and (corners[..., 0] < gray.shape[1]).all()
    assert (corners[..., 1] >= 0).all() and (corners[..., 1] < gray.shape[0]).all()


def test_find_corners_returns_none_without_complete_board():
    """Neither a blank image nor a board cropped by the image border is detected."""
    assert find_chessboard_corners(np.zeros((720, 1280), np.uint8)) is None
    cropped = cv2.imread(str(CALIBRATION_DIR / "calibration1.jpg"), cv2.IMREAD_GRAYSCALE)
    assert find_chessboard_corners(cropped) is None


def test_collect_observations_accounts_for_every_image(observations):
    """Every input image is either used or skipped with a reason, never both."""
    inputs = set(CALIBRATION_DIR.glob("*.jpg"))
    assert set(observations.used) | set(observations.skipped) == inputs
    assert not set(observations.used) & set(observations.skipped)
    assert len(observations.used) == len(observations.image_points)
    assert len(observations.used) >= 10
    assert observations.image_size == (1280, 720)
    assert all(observations.skipped.values())


def test_collect_observations_skips_other_resolutions(tmp_path):
    """An image whose resolution differs from the majority is skipped."""
    for name in ("calibration2.jpg", "calibration3.jpg"):
        (tmp_path / name).write_bytes((CALIBRATION_DIR / name).read_bytes())
    odd = cv2.imread(str(CALIBRATION_DIR / "calibration6.jpg"))
    cv2.imwrite(str(tmp_path / "calibration6.jpg"), cv2.resize(odd, (1281, 721)))

    result = collect_observations(sorted(tmp_path.glob("*.jpg")))

    assert [path.name for path in result.used] == ["calibration2.jpg", "calibration3.jpg"]
    assert "resolution" in result.skipped[tmp_path / "calibration6.jpg"]


def test_collect_observations_rejects_images_without_board(tmp_path):
    """A folder without any visible chessboard raises a clear error."""
    cv2.imwrite(str(tmp_path / "blank.png"), np.zeros((100, 100), np.uint8))
    with pytest.raises(ValueError, match="not found in any"):
        collect_observations([tmp_path / "blank.png", tmp_path / "missing.png"])


def test_calibrate_recovers_known_camera():
    """Calibration reproduces the parameters of a synthetic camera."""
    true_matrix = np.array([[1100.0, 0, 640.0], [0, 1100.0, 360.0], [0, 0, 1]])
    true_dist = np.array([-0.25, 0.05, 0.0, 0.0, 0.0])

    result = calibrate(synthetic_observations(true_matrix, true_dist, (1280, 720), views=12))

    assert result.rms_error < 0.01
    assert result.camera_matrix[0, 0] == pytest.approx(1100.0, rel=0.01)
    assert result.camera_matrix[1, 1] == pytest.approx(1100.0, rel=0.01)
    assert result.camera_matrix[0, 2] == pytest.approx(640.0, abs=3.0)
    assert result.camera_matrix[1, 2] == pytest.approx(360.0, abs=3.0)
    assert result.dist_coeffs.ravel()[0] == pytest.approx(-0.25, abs=0.02)


def test_calibrate_requires_enough_images():
    """Too few views are rejected instead of producing an unreliable result."""
    matrix = np.array([[1100.0, 0, 640.0], [0, 1100.0, 360.0], [0, 0, 1]])
    few_views = synthetic_observations(matrix, np.zeros(5), (1280, 720), views=3)
    with pytest.raises(ValueError, match="at least"):
        calibrate(few_views)


def test_calibration_of_real_images_is_plausible(calibration):
    """The real calibration has a sub-pixel error and a sensible camera matrix."""
    width, height = calibration.image_size
    matrix = calibration.camera_matrix
    assert calibration.rms_error < 1.0
    assert matrix[0, 0] == pytest.approx(matrix[1, 1], rel=0.02)
    assert 0.4 * width < matrix[0, 2] < 0.6 * width
    assert 0.4 * height < matrix[1, 2] < 0.6 * height
    assert calibration.dist_coeffs.ravel()[0] < 0  # wide-angle lens: barrel distortion


def test_undistortion_straightens_chessboard_lines(calibration, observations):
    """Undistorting the detected corners makes the board's rows and columns straighter."""
    pattern = observations.pattern_size
    before = [grid_straightness(corners, pattern) for corners in observations.image_points]
    after = [
        grid_straightness(
            cv2.undistortPoints(
                corners,
                calibration.camera_matrix,
                calibration.dist_coeffs,
                P=calibration.camera_matrix,
            ),
            pattern,
        )
        for corners in observations.image_points
    ]
    assert np.mean(after) < 0.6 * np.mean(before)
    assert np.mean(after) < 0.5  # pixels


def test_undistort_keeps_shape_and_dtype(calibration):
    """The output image has the same shape and dtype as the input."""
    image = cv2.imread(str(CALIBRATION_DIR / "calibration2.jpg"))
    result = calibration.undistort(image)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_undistort_matches_opencv_reference(calibration):
    """The cached-remap result agrees with cv2.undistort up to fixed-point rounding."""
    image = cv2.imread(str(CALIBRATION_DIR / "calibration2.jpg"))
    reference = cv2.undistort(image, calibration.camera_matrix, calibration.dist_coeffs)
    difference = np.abs(calibration.undistort(image).astype(int) - reference.astype(int))
    assert difference.mean() < 0.5


def test_undistort_rejects_wrong_resolution(calibration):
    """Frames of another resolution cannot be undistorted with this calibration."""
    with pytest.raises(ValueError, match="resolution"):
        calibration.undistort(np.zeros((100, 200, 3), np.uint8))


def test_undistort_without_distortion_is_identity():
    """Zero distortion coefficients leave the image unchanged."""
    matrix = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1]])
    camera = CameraCalibration(matrix, np.zeros((1, 5)), (640, 480))
    image = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert np.abs(camera.undistort(image).astype(int) - image.astype(int)).max() <= 1


def test_save_and_load_roundtrip(calibration, tmp_path):
    """A saved calibration is restored with identical values."""
    path = tmp_path / "nested" / "calibration.npz"
    calibration.save(path)
    restored = CameraCalibration.load(path)
    assert np.array_equal(restored.camera_matrix, calibration.camera_matrix)
    assert np.array_equal(restored.dist_coeffs, calibration.dist_coeffs)
    assert restored.image_size == calibration.image_size
    assert restored.rms_error == calibration.rms_error
