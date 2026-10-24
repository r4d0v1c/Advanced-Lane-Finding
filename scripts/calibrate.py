"""Calibrate the camera from chessboard images, save the result and show what it did.

Usage:
    python3 scripts/calibrate.py [IMAGE ...] [--images-dir DIR] [--output FILE]
                                 [--corners-dir DIR] [--comparison-dir DIR]

The calibration always uses all images of the images folder. For every image (or only for
the IMAGEs given) two pictures are saved: the detected chessboard corners drawn and
connected row by row (default folder output/corners) and the image before and after the
lens distortion was removed (default folder output/undistorted).
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # the scripts are in a subfolder: make lane_finding importable

from lane_finding.calibration import (  # noqa: E402
    DEFAULT_PATTERN_SIZE,
    CameraCalibration,
    calibrate,
    collect_observations,
    find_chessboard_corners,
)
from lane_finding.visualization import side_by_side  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = REPO_ROOT / "resources" / "camera_cal_export"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "calibration.npz"
DEFAULT_CORNERS_DIR = REPO_ROOT / "output" / "corners"
DEFAULT_COMPARISON_DIR = REPO_ROOT / "output" / "undistorted"

PREVIEW_SCALE = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` is used when ``None``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="images to save the corners and the comparison for (default: all calibration images)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="folder with the chessboard images (*.jpg) the camera is calibrated with",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to save the calibration (.npz)",
    )
    parser.add_argument(
        "--corners-dir",
        type=Path,
        default=DEFAULT_CORNERS_DIR,
        help="where to save the images with the detected corners drawn",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="where to save the before / after comparison of the distortion removal",
    )
    return parser.parse_args(argv)


def save_corner_image(image: np.ndarray, corners: np.ndarray, path: Path) -> None:
    """Save an image with its chessboard corners drawn and connected row by row.

    The corners are colored from the first to the last, so a wrong or swapped detection
    is easy to spot.

    Args:
        image: BGR image the corners were found in.
        corners: Float32 array of shape ``(N, 1, 2)`` with the corner coordinates.
        path: Where to save the picture.
    """
    drawn = image.copy()
    cv2.drawChessboardCorners(drawn, DEFAULT_PATTERN_SIZE, corners, True)
    cv2.imwrite(str(path), drawn)


def save_comparison(calibration: CameraCalibration, image: np.ndarray, path: Path) -> None:
    """Save an image before and after the lens distortion was removed, side by side.

    Args:
        calibration: Calibration used to undistort the image.
        image: BGR image with the calibration resolution.
        path: Where to save the picture.

    Raises:
        ValueError: If the image resolution differs from the calibration resolution.
    """
    comparison = side_by_side(
        [image, calibration.undistort(image)], ["Original", "Undistorted"], PREVIEW_SCALE
    )
    cv2.imwrite(str(path), comparison)


def main(argv: list[str] | None = None) -> None:
    """Run the calibration, save it and the picture pairs for the chosen images.

    Args:
        argv: Argument list; ``sys.argv[1:]`` is used when ``None``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No .jpg images found in {args.images_dir}")

    observations = collect_observations(image_paths)
    calibration = calibrate(observations)
    calibration.save(args.output)
    logger.info(
        "Camera matrix:\n%s",
        np.array2string(calibration.camera_matrix, precision=2, suppress_small=True),
    )
    logger.info(
        "Distortion coefficients (k1 k2 p1 p2 k3): %s",
        np.array2string(calibration.dist_coeffs.ravel(), precision=4, suppress_small=True),
    )
    logger.info("RMS reprojection error: %.3f px", calibration.rms_error)
    logger.info("Calibration saved to %s", args.output)

    known_corners = {
        path.resolve(): corners
        for path, corners in zip(observations.used, observations.image_points, strict=True)
    }
    args.corners_dir.mkdir(parents=True, exist_ok=True)
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    corner_count = comparison_count = 0
    for path in args.images or image_paths:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Skipping %s: image could not be read", path)
            continue
        corners = known_corners.get(path.resolve())
        if corners is None:
            corners = find_chessboard_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if corners is None:
            logger.warning("%s: chessboard pattern not found, no corner image", path.name)
        else:
            save_corner_image(image, corners, args.corners_dir / f"{path.stem}.png")
            corner_count += 1
        try:
            save_comparison(calibration, image, args.comparison_dir / f"{path.stem}.png")
            comparison_count += 1
        except ValueError as error:
            logger.warning("%s: no comparison (%s)", path.name, error)
    logger.info("Saved %d corner images to %s", corner_count, args.corners_dir)
    logger.info("Saved %d comparisons to %s", comparison_count, args.comparison_dir)


if __name__ == "__main__":
    main()
