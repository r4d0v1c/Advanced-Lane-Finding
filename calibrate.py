"""Calibrate the camera from chessboard images and save the result.

Usage:
    python3 calibrate.py [--images-dir DIR] [--output FILE] [--preview FILE]
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from lane_finding.calibration import CameraCalibration, calibrate, collect_observations

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = REPO_ROOT / "resources" / "camera_cal_export"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "calibration.npz"
DEFAULT_PREVIEW = REPO_ROOT / "output" / "undistort_preview.png"

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
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="folder with the chessboard images (*.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to save the calibration (.npz)",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=DEFAULT_PREVIEW,
        help="where to save an original/undistorted comparison of the first image",
    )
    return parser.parse_args(argv)


def make_preview(calibration: CameraCalibration, image_path: Path) -> np.ndarray:
    """Build a side-by-side comparison of an image before and after undistortion.

    Args:
        calibration: Calibration used to undistort the image.
        image_path: Path of the image to show.

    Returns:
        BGR image with the original on the left and the undistorted one on the right.
    """
    original = cv2.imread(str(image_path))
    panels = []
    for title, panel in (
        ("Original", original),
        ("Undistorted", calibration.undistort(original)),
    ):
        panel = cv2.resize(panel, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
        cv2.putText(panel, title, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        panels.append(panel)
    return np.hstack(panels)


def main(argv: list[str] | None = None) -> None:
    """Run the calibration, save it and write the preview image.

    Args:
        argv: Argument list; ``sys.argv[1:]`` is used when ``None``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No .jpg images found in {args.images_dir}")

    calibration = calibrate(collect_observations(image_paths))
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

    args.preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.preview), make_preview(calibration, image_paths[0]))
    logger.info("Preview saved to %s", args.preview)


if __name__ == "__main__":
    main()
