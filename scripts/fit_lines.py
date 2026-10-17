"""Find the lane lines in the bird's-eye view of road images and fit polynomials to them.

Usage:
    python3 scripts/fit_lines.py [IMAGE ...] [--output-dir DIR]

Without arguments every image of resources/test_images_export is processed. For each
image the result shows the bird's-eye view and the lane-like pixels of the binary image:
the pixels of the left line are red, those of the right line blue, the other pixels white.
The fitted curves are green (a curve that was derived from the other line is yellow). The
camera is recognized by the image resolution. Images of the calibrated camera are
undistorted first (run scripts/calibrate.py before), images of the other camera are used
as they are.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # the scripts are in a subfolder: make lane_finding importable

from lane_finding.calibration import CameraCalibration  # noqa: E402
from lane_finding.cameras import camera_for_size  # noqa: E402
from lane_finding.lane_fit import LaneFit, fit_lane  # noqa: E402
from lane_finding.perspective import PerspectiveTransform  # noqa: E402
from lane_finding.thresholding import threshold_lane_pixels  # noqa: E402
from lane_finding.visualization import draw_lane_fit, side_by_side  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = REPO_ROOT / "resources" / "test_images_export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "fit"
CALIBRATION_PATH = REPO_ROOT / "output" / "calibration.npz"

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
        help="road images to process (default: all test images)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where to save the result images",
    )
    return parser.parse_args(argv)


def describe(fit: LaneFit) -> str:
    """Summarize a lane fit in one line for the log.

    Args:
        fit: Result of the lane line detection.

    Returns:
        The mode and the coefficients ``A, B, C`` of the two polynomials.
    """
    if fit.left is None or fit.right is None:
        return "no lane found"
    left = "A={:.2e} B={:.3f} C={:.1f}".format(*fit.left)
    right = "A={:.2e} B={:.3f} C={:.1f}".format(*fit.right)
    return f"{fit.mode.value:<16} left: {left}  right: {right}"


def main(argv: list[str] | None = None) -> None:
    """Fit the lane lines of every image and save the result.

    Args:
        argv: Argument list; ``sys.argv[1:]`` is used when ``None``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not CALIBRATION_PATH.exists():
        raise SystemExit(f"{CALIBRATION_PATH} not found. Run python3 scripts/calibrate.py first.")
    calibration = CameraCalibration.load(CALIBRATION_PATH)

    image_paths = args.images or sorted(DEFAULT_IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No images found in {DEFAULT_IMAGES_DIR}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Skipping %s: image could not be read", path)
            continue
        height, width = image.shape[:2]
        try:
            camera = camera_for_size(width, height)
        except ValueError as error:
            logger.warning("Skipping %s: %s", path, error)
            continue

        if (width, height) == calibration.image_size:
            image = calibration.undistort(image)
        transform = PerspectiveTransform.for_camera(camera)
        binary = transform.warp(threshold_lane_pixels(image), cv2.INTER_NEAREST)
        fit = fit_lane(binary)

        title = f"Lane fit: {fit.mode.value}"
        if not fit.is_plausible((width, height)):
            title += " (implausible)"
        comparison = side_by_side(
            [transform.warp(image), draw_lane_fit(binary, fit)],
            ["Bird's-eye view", title],
            PREVIEW_SCALE,
        )
        cv2.imwrite(str(args.output_dir / f"{path.stem}.png"), comparison)
        logger.info("%-22s %s", path.name, describe(fit))
        saved += 1
    logger.info("Saved %d result images to %s", saved, args.output_dir)


if __name__ == "__main__":
    main()
