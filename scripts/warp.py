"""Show the bird's-eye view of road images.

Usage:
    python3 scripts/warp.py [IMAGE ...] [--output-dir DIR]

Without arguments every image of resources/test_images_export is processed. For each
image the result shows the source trapezoid, the bird's-eye view and the binary
lane-pixel image in the bird's-eye view. The camera is recognized by the image
resolution. Images of the calibrated camera are undistorted first (run
scripts/calibrate.py before), images of the other camera are used as they are.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # the scripts are in a subfolder: make lane_finding importable

from lane_finding.calibration import CameraCalibration  # noqa: E402
from lane_finding.cameras import camera_for_size  # noqa: E402
from lane_finding.perspective import PerspectiveTransform  # noqa: E402
from lane_finding.thresholding import threshold_lane_pixels  # noqa: E402
from lane_finding.visualization import side_by_side  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = REPO_ROOT / "resources" / "test_images_export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "warp"
CALIBRATION_PATH = REPO_ROOT / "output" / "calibration.npz"

PREVIEW_SCALE = 0.5
TRAPEZOID_COLOR = (0, 0, 255)  # BGR red


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
        help="where to save the comparison images",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Create the bird's-eye comparison of every image and save it.

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
        marked = image.copy()
        trapezoid = np.round(transform.source_points).astype(np.int32)
        cv2.polylines(marked, [trapezoid], True, TRAPEZOID_COLOR, 3)
        comparison = side_by_side(
            [
                marked,
                transform.warp(image),
                transform.warp(threshold_lane_pixels(image), cv2.INTER_NEAREST),
            ],
            ["Source points", "Bird's-eye view", "Binary"],
            PREVIEW_SCALE,
        )
        cv2.imwrite(str(args.output_dir / f"{path.stem}.png"), comparison)
        saved += 1
    logger.info("Saved %d comparison images to %s", saved, args.output_dir)


if __name__ == "__main__":
    main()
