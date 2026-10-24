"""Find the lane in road images and draw it on them together with its measurements.

Usage:
    python3 scripts/run_images.py [IMAGE ...] [--output-dir DIR]

Without arguments every image of resources/test_images_export is processed. In each result
the lane area is green, its left boundary red and its right boundary blue; the radius of
curvature, the position of the vehicle and the lane width are written at the top. The
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
# The scripts are in a subfolder: make lane_finding importable. The lane_finding imports
# therefore come after this line and not at the top of the file.
sys.path.insert(0, str(REPO_ROOT))

from lane_finding.calibration import CameraCalibration
from lane_finding.overlay import NO_LANE_TEXT, draw_lane_overlay
from lane_finding.pipeline import LanePipeline

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = REPO_ROOT / "resources" / "test_images_export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "images"
CALIBRATION_PATH = REPO_ROOT / "output" / "calibration.npz"


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


def main(argv: list[str] | None = None) -> None:
    """Find the lane in every image and save the image with the lane drawn on it.

    Args:
        argv: Argument list; ``sys.argv[1:]`` is used when ``None``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not CALIBRATION_PATH.exists():
        raise SystemExit(f"{CALIBRATION_PATH} not found. Run python3 scripts/calibrate.py first.")
    pipeline = LanePipeline(CameraCalibration.load(CALIBRATION_PATH))

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
        try:
            result = pipeline.process(image)
        except ValueError as error:
            logger.warning("Skipping %s: %s", path, error)
            continue

        overlay = draw_lane_overlay(result.image, result.transform, result.fit, result.measurements)
        cv2.imwrite(str(args.output_dir / f"{path.stem}.png"), overlay)
        lines = result.measurements.text_lines() if result.measurements else [NO_LANE_TEXT]
        logger.info("%-22s %s", path.name, " | ".join(lines))
        saved += 1
    logger.info("Saved %d result images to %s", saved, args.output_dir)


if __name__ == "__main__":
    main()
