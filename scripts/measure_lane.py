"""Measure the curvature of the lane and the position of the vehicle in road images.

Usage:
    python3 scripts/measure_lane.py [IMAGE ...] [--output-dir DIR]

Without arguments every image of resources/test_images_export is processed. The numbers are
printed and written on the fitted lane in the bird's-eye view (see scripts/fit_lines.py for
the colors). The camera is recognized by the image resolution. Images of the calibrated
camera are undistorted first (run scripts/calibrate.py before), images of the other camera
are used as they are.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # the scripts are in a subfolder: make lane_finding importable

from lane_finding.calibration import CameraCalibration  # noqa: E402
from lane_finding.overlay import NO_LANE_TEXT  # noqa: E402
from lane_finding.pipeline import LanePipeline  # noqa: E402
from lane_finding.visualization import (  # noqa: E402
    draw_lane_fit,
    draw_text_lines,
    side_by_side,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = REPO_ROOT / "resources" / "test_images_export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "measure"
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


def main(argv: list[str] | None = None) -> None:
    """Measure the lane of every image and save the result.

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

        height, width = result.image.shape[:2]
        measured = result.measurements
        lines = measured.text_lines() if measured else [NO_LANE_TEXT]
        panel = draw_text_lines(
            draw_lane_fit(result.binary, result.fit),
            lines,
            (int(0.27 * width), int(0.3 * height)),
            scale=1.3 * width / 1280,
        )
        comparison = side_by_side(
            [result.transform.warp(result.image), panel],
            ["Bird's-eye view", "Measurements"],
            PREVIEW_SCALE,
        )
        cv2.imwrite(str(args.output_dir / f"{path.stem}.png"), comparison)
        logger.info("%-22s %s", path.name, " | ".join(lines))
        saved += 1
    logger.info("Saved %d result images to %s", saved, args.output_dir)


if __name__ == "__main__":
    main()
