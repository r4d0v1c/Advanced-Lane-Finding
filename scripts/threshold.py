"""Create the binary lane-pixel image of road images and save it next to the original.

Usage:
    python3 scripts/threshold.py [IMAGE ...] [--output-dir DIR]

Without arguments every image of resources/test_images_export is processed. The
images are thresholded as they are; undistortion is part of the full pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # the scripts are in a subfolder: make lane_finding importable

from lane_finding.thresholding import threshold_lane_pixels  # noqa: E402
from lane_finding.visualization import side_by_side  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = REPO_ROOT / "resources" / "test_images_export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "threshold"

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
        help="where to save the original/binary comparison images",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Threshold the images and save an original/binary comparison of each.

    Args:
        argv: Argument list; ``sys.argv[1:]`` is used when ``None``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
        comparison = side_by_side(
            [image, threshold_lane_pixels(image)], ["Original", "Binary"], PREVIEW_SCALE
        )
        cv2.imwrite(str(args.output_dir / f"{path.stem}.png"), comparison)
        saved += 1
    logger.info("Saved %d comparison images to %s", saved, args.output_dir)


if __name__ == "__main__":
    main()
