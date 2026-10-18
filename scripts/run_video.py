"""Find the lane in videos and draw it on every frame together with its measurements.

Usage:
    python3 scripts/run_video.py [VIDEO ...] [--output-dir DIR]

Without arguments every video of resources/test_videos_export is processed frame by frame,
using only the frames seen so far, as a system in a car would: the horizon is estimated from
the frames so far (starting with the usual one of the camera), and a tracker follows the
lane from frame to frame: it smooths the lane,
shows the last lane (with a note) while the lane lines are not visible and ignores results
that jump. The colors are the ones of scripts/run_images.py. The camera is recognized by
the video resolution. Videos of the calibrated camera are undistorted first (run
scripts/calibrate.py before), videos of the other camera are used as they are.
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # the scripts are in a subfolder: make lane_finding importable

from lane_finding.calibration import CameraCalibration  # noqa: E402
from lane_finding.cameras import camera_for_size  # noqa: E402
from lane_finding.horizon import HorizonEstimator  # noqa: E402
from lane_finding.overlay import draw_lane_overlay  # noqa: E402
from lane_finding.pipeline import LanePipeline  # noqa: E402
from lane_finding.tracking import LaneTracker, TrackStatus  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_VIDEOS_DIR = REPO_ROOT / "resources" / "test_videos_export"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "videos"
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
        "videos",
        nargs="*",
        type=Path,
        help="videos to process (default: all test videos)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where to save the result videos",
    )
    return parser.parse_args(argv)


def process_video(path: Path, calibration: CameraCalibration, output_dir: Path) -> bool:
    """Draw the lane on every frame of a video and save the result.

    Args:
        path: The video.
        calibration: Calibration of the main camera.
        output_dir: Folder for the result video.

    Returns:
        ``True`` if the result was saved, ``False`` if the video was skipped.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        logger.warning("Skipping %s: video could not be opened", path)
        return False
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        camera = camera_for_size(width, height)
    except ValueError as error:
        logger.warning("Skipping %s: %s", path, error)
        return False

    logger.info("Processing %s ...", path.name)
    pipeline = LanePipeline(
        calibration, tracker=LaneTracker(), horizon_estimator=HorizonEstimator()
    )
    output_path = output_dir / f"{path.stem}.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not write {output_path}")
    counts = Counter()
    start = time.perf_counter()
    while True:
        found, frame = capture.read()
        if not found:
            break
        result = pipeline.process(frame)
        overlay = draw_lane_overlay(
            result.image, result.transform, result.fit, result.measurements, result.status
        )
        writer.write(overlay)
        counts[result.status] += 1
    seconds = time.perf_counter() - start
    capture.release()
    writer.release()

    frames = sum(counts.values())
    horizon = camera.horizon_y if pipeline.horizon_y is None else pipeline.horizon_y
    share = {status: 100 * counts[status] / max(frames, 1) for status in TrackStatus}
    logger.info(
        "%-20s final horizon %5.1f | %4d frames | found %5.1f %%  held %5.1f %%  lost %5.1f %% | "
        "%4.1f ms per frame",
        path.name,
        horizon,
        frames,
        share[TrackStatus.DETECTED],
        share[TrackStatus.HELD],
        share[TrackStatus.LOST],
        1000 * seconds / max(frames, 1),
    )
    return True


def main(argv: list[str] | None = None) -> None:
    """Process every video and save the result.

    Args:
        argv: Argument list; ``sys.argv[1:]`` is used when ``None``.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not CALIBRATION_PATH.exists():
        raise SystemExit(f"{CALIBRATION_PATH} not found. Run python3 scripts/calibrate.py first.")
    calibration = CameraCalibration.load(CALIBRATION_PATH)

    video_paths = args.videos or sorted(DEFAULT_VIDEOS_DIR.glob("*.mp4"))
    if not video_paths:
        raise SystemExit(f"No videos found in {DEFAULT_VIDEOS_DIR}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = sum(process_video(path, calibration, args.output_dir) for path in video_paths)
    logger.info("Saved %d result videos to %s", saved, args.output_dir)


if __name__ == "__main__":
    main()
