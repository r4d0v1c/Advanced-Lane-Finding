"""Tests for lane_finding.thresholding."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_finding.cameras import LANE_WIDTH_METERS, MAIN_CAMERA, SECOND_CAMERA
from lane_finding.thresholding import (
    COMPARE_DISTANCE_MARKINGS,
    MARKING_WIDTH_METERS,
    compare_distances,
    threshold_lane_pixels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "resources" / "test_images_export"

# BGR colors
ASPHALT = (90, 90, 90)
WHITE_PAINT = (225, 225, 225)
YELLOW_PAINT = (40, 185, 215)
DARK_SHADOW = (40, 40, 40)

WIDTH, HEIGHT = 400, 40
# A lane of 148 pixels gives a compare distance of 12 pixels: 2 * 0.15 / 3.7 * 148 = 12.
LANE = 148.0
DISTANCE = 12


def lane_widths(height: int = HEIGHT, lane: float = LANE) -> np.ndarray:
    """Give every row of a synthetic image the same lane width.

    Args:
        height: Number of rows.
        lane: Lane width in pixels.

    Returns:
        One lane width per row.
    """
    return np.full(height, lane)


def road(color: tuple[int, int, int] = ASPHALT, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Create a synthetic road image.

    Args:
        color: BGR color of the road.
        noise: Standard deviation of the grey-level noise of the surface.
        seed: Seed of the noise.

    Returns:
        8-bit BGR image of ``HEIGHT`` x ``WIDTH`` pixels.
    """
    image = np.full((HEIGHT, WIDTH, 3), color, np.float64)
    if noise:
        image += np.random.default_rng(seed).normal(0, noise, (HEIGHT, WIDTH, 1))
    return np.clip(image, 0, 255).astype(np.uint8)


def with_stripe(image: np.ndarray, color: tuple[int, int, int], start: int, width: int):
    """Paint a vertical stripe across an image.

    Args:
        image: BGR image (it is copied).
        color: BGR color of the stripe.
        start: First column of the stripe.
        width: Width of the stripe in pixels.

    Returns:
        The image with the stripe.
    """
    image = image.copy()
    image[:, start : start + width] = color
    return image


def selected_columns(mask: np.ndarray) -> list[int]:
    """List the columns selected in the middle row of a mask.

    Args:
        mask: Binary mask.

    Returns:
        The selected columns.
    """
    return np.flatnonzero(mask[HEIGHT // 2]).tolist()


# ------------------------------------------------------------------ compare distances


def test_the_compare_distance_is_twice_a_marking_at_the_scale_of_the_lane():
    """A marking is 0.15 m of a 3.7 m lane; the road is looked at two markings away."""
    widths = np.array([0.0, 10.0, 148.0, 740.0])
    scale = COMPARE_DISTANCE_MARKINGS * MARKING_WIDTH_METERS / LANE_WIDTH_METERS
    expected = np.rint(scale * widths)
    assert compare_distances(widths).tolist() == expected.astype(int).tolist()
    assert compare_distances(widths)[2] == DISTANCE


# ------------------------------------------------------------------- synthetic roads


@pytest.mark.parametrize("color", [WHITE_PAINT, YELLOW_PAINT], ids=["white", "yellow"])
def test_a_marking_on_the_road_is_selected(color):
    """A painted stripe lighter or yellower than the road is selected, the road is not."""
    image = with_stripe(road(), color, 200, 8)

    mask = threshold_lane_pixels(image, lane_widths())

    assert selected_columns(mask) == list(range(200, 208))


def test_a_yellow_marking_is_found_by_its_colour_alone():
    """Yellow paint as light as the road still stands out in the blue-yellow channel."""
    asphalt = (130, 150, 160)  # a slightly warm grey road
    yellow = (50, 148, 172)  # as light, but much more yellow
    image = with_stripe(road(asphalt), yellow, 200, 8)
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 0]
    assert abs(int(lightness[0, 204]) - int(lightness[0, 0])) <= 1

    assert selected_columns(threshold_lane_pixels(image, lane_widths())) == list(range(200, 208))


def test_wide_bright_areas_are_not_selected():
    """Sunlit asphalt or a white car has no darker road on both sides of its pixels."""
    image = with_stripe(road(), WHITE_PAINT, 150, 100)
    assert not threshold_lane_pixels(image, lane_widths()).any()


def test_only_the_middle_of_a_wider_marking_is_kept():
    """Pixels closer to the edge than the compare distance see the marking on one side."""
    image = with_stripe(road(), WHITE_PAINT, 200, 18)  # wider than the distance of 12

    assert selected_columns(threshold_lane_pixels(image, lane_widths())) == list(range(206, 212))


def test_a_dark_stripe_is_not_selected():
    """The shadow of a pole is darker than the road, not lighter."""
    image = with_stripe(road(), DARK_SHADOW, 200, 8)
    assert not threshold_lane_pixels(image, lane_widths()).any()


def test_bright_strips_at_the_image_borders_are_not_selected():
    """Outside the image counts as the brightest value, so a strip cut by a border is no line."""
    image = with_stripe(with_stripe(road(), WHITE_PAINT, 0, 5), WHITE_PAINT, WIDTH - 5, 5)
    assert not threshold_lane_pixels(image, lane_widths()).any()


def test_rows_without_road_stay_empty():
    """Above the horizon and on the car hood (lane width 0) nothing is selected."""
    image = with_stripe(road(), WHITE_PAINT, 200, 8)
    widths = lane_widths()
    widths[: HEIGHT // 2] = 0.0

    mask = threshold_lane_pixels(image, widths)

    assert not mask[: HEIGHT // 2].any()
    assert mask[HEIGHT // 2 :, 200:208].all()


def test_the_same_markings_are_found_in_darker_or_paler_images():
    """No fixed brightness: a darker image or one with less contrast gives the same mask."""
    image = with_stripe(with_stripe(road(), WHITE_PAINT, 100, 8), YELLOW_PAINT, 300, 8)
    darker = (image * 0.5).astype(np.uint8)
    paler = (128 + (image.astype(np.float64) - 128) * 0.4).astype(np.uint8)

    expected = threshold_lane_pixels(image, lane_widths())

    assert np.array_equal(threshold_lane_pixels(darker, lane_widths()), expected)
    assert np.array_equal(threshold_lane_pixels(paler, lane_widths()), expected)


def test_the_threshold_follows_the_noise_of_the_image():
    """A faint stripe stands out on a smooth road but not on a rough one."""
    faint = (102, 102, 102)  # 12 grey levels lighter than the road
    smooth = with_stripe(road(), faint, 200, 8)
    rough = with_stripe(road(noise=8.0), faint, 200, 8)
    strong = with_stripe(road(noise=8.0), WHITE_PAINT, 200, 8)

    assert threshold_lane_pixels(smooth, lane_widths())[:, 200:208].all()
    assert threshold_lane_pixels(rough, lane_widths())[:, 200:208].mean() < 0.2
    assert threshold_lane_pixels(strong, lane_widths())[:, 200:208].all()


def test_threshold_lane_pixels_returns_a_binary_uint8_image():
    """The result is a single-channel uint8 image with the values 0 and 255 only."""
    image = np.random.default_rng(0).integers(0, 256, (60, 90, 3), dtype=np.uint8)
    result = threshold_lane_pixels(image, lane_widths(60))
    assert result.shape == (60, 90)
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 255}


def test_threshold_lane_pixels_does_not_modify_its_input():
    """The input image is left untouched."""
    image = with_stripe(road(), WHITE_PAINT, 200, 8)
    original = image.copy()
    threshold_lane_pixels(image, lane_widths())
    assert np.array_equal(image, original)


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((10, 10), np.uint8),  # grayscale
        np.zeros((10, 10, 4), np.uint8),  # four channels
        np.zeros((10, 10, 3), np.float32),  # not 8-bit
        np.zeros(10, np.uint8),  # not an image
    ],
    ids=["grayscale", "four-channels", "float", "1d"],
)
def test_images_that_are_not_8bit_bgr_are_rejected(bad_image):
    """Invalid input raises a clear error instead of a failure inside OpenCV."""
    with pytest.raises(ValueError, match="8-bit BGR"):
        threshold_lane_pixels(bad_image, lane_widths(10))


def test_one_lane_width_per_row_is_needed():
    """The lane widths have to match the image height."""
    with pytest.raises(ValueError, match="one lane width per image row"):
        threshold_lane_pixels(road(), lane_widths(HEIGHT - 1))


# ---------------------------------------------------------------------- real images


@pytest.mark.parametrize("name", ["straight_lines1", "straight_lines2"])
def test_the_lines_are_selected_and_the_asphalt_between_them_is_not(name, calibration):
    """On a straight road both lines are found in the near rows, hardly anything between them."""
    image = calibration.undistort(cv2.imread(str(IMAGES_DIR / f"{name}.jpg")))
    near = threshold_lane_pixels(image, MAIN_CAMERA.lane_widths())[600:668]

    assert near[:, 250:420].any()  # the yellow line on the left
    assert near[:, 950:1050].any()  # a dash of the white line on the right
    # Only scattered specks of the asphalt texture pass the three-sigma test.
    between = near[:, 420:900]
    assert np.count_nonzero(between) < 0.01 * between.size


def test_the_second_camera_has_its_own_geometry():
    """Images of the 960x540 camera are processed with their lane widths."""
    image = cv2.imread(str(IMAGES_DIR / "solidYellowCurve.jpg"))
    result = threshold_lane_pixels(image, SECOND_CAMERA.lane_widths())
    assert result.shape == (540, 960)
    assert result.any()


def test_dry_grass_next_to_the_road_is_not_selected():
    """Yellowish grass on the shoulder is not a narrow stripe."""
    image = cv2.imread(str(IMAGES_DIR / "straight_lines2.jpg"))
    height, width = image.shape[:2]
    shoulder = (slice(int(0.55 * height), int(0.66 * height)), slice(int(0.8 * width), width))
    selected = threshold_lane_pixels(image, MAIN_CAMERA.lane_widths())[shoulder]
    assert np.count_nonzero(selected) < 0.05 * selected.size


@pytest.mark.parametrize("seconds", [32.0, 44.0])
def test_little_of_the_sunlit_asphalt_is_selected(seconds, read_video_frame):
    """Of the many bright pixels of sunlit asphalt (challenge video) few are taken."""
    frame = read_video_frame("challenge03.mp4", seconds)
    lower_half = slice(frame.shape[0] // 2, None)
    lightness = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:, :, 1]
    bright = np.count_nonzero(lightness[lower_half] >= 200)
    selected = np.count_nonzero(threshold_lane_pixels(frame, MAIN_CAMERA.lane_widths())[lower_half])
    assert selected < 0.3 * bright


@pytest.mark.parametrize(
    "video, seconds", [("challenge02.mp4", 8), ("project_video03.mp4", 24), ("challenge01.mp4", 3)]
)
def test_real_frames_give_nearly_the_same_mask_when_darker_or_paler(
    video, seconds, calibration, read_video_frame
):
    """Without any change of parameters, darker or paler versions of a frame keep the lines."""
    frame = calibration.undistort(read_video_frame(video, seconds))
    widths = MAIN_CAMERA.lane_widths()
    original = threshold_lane_pixels(frame, widths) > 0
    for changed in (frame * 0.55, 128 + (frame.astype(np.float64) - 128) * 0.5):
        mask = threshold_lane_pixels(np.clip(changed, 0, 255).astype(np.uint8), widths) > 0
        overlap = np.count_nonzero(original & mask) / np.count_nonzero(original | mask)
        assert overlap > 0.75
