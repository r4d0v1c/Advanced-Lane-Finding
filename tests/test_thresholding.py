"""Tests for lane_finding.thresholding."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from lane_finding.thresholding import (
    WHITE_MIN_CONTRAST,
    WHITE_MIN_LIGHTNESS,
    remove_wide_runs,
    threshold_lane_pixels,
    white_mask,
    yellow_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "resources" / "test_images_export"

# BGR colors
ASPHALT = (60, 60, 60)
WHITE_PAINT = (235, 235, 235)
YELLOW_PAINT = (30, 190, 230)
SHADED_YELLOW_PAINT = (10, 95, 110)
CONCRETE = (190, 190, 190)
SKY_BLUE = (235, 180, 100)


def solid(color: tuple[int, int, int], height: int = 4, width: int = 4) -> np.ndarray:
    """Create a single-color BGR image.

    Args:
        color: BGR color.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        8-bit BGR image filled with the color.
    """
    return np.full((height, width, 3), color, np.uint8)


def synthetic_road() -> tuple[np.ndarray, np.ndarray]:
    """Draw a road image with known lane lines and distractors.

    The image is 400 pixels wide, so runs of 20 pixels or more are removed and a white
    pixel is compared with the pixels 8 pixels to its left and right.

    Returns:
        The BGR image and the binary image that should be produced from it: a yellow
        stripe (12 pixels wide) and a white stripe (6 pixels wide) are lane pixels, while
        a white and a yellow block (100 pixels wide), a concrete patch and a sky patch are
        not.
    """
    image = solid(ASPHALT, height=200, width=400)
    image[:, 100:112] = YELLOW_PAINT
    image[:, 300:306] = WHITE_PAINT
    image[20:60, 150:250] = WHITE_PAINT  # too wide for a lane line
    image[70:90, 150:250] = YELLOW_PAINT  # too wide for a lane line
    image[100:140, 150:250] = CONCRETE
    image[150:190, 150:250] = SKY_BLUE
    expected = np.zeros(image.shape[:2], np.uint8)
    expected[:, 100:112] = 255
    expected[:, 300:306] = 255
    return image, expected


def stripe(
    color: tuple[int, int, int], background: tuple[int, int, int] = ASPHALT, width: int = 6
) -> np.ndarray:
    """Draw a vertical stripe across a 400 pixels wide road image.

    Args:
        color: BGR color of the stripe.
        background: BGR color of the road.
        width: Width of the stripe in pixels; it starts at column 200.

    Returns:
        8-bit BGR image.
    """
    image = solid(background, height=10, width=400)
    image[:, 200 : 200 + width] = color
    return image


@pytest.mark.parametrize(
    "color, expected",
    [
        (YELLOW_PAINT, True),
        (SHADED_YELLOW_PAINT, True),
        ((40, 100, 200), False),  # orange
        ((40, 150, 60), False),  # green
        (SKY_BLUE, False),
        (CONCRETE, False),
        (WHITE_PAINT, False),
        (ASPHALT, False),
        ((0, 60, 70), False),  # too dark to be a visible yellow line
    ],
)
def test_yellow_mask_selects_only_yellow(color, expected):
    """Only yellow hues with enough saturation and brightness are selected."""
    assert yellow_mask(solid(color)).any() == expected


def test_white_mask_selects_a_light_stripe_from_the_minimum_lightness_upwards():
    """A narrow gray stripe on asphalt is white exactly from the minimum lightness upwards."""
    assert white_mask(stripe((WHITE_MIN_LIGHTNESS,) * 3))[:, 200:206].all()
    assert not white_mask(stripe((WHITE_MIN_LIGHTNESS - 1,) * 3)).any()


def test_white_mask_needs_a_darker_road_on_both_sides():
    """A stripe is white only if it is lighter than the road by the minimum contrast."""
    road = (200, 200, 200)
    lighter = (200 + WHITE_MIN_CONTRAST,) * 3
    slightly_lighter = (199 + WHITE_MIN_CONTRAST,) * 3
    assert white_mask(stripe(lighter, background=road))[:, 200:206].all()
    assert not white_mask(stripe(slightly_lighter, background=road)).any()


def test_white_mask_ignores_wide_bright_areas():
    """Sunlit asphalt or concrete is bright, but it has no darker road on both sides."""
    assert not white_mask(stripe(WHITE_PAINT, width=100)).any()
    assert not white_mask(solid(WHITE_PAINT, height=10, width=400)).any()


def test_white_mask_keeps_only_the_middle_of_a_wider_line():
    """A pixel near the edge of a line wider than the comparison distance sees the line."""
    selected = np.flatnonzero(white_mask(stripe(WHITE_PAINT, width=12))[0])
    assert selected.tolist() == [204, 205, 206, 207]


def test_white_mask_keeps_a_line_in_the_shade():
    """A white line in a shadow is far less bright, but still lighter than the road."""
    assert white_mask(stripe((170, 170, 170), background=(50, 50, 50)))[:, 200:206].all()


def test_white_mask_ignores_bright_strips_at_the_image_borders():
    """Outside the image counts as white, so a bright strip cut by a border is no line."""
    image = solid(ASPHALT, height=10, width=400)
    image[:, :5] = WHITE_PAINT
    image[:, 395:] = WHITE_PAINT
    assert not white_mask(image).any()


@pytest.mark.parametrize("color", [YELLOW_PAINT, ASPHALT, (40, 150, 60)])
def test_white_mask_ignores_dark_and_colored_stripes(color):
    """Yellow paint, asphalt and green are too dark to be a white line."""
    assert not white_mask(stripe(color)).any()


@pytest.mark.parametrize("max_width", [30, 31])
def test_remove_wide_runs_keeps_only_narrow_features(max_width):
    """Runs of the maximum width or wider are removed, narrower ones are kept (even and odd)."""
    mask = np.zeros((10, 200), np.uint8)
    mask[:, 5:15] = 255  # 10 pixels wide
    mask[:, 30 : 30 + max_width - 1] = 255  # one pixel narrower than the limit
    mask[:, 90 : 90 + max_width] = 255  # exactly the limit
    mask[:, 150:200] = 255  # wider, touching the right border
    expected = np.zeros_like(mask)
    expected[:, 5:15] = 255
    expected[:, 30 : 30 + max_width - 1] = 255
    assert np.array_equal(remove_wide_runs(mask, max_width), expected)


def test_remove_wide_runs_keeps_narrow_runs_at_the_image_border():
    """A narrow run cut off by the image border is not mistaken for a wide one."""
    mask = np.zeros((10, 100), np.uint8)
    mask[:, :10] = 255
    mask[:, 90:] = 255
    assert np.array_equal(remove_wide_runs(mask, max_width=30), mask)


def test_remove_wide_runs_does_not_modify_its_input():
    """The input mask is left untouched."""
    mask = np.full((5, 40), 255, np.uint8)
    remove_wide_runs(mask, max_width=10)
    assert mask.all()


def test_threshold_lane_pixels_on_synthetic_road():
    """Lane stripes are found, while wide white or yellow areas, concrete and sky are not."""
    image, expected = synthetic_road()
    assert np.array_equal(threshold_lane_pixels(image), expected)


def test_threshold_lane_pixels_removes_areas_wider_than_a_line():
    """Lines cross a row with at most about 3 % of the width; 6 % is something else."""
    assert threshold_lane_pixels(stripe(YELLOW_PAINT, width=12)).any()  # 3 % of 400 pixels
    assert not threshold_lane_pixels(stripe(YELLOW_PAINT, width=24)).any()  # 6 %


def test_threshold_lane_pixels_returns_binary_uint8_image():
    """The result is a single-channel uint8 image with the values 0 and 255 only."""
    image = np.random.default_rng(0).integers(0, 256, (60, 90, 3), dtype=np.uint8)
    result = threshold_lane_pixels(image)
    assert result.shape == (60, 90)
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 255}


def test_threshold_lane_pixels_does_not_modify_its_input():
    """The input image is left untouched."""
    image, _ = synthetic_road()
    original = image.copy()
    threshold_lane_pixels(image)
    assert np.array_equal(image, original)


@pytest.mark.parametrize("function", [yellow_mask, white_mask, threshold_lane_pixels])
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
def test_functions_reject_images_that_are_not_8bit_bgr(function, bad_image):
    """Invalid input raises a clear error instead of a failure inside OpenCV."""
    with pytest.raises(ValueError, match="8-bit BGR"):
        function(bad_image)


@pytest.mark.parametrize(
    "name, yellow_columns, white_columns",
    [
        ("straight_lines1", (300, 320), (995, 1000)),
        ("straight_lines2", (310, 322), (996, 1010)),
    ],
)
def test_lane_pixels_lie_on_the_lines_and_not_on_the_asphalt(name, yellow_columns, white_columns):
    """On a straight road the lines are detected and the asphalt between them is not."""
    row = threshold_lane_pixels(cv2.imread(str(IMAGES_DIR / f"{name}.jpg")))[650]
    assert row[slice(*yellow_columns)].all()
    assert row[slice(*white_columns)].all()
    assert not row[400:900].any()


def test_works_for_other_resolutions():
    """Images of another resolution are processed as well (960x540 camera)."""
    image = cv2.imread(str(IMAGES_DIR / "solidYellowCurve.jpg"))
    result = threshold_lane_pixels(image)
    assert image.shape[:2] == (540, 960)
    assert result.shape == (540, 960)
    assert result.any()


def test_dry_grass_next_to_the_road_is_removed():
    """Yellowish grass is selected by the color mask but removed by the width filter."""
    image = cv2.imread(str(IMAGES_DIR / "straight_lines2.jpg"))
    height, width = image.shape[:2]
    shoulder = (slice(int(0.55 * height), int(0.66 * height)), slice(int(0.8 * width), width))
    unfiltered = cv2.bitwise_or(yellow_mask(image), white_mask(image))[shoulder]
    filtered = threshold_lane_pixels(image)[shoulder]
    assert np.count_nonzero(unfiltered) > 0.95 * unfiltered.size
    assert np.count_nonzero(filtered) < 0.02 * filtered.size


@pytest.mark.parametrize("seconds", [32.0, 44.0])
def test_sunlit_asphalt_is_not_selected_as_white(seconds, read_video_frame):
    """Of the many bright pixels of sunlit asphalt (challenge video) few are taken as white."""
    frame = read_video_frame("challenge03.mp4", seconds)
    lower_half = slice(frame.shape[0] // 2, None)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    bright = cv2.inRange(hls, (0, 200, 0), (179, 255, 255))[lower_half]
    white = white_mask(frame)[lower_half]
    assert np.count_nonzero(white) < 0.25 * np.count_nonzero(bright)
    assert np.count_nonzero(threshold_lane_pixels(frame)[lower_half]) < 0.1 * white.size
