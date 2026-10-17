"""Tests for lane_finding.visualization."""

import numpy as np
import pytest

from lane_finding.visualization import TITLE_COLOR, side_by_side


def test_side_by_side_combines_color_and_grayscale_panels():
    """Panels are resized and placed in one BGR row."""
    color = np.zeros((100, 160, 3), np.uint8)
    gray = np.zeros((100, 160), np.uint8)
    result = side_by_side([color, gray], ["Color", "Gray"], scale=0.5)
    assert result.shape == (50, 160, 3)
    assert result.dtype == np.uint8


def test_side_by_side_draws_the_titles():
    """Every panel contains title pixels."""
    black = np.zeros((100, 200, 3), np.uint8)
    result = side_by_side([black, black], ["A", "B"])
    for panel in (result[:, :200], result[:, 200:]):
        assert (panel == TITLE_COLOR).all(axis=-1).any()


def test_side_by_side_does_not_modify_its_input():
    """The input images are left untouched."""
    image = np.zeros((100, 200, 3), np.uint8)
    side_by_side([image], ["Title"])
    assert not image.any()


def test_side_by_side_rejects_mismatching_titles():
    """The number of titles must match the number of images."""
    with pytest.raises(ValueError, match="titles"):
        side_by_side([np.zeros((10, 10), np.uint8)], ["A", "B"])


def test_side_by_side_rejects_images_of_different_heights():
    """Images of different heights cannot be placed in one row."""
    with pytest.raises(ValueError, match="height"):
        side_by_side([np.zeros((10, 10), np.uint8), np.zeros((20, 10), np.uint8)], ["A", "B"])
