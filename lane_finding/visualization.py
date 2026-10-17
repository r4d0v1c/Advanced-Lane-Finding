"""Helpers for inspecting intermediate results of the pipeline."""

from collections.abc import Sequence

import cv2
import numpy as np

TITLE_COLOR = (0, 0, 255)  # BGR red


def side_by_side(
    panels: Sequence[np.ndarray], titles: Sequence[str], scale: float = 1.0
) -> np.ndarray:
    """Place images in a row and label each one with a title.

    Args:
        panels: 8-bit images of the same height, grayscale or BGR.
        titles: One title per image.
        scale: Factor by which every image is resized before it is placed.

    Returns:
        BGR image containing all panels next to each other.

    Raises:
        ValueError: If the number of titles differs from the number of images or if
            the images do not have the same height.
    """
    if len(panels) != len(titles):
        raise ValueError(f"Got {len(panels)} images but {len(titles)} titles")
    if len({panel.shape[0] for panel in panels}) != 1:
        raise ValueError("All images must have the same height")

    labelled = []
    for panel, title in zip(panels, titles):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(panel, title, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, TITLE_COLOR, 3)
        labelled.append(panel)
    return np.hstack(labelled)
