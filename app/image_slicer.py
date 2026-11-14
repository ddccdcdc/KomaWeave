"""Smart gutter-aware vertical image slicer for ultra-long webtoon/manhwa strips."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def smart_slice_vertical(
    image_bgr: np.ndarray,
    target_height: int = 2000,
    min_height: int = 1000,
    max_height: int = 2600,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Slice an ultra-long vertical image into safe chunks at quiet horizontal gutters
    so text bubbles and panels are never cut in half.
    Returns list of (y_start, y_end, slice_bgr).
    """
    h, w = image_bgr.shape[:2]
    if h <= max_height:
        return [(0, h, image_bgr)]

    # Downsample width for O(H) ~2ms energy scan
    scale_w = min(w, 240)
    gray_small = cv2.cvtColor(
        cv2.resize(image_bgr, (scale_w, h), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    # Row-wise standard deviation + horizontal gradient energy
    row_std = gray_small.std(axis=1)
    gx = np.abs(np.diff(gray_small.astype(np.int16), axis=1)).mean(axis=1)
    energy = row_std + gx * 1.5

    # Smooth energy over a 15-pixel vertical window to prefer wide clean gutters
    kernel = np.ones(15, dtype=np.float32) / 15.0
    smooth_energy = np.convolve(energy, kernel, mode="same")

    slices: List[Tuple[int, int, np.ndarray]] = []
    cur_y = 0
    while cur_y < h:
        if h - cur_y <= max_height:
            slices.append((cur_y, h, image_bgr[cur_y:h]))
            break

        lo = cur_y + min_height
        hi = min(h - 1, cur_y + max_height)
        ideal = cur_y + target_height

        window = smooth_energy[lo : hi + 1]
        # Slight distance penalty so we stay close to target_height among equally flat gutters
        dists = np.abs(np.arange(lo, hi + 1) - ideal) * 0.005
        best_offset = int(np.argmin(window + dists))
        cut_y = lo + best_offset

        slices.append((cur_y, cut_y, image_bgr[cur_y:cut_y]))
        cur_y = cut_y

    return slices


def stitch_vertical(slices: List[np.ndarray]) -> np.ndarray:
    """Seamlessly stitch processed vertical slices back into a single image."""
    if len(slices) == 1:
        return slices[0]
    return np.vstack(slices)
