"""
Window generator for sliding-window detection.

Generates ``(x1, y1, x2, y2)`` candidate windows at multiple scales and
optional aspect ratios over an image.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


class WindowGenerator:
    """Generate candidate bounding-box windows for sliding-window detection.

    Parameters
    ----------
    input_size:
        ``(H, W)`` – the patch size expected by the classifier.
    scales:
        List of window sizes (in pixels) *or* relative scale factors with
        respect to the shortest image side.  If values are <= 1.0 they are
        treated as relative; otherwise as absolute pixel sizes.
    stride:
        Stride between windows.  If <= 1.0 it is treated as a fraction of the
        window size at each scale; otherwise as an absolute pixel value.
    aspect_ratios:
        Optional list of aspect ratios (width/height) to apply to each scale.
        Default is ``[1.0]`` (square windows).
    min_size:
        Minimum window side length in pixels (after scale is applied).  Windows
        smaller than this are skipped.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        scales: List[float],
        stride: Union[float, int] = 0.25,
        aspect_ratios: Optional[List[float]] = None,
        min_size: int = 4,
    ):
        self.input_size = input_size  # (H, W)
        self.scales = scales
        self.stride = stride
        self.aspect_ratios = aspect_ratios if aspect_ratios is not None else [1.0]
        self.min_size = min_size

    # ------------------------------------------------------------------
    def _resolve_window_size(
        self, scale: float, img_h: int, img_w: int
    ) -> Tuple[int, int]:
        """Return (win_h, win_w) for a given scale value."""
        short_side = min(img_h, img_w)
        if scale <= 1.0:
            base = int(round(scale * short_side))
        else:
            base = int(round(scale))
        return base, base

    # ------------------------------------------------------------------
    def generate(
        self, img_h: int, img_w: int
    ) -> List[Tuple[int, int, int, int]]:
        """Return a list of ``(x1, y1, x2, y2)`` windows.

        Coordinates are clipped to the image boundaries.

        Parameters
        ----------
        img_h:
            Image height in pixels.
        img_w:
            Image width in pixels.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            Unique list of ``(x1, y1, x2, y2)`` windows (may overlap).
        """
        windows: List[Tuple[int, int, int, int]] = []
        seen = set()

        for scale in self.scales:
            base_h, base_w = self._resolve_window_size(scale, img_h, img_w)

            for ar in self.aspect_ratios:
                # ar = width / height → width = ar * height
                win_h = base_h
                win_w = int(round(base_w * ar))

                if win_h < self.min_size or win_w < self.min_size:
                    continue
                if win_h > img_h or win_w > img_w:
                    continue

                # Stride
                if self.stride <= 1.0:
                    stride_y = max(1, int(round(self.stride * win_h)))
                    stride_x = max(1, int(round(self.stride * win_w)))
                else:
                    stride_y = int(self.stride)
                    stride_x = int(self.stride)

                y1 = 0
                last_y1 = -1
                while y1 + win_h <= img_h:
                    x1 = 0
                    last_x1 = -1
                    while x1 + win_w <= img_w:
                        box = (x1, y1, x1 + win_w, y1 + win_h)
                        if box not in seen:
                            seen.add(box)
                            windows.append(box)
                        last_x1 = x1
                        x1 += stride_x
                    # ensure the rightmost column is always included
                    if last_x1 + win_w < img_w:
                        box = (img_w - win_w, y1, img_w, y1 + win_h)
                        if box not in seen:
                            seen.add(box)
                            windows.append(box)
                    last_y1 = y1
                    y1 += stride_y
                # ensure the bottom row is always included
                if last_y1 + win_h < img_h:
                    x1 = 0
                    while x1 + win_w <= img_w:
                        box = (x1, img_h - win_h, x1 + win_w, img_h)
                        if box not in seen:
                            seen.add(box)
                            windows.append(box)
                        x1 += stride_x

        return windows

    # ------------------------------------------------------------------
    def generate_multiscale(
        self, img_h: int, img_w: int
    ) -> List[Tuple[int, int, int, int]]:
        """Alias for :meth:`generate` – returns all windows across all scales."""
        return self.generate(img_h, img_w)
