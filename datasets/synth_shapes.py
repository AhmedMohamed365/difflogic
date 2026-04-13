"""
Synthetic shapes dataset for YOLOv1-style training.

Generates greyscale images containing a single shape (rectangle, circle, or
triangle) placed randomly.  Returns the image plus a YOLO-formatted target
tensor.

YOLOTarget layout (per-cell):
    [obj(1) | class_one_hot(C) | x_bin_one_hot(Q) | y_bin_one_hot(Q) |
     w_bin_one_hot(Q) | h_bin_one_hot(Q)]

Only the cell whose centre is closest to the object centre has obj=1 and
non-zero coordinate targets; all other cells have obj=0 and zero targets.
"""

import math
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Low-level shape drawing (no PIL dependency)
# ---------------------------------------------------------------------------

def _draw_rectangle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    fill: float) -> None:
    img[y1:y2, x1:x2] = fill


def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int,
                 fill: float) -> None:
    H, W = img.shape
    y_grid, x_grid = np.ogrid[:H, :W]
    mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= r ** 2
    img[mask] = fill


def _draw_triangle(img: np.ndarray, cx: int, cy: int, half_size: int,
                   fill: float) -> None:
    H, W = img.shape
    for y in range(max(0, cy - half_size), min(H, cy + half_size)):
        # equilateral-like triangle pointing up
        row_half = int(half_size * (1 - abs(y - cy) / half_size))
        x_lo = max(0, cx - row_half)
        x_hi = min(W, cx + row_half)
        img[y, x_lo:x_hi] = fill


# ---------------------------------------------------------------------------
# YOLO target builder
# ---------------------------------------------------------------------------

class YOLOTargetBuilder:
    """
    Converts a single ground-truth bounding box (xywh normalised to [0,1])
    into a dense YOLO target tensor of shape [S, S, 1 + C + 4*Q].

    The cell whose centre contains the object centre receives:
        - objectness = 1
        - one-hot class label
        - one-hot bin indices for (x, y, w, h) relative to cell / image

    All other cells have objectness = 0 and the rest is zeros.

    Args:
        S: grid size.
        C: number of classes.
        Q: quantisation bins per coordinate.
    """

    def __init__(self, S: int, C: int, Q: int):
        self.S = S
        self.C = C
        self.Q = Q
        self.K = 1 + C + 4 * Q

    def _bin_index(self, value: float) -> int:
        """Map a float in [0, 1) to a bin index in [0, Q)."""
        idx = int(value * self.Q)
        return min(idx, self.Q - 1)

    def build(self, class_id: int,
              bbox_xywh_norm: Tuple[float, float, float, float]) -> torch.Tensor:
        """
        Args:
            class_id: integer class label in [0, C).
            bbox_xywh_norm: (cx, cy, w, h) all normalised to [0, 1].
        Returns:
            target: float32 tensor of shape [S, S, K].
        """
        target = torch.zeros(self.S, self.S, self.K, dtype=torch.float32)

        cx, cy, bw, bh = bbox_xywh_norm

        # Identify responsible cell
        cell_x = int(cx * self.S)
        cell_y = int(cy * self.S)
        cell_x = min(cell_x, self.S - 1)
        cell_y = min(cell_y, self.S - 1)

        # Cell-relative x, y (offset from cell top-left, scaled to [0,1])
        x_rel = cx * self.S - cell_x
        y_rel = cy * self.S - cell_y

        # objectness
        target[cell_y, cell_x, 0] = 1.0

        # class one-hot
        target[cell_y, cell_x, 1 + class_id] = 1.0

        # coord bins (one-hot)
        offset = 1 + self.C
        for coord_val, coord_idx in [
            (x_rel, 0),
            (y_rel, 1),
            (bw,    2),
            (bh,    3),
        ]:
            bin_i = self._bin_index(float(coord_val))
            target[cell_y, cell_x, offset + coord_idx * self.Q + bin_i] = 1.0

        return target


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_SHAPE_RECT = 0
_SHAPE_CIRCLE = 1
_SHAPE_TRIANGLE = 2

_SHAPE_NAMES = ['rectangle', 'circle', 'triangle']


class SynthShapesDataset(Dataset):
    """
    Generates synthetic images containing exactly one shape.

    Args:
        num_samples : number of samples in the dataset.
        img_size    : (H, W) spatial size of each image.
        S           : YOLO grid size.
        C           : number of classes (≤ 3; shapes are rectangle/circle/triangle).
        Q           : YOLO quantisation bins per coordinate.
        min_rel_size: minimum object size as fraction of image (default 0.1).
        max_rel_size: maximum object size as fraction of image (default 0.4).
        seed        : optional random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        img_size: Tuple[int, int] = (32, 32),
        S: int = 8,
        C: int = 3,
        Q: int = 16,
        min_rel_size: float = 0.1,
        max_rel_size: float = 0.4,
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.img_h, self.img_w = img_size
        self.S = S
        self.C = C
        self.Q = Q
        self.min_rel_size = min_rel_size
        self.max_rel_size = max_rel_size

        assert C <= 3, 'SynthShapesDataset supports at most 3 classes.'

        self.target_builder = YOLOTargetBuilder(S, C, Q)

        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        self._samples = [
            self._generate(rng, np_rng) for _ in range(num_samples)
        ]

    def _generate(self, rng: random.Random, np_rng: np.random.Generator):
        H, W = self.img_h, self.img_w
        img = np.zeros((H, W), dtype=np.float32)

        # Random shape type (restricted to C classes)
        class_id = rng.randint(0, self.C - 1)

        # Random size
        rel_size = rng.uniform(self.min_rel_size, self.max_rel_size)
        half_size = max(2, int(rel_size * min(H, W) / 2))

        # Random centre (keep shape within image)
        cx_px = rng.randint(half_size, W - half_size - 1)
        cy_px = rng.randint(half_size, H - half_size - 1)

        fill = rng.uniform(0.5, 1.0)

        if class_id == _SHAPE_RECT:
            x1, y1 = cx_px - half_size, cy_px - half_size
            x2, y2 = cx_px + half_size, cy_px + half_size
            _draw_rectangle(img, x1, y1, x2, y2, fill)
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
        elif class_id == _SHAPE_CIRCLE:
            _draw_circle(img, cx_px, cy_px, half_size, fill)
            bw = 2 * half_size / W
            bh = 2 * half_size / H
        else:  # triangle
            _draw_triangle(img, cx_px, cy_px, half_size, fill)
            bw = 2 * half_size / W
            bh = 2 * half_size / H

        cx_norm = cx_px / W
        cy_norm = cy_px / H

        # Convert to torch tensor [1, H, W]
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Build YOLO target
        target = self.target_builder.build(class_id, (cx_norm, cy_norm, bw, bh))

        # Store also raw annotation for visualisation
        annotation = {
            'class_id': class_id,
            'cx_norm': cx_norm,
            'cy_norm': cy_norm,
            'bw': bw,
            'bh': bh,
        }

        return img_tensor, target, annotation

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        img, target, _ = self._samples[idx]
        return img, target

    def get_annotation(self, idx) -> dict:
        """Return raw annotation dict for the sample at *idx*."""
        return self._samples[idx][2]
