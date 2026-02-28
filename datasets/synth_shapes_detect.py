"""
Synthetic shapes dataset for the sliding-window detection demo.

Each image contains one or more randomly placed geometric shapes (circles,
rectangles, triangles) on a uniform background.  Ground-truth bounding boxes
are returned alongside the image.

Usage::

    dataset = SynthShapesDataset(num_images=200, img_size=(128, 128))
    img, boxes, labels = dataset[0]
    # img: (H, W, 3) uint8
    # boxes: list of (x1, y1, x2, y2) ints
    # labels: list of class_id ints  (1=circle, 2=rectangle, 3=triangle)

Standalone image generator::

    img, boxes, labels = generate_synth_image(
        img_size=(128, 128),
        num_objects=3,
        rng=np.random.default_rng(42),
    )
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

# Class IDs
BACKGROUND = 0
CIRCLE = 1
RECTANGLE = 2
TRIANGLE = 3

CLASS_NAMES = ["background", "circle", "rectangle", "triangle"]


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_synth_image(
    img_size: Tuple[int, int] = (128, 128),
    num_objects: int = 3,
    min_shape_frac: float = 0.10,
    max_shape_frac: float = 0.30,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[int]]:
    """Generate a synthetic image with random shapes.

    Parameters
    ----------
    img_size:
        ``(H, W)`` of the generated image.
    num_objects:
        Number of shapes to place.
    min_shape_frac / max_shape_frac:
        Min / max shape size as a fraction of the shorter image side.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    image : np.ndarray of shape (H, W, 3), dtype uint8
    boxes : list of (x1, y1, x2, y2)
    labels : list of class_id (CIRCLE/RECTANGLE/TRIANGLE)
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = img_size
    # Random background colour
    bg = rng.integers(200, 256, size=3).astype(np.uint8)
    image = np.ones((h, w, 3), dtype=np.uint8) * bg

    boxes: List[Tuple[int, int, int, int]] = []
    labels: List[int] = []

    short_side = min(h, w)

    for _ in range(num_objects):
        shape_type = rng.integers(1, 4)  # 1, 2, 3
        size = int(rng.uniform(min_shape_frac, max_shape_frac) * short_side)
        size = max(6, size)

        # Pick a random top-left so the shape fits
        max_x = w - size - 1
        max_y = h - size - 1
        if max_x < 0 or max_y < 0:
            continue
        x1 = int(rng.integers(0, max_x + 1))
        y1 = int(rng.integers(0, max_y + 1))
        x2 = x1 + size
        y2 = y1 + size

        colour = rng.integers(30, 180, size=3).astype(np.uint8).tolist()

        if shape_type == CIRCLE:
            _draw_circle(image, x1, y1, x2, y2, colour)
        elif shape_type == RECTANGLE:
            _draw_rectangle(image, x1, y1, x2, y2, colour)
        elif shape_type == TRIANGLE:
            _draw_triangle(image, x1, y1, x2, y2, colour)

        boxes.append((x1, y1, x2, y2))
        labels.append(int(shape_type))

    return image, boxes, labels


# ---------------------------------------------------------------------------
# Drawing helpers (pure numpy, no external deps)
# ---------------------------------------------------------------------------

def _draw_circle(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int, colour: List[int]
) -> None:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    r = (x2 - x1) / 2.0
    ys, xs = np.ogrid[y1:y2, x1:x2]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
    image[y1:y2, x1:x2][mask] = colour


def _draw_rectangle(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int, colour: List[int]
) -> None:
    image[y1:y2, x1:x2] = colour


def _draw_triangle(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int, colour: List[int]
) -> None:
    """Draw an upward-pointing triangle inside the bounding box."""
    h = y2 - y1
    w = x2 - x1
    for row in range(h):
        frac = row / max(h - 1, 1)
        half_w = int(frac * w / 2)
        left = x1 + (w // 2) - half_w
        right = min(x2 - 1, x1 + (w // 2) + half_w)
        image[y2 - row - 1, left:right + 1] = colour


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SynthShapesDataset:
    """A synthetic dataset of images with geometric shapes.

    Parameters
    ----------
    num_images:
        Total number of images.
    img_size:
        ``(H, W)`` of each image.
    num_objects_range:
        ``(min, max)`` number of shapes per image.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_images: int = 200,
        img_size: Tuple[int, int] = (128, 128),
        num_objects_range: Tuple[int, int] = (1, 4),
        seed: int = 0,
    ):
        self.num_images = num_images
        self.img_size = img_size
        self.num_objects_range = num_objects_range
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[int]]:
        if idx not in self._cache:
            rng = np.random.default_rng(self.seed + idx)
            n_obj = int(rng.integers(*self.num_objects_range))
            img, boxes, labels = generate_synth_image(
                img_size=self.img_size,
                num_objects=n_obj,
                rng=rng,
            )
            self._cache[idx] = (img, boxes, labels)
        return self._cache[idx]

    def class_names(self) -> List[str]:
        return CLASS_NAMES
