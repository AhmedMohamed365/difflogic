"""
PatchSampler – build a patch classification dataset from a detection dataset.

Given a detection dataset (image + GT boxes), this class samples:
- Positive patches: crops around GT boxes with random jitter.
- Negative patches: random crops with IoU < ``neg_iou_thresh`` to any GT box.

The result can be fed directly to a difflogic classifier.

Usage::

    sampler = PatchSampler(
        detection_dataset=my_det_dataset,  # items: (image, boxes, labels)
        patch_size=(28, 28),
        positives_per_image=4,
        negatives_per_image=8,
    )
    patches, labels = sampler.build()
    # patches: (N, H*W*C)  float32 in [0, 1]
    # labels:  (N,)        int
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .synth_shapes_detect import BACKGROUND


class PatchSampler:
    """Sample positive and negative patches from a detection dataset.

    Parameters
    ----------
    detection_dataset:
        Iterable of ``(image, boxes, labels)`` tuples where ``image`` is a
        numpy array, ``boxes`` is a list of ``(x1,y1,x2,y2)`` tuples, and
        ``labels`` is a list of integer class IDs.
    patch_size:
        ``(H, W)`` of the output patches.
    positives_per_image:
        Number of positive patches to sample per image.
    negatives_per_image:
        Number of negative patches to sample per image.
    jitter_frac:
        Maximum fractional jitter applied to GT box coordinates.
    neg_iou_thresh:
        Maximum IoU a random crop may have with any GT box to count as negative.
    preprocess_fn:
        Optional callable that converts a raw patch array to a flat vector.
        If ``None``, patches are resized to ``patch_size`` and flattened.
    seed:
        Random seed.
    """

    def __init__(
        self,
        detection_dataset,
        patch_size: Tuple[int, int] = (28, 28),
        positives_per_image: int = 4,
        negatives_per_image: int = 8,
        jitter_frac: float = 0.1,
        neg_iou_thresh: float = 0.1,
        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: int = 0,
    ):
        self.detection_dataset = detection_dataset
        self.patch_size = patch_size
        self.positives_per_image = positives_per_image
        self.negatives_per_image = negatives_per_image
        self.jitter_frac = jitter_frac
        self.neg_iou_thresh = neg_iou_thresh
        self.preprocess_fn = preprocess_fn
        self.seed = seed

    # ------------------------------------------------------------------

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the patch dataset.

        Returns
        -------
        patches : np.ndarray of shape ``(N, D)`` float32
        labels  : np.ndarray of shape ``(N,)`` int32
        """
        rng = np.random.default_rng(self.seed)
        all_patches: List[np.ndarray] = []
        all_labels: List[int] = []

        for item in self.detection_dataset:
            image, boxes, labels = item
            img_h, img_w = image.shape[:2]

            # --- Positive patches ---
            for box, label in zip(boxes, labels):
                for _ in range(self.positives_per_image):
                    patch = self._jitter_crop(image, box, img_h, img_w, rng)
                    all_patches.append(self._encode(patch))
                    all_labels.append(label)

            # --- Negative patches ---
            attempts = 0
            neg_count = 0
            max_attempts = self.negatives_per_image * 20
            while neg_count < self.negatives_per_image and attempts < max_attempts:
                attempts += 1
                patch, ok = self._random_negative(
                    image, boxes, img_h, img_w, rng
                )
                if ok:
                    all_patches.append(self._encode(patch))
                    all_labels.append(BACKGROUND)
                    neg_count += 1

        if not all_patches:
            return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.int32)

        patches = np.stack(all_patches, axis=0)
        labels_arr = np.array(all_labels, dtype=np.int32)
        return patches, labels_arr

    # ------------------------------------------------------------------

    def _encode(self, patch: np.ndarray) -> np.ndarray:
        """Resize and flatten a patch to a 1-D float32 vector."""
        resized = _resize_patch(patch, self.patch_size)
        if self.preprocess_fn is not None:
            return self.preprocess_fn(resized).ravel().astype(np.float32)
        f = resized.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        return f.ravel()

    def _jitter_crop(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        img_h: int,
        img_w: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        dx = int(rng.uniform(-self.jitter_frac, self.jitter_frac) * w)
        dy = int(rng.uniform(-self.jitter_frac, self.jitter_frac) * h)
        nx1 = max(0, x1 + dx)
        ny1 = max(0, y1 + dy)
        nx2 = min(img_w, x2 + dx)
        ny2 = min(img_h, y2 + dy)
        if nx2 <= nx1:
            nx2 = nx1 + 1
        if ny2 <= ny1:
            ny2 = ny1 + 1
        return image[ny1:ny2, nx1:nx2]

    def _random_negative(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        img_h: int,
        img_w: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, bool]:
        ph, pw = self.patch_size
        if img_w <= pw or img_h <= ph:
            return image, False
        x1 = int(rng.integers(0, img_w - pw))
        y1 = int(rng.integers(0, img_h - ph))
        x2 = x1 + pw
        y2 = y1 + ph
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        for gt in boxes:
            if _iou(box, np.array(gt, dtype=np.float32)) >= self.neg_iou_thresh:
                return image, False
        return image[y1:y2, x1:x2], True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _resize_patch(patch: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize ``patch`` to ``size=(H, W)`` using nearest-neighbour."""
    out_h, out_w = size
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.zeros((out_h, out_w) + patch.shape[2:], dtype=patch.dtype)
    try:
        from PIL import Image as PILImage  # type: ignore

        mode = "RGB" if patch.ndim == 3 and patch.shape[2] == 3 else "L"
        if patch.ndim == 2:
            pil = PILImage.fromarray(patch)
        else:
            pil = PILImage.fromarray(patch)
        pil = pil.resize((out_w, out_h), PILImage.BILINEAR)
        return np.array(pil)
    except ImportError:
        y_idx = np.linspace(0, patch.shape[0] - 1, out_h).astype(int)
        x_idx = np.linspace(0, patch.shape[1] - 1, out_w).astype(int)
        return patch[np.ix_(y_idx, x_idx)]
