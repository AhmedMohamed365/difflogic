"""
Inference-time bounding-box refinement via multi-window snapping.

For each candidate detection, small perturbations around the box coordinates
and scale are tried.  The variant that scores highest is returned as the
refined box.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Union

import numpy as np


Detection = Dict  # {"x1","y1","x2","y2","class_id","score"}


def refine_detections(
    detections: List[Detection],
    image: np.ndarray,
    backend,
    preprocess_fn: Callable[[np.ndarray], np.ndarray],
    input_size: Tuple[int, int],
    offsets: Tuple[float, ...] = (-0.05, 0.0, 0.05),
    scale_factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
    max_candidates: int = 9,
) -> List[Detection]:
    """Refine bounding boxes by testing small perturbations around each detection.

    For each detection the function:

    1. Generates a grid of candidate boxes by shifting and scaling the
       original box.
    2. Runs the classifier on each candidate patch.
    3. Keeps the candidate with the highest score for the predicted class.

    Parameters
    ----------
    detections:
        List of detection dicts ``{x1,y1,x2,y2,class_id,score}``.
    image:
        Original image as ``(H, W)`` or ``(H, W, C)`` uint8 / float32 numpy array.
    backend:
        A ``TorchBackend`` or ``CompiledBackend`` instance.
    preprocess_fn:
        Callable that converts a raw patch (numpy array) to a flat 1-D numpy
        array ready for the backend.
    input_size:
        ``(H, W)`` expected by the classifier.
    offsets:
        Fractional offsets (relative to box size) to apply in x and y directions.
    scale_factors:
        Multiplicative scale factors to apply to the window size.
    max_candidates:
        Maximum number of offset × scale combinations to evaluate per detection.
        Truncated from the grid when needed.

    Returns
    -------
    List[Detection]
        Refined detections (same length as input).
    """
    if not detections:
        return detections

    img_h, img_w = image.shape[:2]
    in_h, in_w = input_size
    refined: List[Detection] = []

    for det in detections:
        bx1, by1, bx2, by2 = det["x1"], det["y1"], det["x2"], det["y2"]
        bw = bx2 - bx1
        bh = by2 - by1
        cid = det["class_id"]
        best_score = det["score"]
        best_box = (bx1, by1, bx2, by2)

        # Build candidate list
        candidates: List[Tuple[float, float, float, float]] = []
        for sf in scale_factors:
            new_w = bw * sf
            new_h = bh * sf
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            for dy_frac in offsets:
                for dx_frac in offsets:
                    ncx = cx + dx_frac * bw
                    ncy = cy + dy_frac * bh
                    nx1 = ncx - new_w / 2
                    ny1 = ncy - new_h / 2
                    nx2 = ncx + new_w / 2
                    ny2 = ncy + new_h / 2
                    # clip
                    nx1 = max(0.0, nx1)
                    ny1 = max(0.0, ny1)
                    nx2 = min(float(img_w), nx2)
                    ny2 = min(float(img_h), ny2)
                    if nx2 > nx1 and ny2 > ny1:
                        candidates.append((nx1, ny1, nx2, ny2))

        # Limit candidates
        candidates = candidates[:max_candidates]

        if not candidates:
            refined.append(det)
            continue

        # Crop, resize and classify all candidates at once
        patches = []
        for (cx1, cy1, cx2, cy2) in candidates:
            patch = _crop_resize(image, cx1, cy1, cx2, cy2, in_h, in_w)
            patches.append(preprocess_fn(patch))

        batch = np.stack(patches, axis=0)
        scores_all = backend.predict(batch)  # (N_cands, num_classes)

        for k, (cx1, cy1, cx2, cy2) in enumerate(candidates):
            s = float(scores_all[k, cid])
            if s > best_score:
                best_score = s
                best_box = (cx1, cy1, cx2, cy2)

        refined.append({
            "x1": best_box[0],
            "y1": best_box[1],
            "x2": best_box[2],
            "y2": best_box[3],
            "class_id": cid,
            "score": best_score,
        })

    return refined


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _crop_resize(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Crop ``image`` to ``[x1,y1,x2,y2]`` and resize to ``(out_h, out_w)``."""
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))
    ix1 = max(0, ix1)
    iy1 = max(0, iy1)
    ix2 = min(image.shape[1], ix2)
    iy2 = min(image.shape[0], iy2)

    patch = image[iy1:iy2, ix1:ix2]
    if patch.size == 0:
        return np.zeros((out_h, out_w) + image.shape[2:], dtype=image.dtype)

    try:
        from PIL import Image as PILImage  # type: ignore

        pil = PILImage.fromarray(patch if patch.ndim == 3 else patch)
        pil = pil.resize((out_w, out_h), PILImage.BILINEAR)
        return np.array(pil)
    except ImportError:
        # Fall back to simple nearest-neighbour resize with numpy
        y_idx = np.linspace(0, patch.shape[0] - 1, out_h).astype(int)
        x_idx = np.linspace(0, patch.shape[1] - 1, out_w).astype(int)
        return patch[np.ix_(y_idx, x_idx)]
