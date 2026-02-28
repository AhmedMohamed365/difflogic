"""
Post-processing for detection: NMS, Soft-NMS, and Weighted Box Fusion.

All functions accept and return detections as a list of dicts with keys::

    {"x1": float, "y1": float, "x2": float, "y2": float,
     "class_id": int, "score": float}
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


Detection = Dict  # {"x1","y1","x2","y2","class_id","score"}


# ---------------------------------------------------------------------------
# Helper – IoU
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes ``[x1, y1, x2, y2]``."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def _to_arrays(dets: List[Detection]):
    """Convert list of detection dicts to numpy arrays."""
    if not dets:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32)
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets], dtype=np.float32)
    scores = np.array([d["score"] for d in dets], dtype=np.float32)
    class_ids = np.array([d["class_id"] for d in dets], dtype=np.int32)
    return boxes, scores, class_ids


def _from_arrays(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
) -> List[Detection]:
    return [
        {"x1": float(boxes[i, 0]), "y1": float(boxes[i, 1]),
         "x2": float(boxes[i, 2]), "y2": float(boxes[i, 3]),
         "class_id": int(class_ids[i]), "score": float(scores[i])}
        for i in range(len(scores))
    ]


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(
    dets: List[Detection],
    iou_thresh: float = 0.5,
    class_agnostic: bool = False,
) -> List[Detection]:
    """Standard greedy Non-Maximum Suppression.

    Parameters
    ----------
    dets:
        List of detection dicts.
    iou_thresh:
        Overlapping boxes with IoU above this threshold are suppressed.
    class_agnostic:
        If ``True``, suppress across all classes; otherwise per-class.

    Returns
    -------
    List[Detection]
        Filtered detections, sorted by descending score.
    """
    if not dets:
        return []

    boxes, scores, class_ids = _to_arrays(dets)
    keep = _nms_cpu(boxes, scores, class_ids, iou_thresh, class_agnostic)
    return _from_arrays(boxes[keep], scores[keep], class_ids[keep])


def _nms_cpu(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thresh: float,
    class_agnostic: bool,
) -> List[int]:
    """Returns kept indices (sorted by score descending)."""
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        for jdx in order:
            if suppressed[jdx] or jdx == idx:
                continue
            if not class_agnostic and class_ids[idx] != class_ids[jdx]:
                continue
            if _iou(boxes[idx], boxes[jdx]) > iou_thresh:
                suppressed[jdx] = True

    return keep


# ---------------------------------------------------------------------------
# Soft-NMS
# ---------------------------------------------------------------------------

def soft_nms(
    dets: List[Detection],
    iou_thresh: float = 0.5,
    method: str = "linear",
    sigma: float = 0.5,
    score_thresh: float = 0.001,
    class_agnostic: bool = False,
) -> List[Detection]:
    """Soft-NMS – decays scores of overlapping boxes rather than removing them.

    Parameters
    ----------
    dets:
        List of detection dicts.
    iou_thresh:
        IoU threshold for the *linear* method (not used by Gaussian).
    method:
        ``"linear"`` or ``"gaussian"``.
    sigma:
        Bandwidth for the Gaussian method.
    score_thresh:
        Boxes with decayed score below this are removed.
    class_agnostic:
        If ``True``, suppress across all classes; otherwise per-class.

    Returns
    -------
    List[Detection]
        Filtered detections.
    """
    if not dets:
        return []

    boxes, scores, class_ids = _to_arrays(dets)
    scores = scores.copy()

    n = len(scores)
    for i in range(n):
        # Find the un-processed detection with the highest score
        best = i + np.argmax(scores[i:])
        # Swap it to position i
        boxes[[i, best]] = boxes[[best, i]]
        scores[[i, best]] = scores[[best, i]]
        class_ids[[i, best]] = class_ids[[best, i]]

        for j in range(i + 1, n):
            if not class_agnostic and class_ids[i] != class_ids[j]:
                continue
            iou = _iou(boxes[i], boxes[j])
            if method == "linear":
                if iou > iou_thresh:
                    scores[j] *= 1.0 - iou
            elif method == "gaussian":
                scores[j] *= np.exp(-(iou ** 2) / sigma)
            else:
                raise ValueError(f"Unknown soft_nms method: {method!r}")

    mask = scores >= score_thresh
    return _from_arrays(boxes[mask], scores[mask], class_ids[mask])


# ---------------------------------------------------------------------------
# Weighted Box Fusion
# ---------------------------------------------------------------------------

def weighted_box_fusion(
    dets: List[Detection],
    iou_thresh: float = 0.55,
    class_agnostic: bool = False,
) -> List[Detection]:
    """Weighted Box Fusion (WBF) – merges overlapping boxes by score-weighted average.

    Reference: Solovyev et al., "Weighted boxes fusion: Ensembling boxes from
    different object detection models", Image and Vision Computing, 2021.

    Parameters
    ----------
    dets:
        List of detection dicts.
    iou_thresh:
        Boxes with IoU above this are merged into a cluster.
    class_agnostic:
        If ``True``, merge across all classes; otherwise per-class.

    Returns
    -------
    List[Detection]
        Fused detections.
    """
    if not dets:
        return []

    boxes, scores, class_ids = _to_arrays(dets)

    # Sort by descending score
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    class_ids = class_ids[order]

    used = np.zeros(len(scores), dtype=bool)
    result: List[Detection] = []

    for i in range(len(scores)):
        if used[i]:
            continue
        # Start a new cluster
        cluster_boxes = [boxes[i]]
        cluster_scores = [scores[i]]
        cluster_class = class_ids[i]
        used[i] = True

        for j in range(i + 1, len(scores)):
            if used[j]:
                continue
            if not class_agnostic and class_ids[j] != cluster_class:
                continue
            # Compute IoU against the *current fused box*
            fused = _weighted_mean(cluster_boxes, cluster_scores)
            if _iou(fused, boxes[j]) >= iou_thresh:
                cluster_boxes.append(boxes[j])
                cluster_scores.append(scores[j])
                used[j] = True

        fused_box = _weighted_mean(cluster_boxes, cluster_scores)
        fused_score = float(np.mean(cluster_scores))
        result.append({
            "x1": float(fused_box[0]),
            "y1": float(fused_box[1]),
            "x2": float(fused_box[2]),
            "y2": float(fused_box[3]),
            "class_id": int(cluster_class),
            "score": fused_score,
        })

    return result


def _weighted_mean(
    boxes: List[np.ndarray], scores: List[float]
) -> np.ndarray:
    w = np.array(scores, dtype=np.float64)
    w = w / w.sum()
    stacked = np.stack(boxes, axis=0)
    return (stacked * w[:, None]).sum(axis=0)
