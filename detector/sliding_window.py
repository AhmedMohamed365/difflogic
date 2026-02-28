"""
SlidingWindowDetector – turns a difflogic classification model into an
object detector via sliding-window / tiling inference.

Usage example::

    from detector import SlidingWindowDetector, TorchBackend

    backend = TorchBackend(my_model, device="cpu")
    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(28, 28),
        classes=["background", "circle", "rectangle"],
        preprocess_fn=my_preprocess,
    )

    detections = detector.detect(
        image=img_array,
        scales=[0.25, 0.4, 0.6],
        stride=0.25,
        score_thresh=0.7,
    )
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .windows import WindowGenerator
from .postprocess import nms, soft_nms, weighted_box_fusion
from .refine import refine_detections, _crop_resize


Detection = Dict  # {"x1","y1","x2","y2","class_id","score"}


class SlidingWindowDetector:
    """Detection by dense sliding-window classification.

    Parameters
    ----------
    backend:
        A ``TorchBackend`` or ``CompiledBackend`` instance exposing a
        ``predict(batch_inputs) -> batch_probs`` method.
    input_size:
        ``(H, W)`` that the classifier expects.
    classes:
        List of class names. Index 0 is typically "background" and is **not**
        reported as a detection unless ``background_class`` is overridden to
        ``-1`` (no background).
    preprocess_fn:
        Callable that converts a raw numpy patch (H′×W′ or H′×W′×C) to a
        1-D float/bool numpy vector matching the classifier's expected input
        dimensionality.
    background_class:
        Class index that represents "no object".  Boxes where the top class is
        the background class are discarded.  Set to ``-1`` to report all classes.
    merge_method:
        ``"nms"`` | ``"soft_nms"`` | ``"wbf"``.
    nms_iou_thresh:
        IoU threshold for NMS / Soft-NMS / WBF.
    """

    def __init__(
        self,
        backend,
        input_size: Tuple[int, int],
        classes: List[str],
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        background_class: int = 0,
        merge_method: str = "nms",
        nms_iou_thresh: float = 0.5,
    ):
        self.backend = backend
        self.input_size = input_size  # (H, W)
        self.classes = classes
        self.preprocess_fn = preprocess_fn
        self.background_class = background_class
        self.merge_method = merge_method
        self.nms_iou_thresh = nms_iou_thresh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        scales: List[float],
        stride: Union[float, int] = 0.25,
        score_thresh: Union[float, Dict[int, float]] = 0.5,
        topk_per_class: Optional[int] = None,
        batch_size: int = 64,
        aspect_ratios: Optional[List[float]] = None,
        max_detections: Optional[int] = None,
        use_prefilter: bool = True,
        prefilter_var_thresh: float = 1e-3,
        coarse_to_fine: bool = False,
        coarse_stride_factor: float = 2.0,
        refine: bool = False,
    ) -> List[Detection]:
        """Run sliding-window detection on ``image``.

        Parameters
        ----------
        image:
            ``(H, W)`` or ``(H, W, C)`` uint8 or float32 numpy array.
        scales:
            Window sizes or relative scale factors (see :class:`WindowGenerator`).
        stride:
            Step between windows (fraction of window size or absolute pixels).
        score_thresh:
            Minimum score to keep a detection.  Either a single float (applied
            to all non-background classes) or a dict ``{class_id: threshold}``.
        topk_per_class:
            If set, keep at most this many detections per class *before* NMS.
        batch_size:
            Number of patches to classify in a single forward pass.
        aspect_ratios:
            Optional list of aspect ratios; defaults to ``[1.0]``.
        max_detections:
            Cap on the total number of returned detections after NMS.
        use_prefilter:
            Apply a variance-based early-rejection filter before classification.
        prefilter_var_thresh:
            Minimum normalised pixel variance for a patch to be classified.
        coarse_to_fine:
            First scan with a larger stride, then refine around high-score regions.
        coarse_stride_factor:
            Multiplier applied to ``stride`` for the coarse pass.
        refine:
            Apply :func:`~detector.refine.refine_detections` after NMS.

        Returns
        -------
        List[Detection]
            Each element is ``{x1,y1,x2,y2,class_id,score}``.
        """
        img_h, img_w = image.shape[:2]

        if coarse_to_fine:
            return self._coarse_to_fine(
                image, scales, stride, score_thresh, topk_per_class,
                batch_size, aspect_ratios, max_detections,
                use_prefilter, prefilter_var_thresh,
                coarse_stride_factor, refine,
            )

        win_gen = WindowGenerator(
            input_size=self.input_size,
            scales=scales,
            stride=stride,
            aspect_ratios=aspect_ratios,
        )
        windows = win_gen.generate(img_h, img_w)

        raw_dets = self._classify_windows(
            image, windows, score_thresh, topk_per_class,
            batch_size, use_prefilter, prefilter_var_thresh,
        )

        merged = self._merge(raw_dets)

        if refine and merged:
            merged = refine_detections(
                merged, image, self.backend, self.preprocess_fn, self.input_size
            )

        if max_detections is not None:
            merged = sorted(merged, key=lambda d: -d["score"])[:max_detections]

        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_windows(
        self,
        image: np.ndarray,
        windows: List[Tuple[int, int, int, int]],
        score_thresh: Union[float, Dict],
        topk_per_class: Optional[int],
        batch_size: int,
        use_prefilter: bool,
        prefilter_var_thresh: float,
    ) -> List[Detection]:
        """Crop, preprocess, classify all windows and return raw detections."""
        in_h, in_w = self.input_size
        raw_dets: List[Detection] = []

        # Split into batches
        for batch_start in range(0, len(windows), batch_size):
            batch_wins = windows[batch_start: batch_start + batch_size]
            patches = []
            valid_wins = []

            for win in batch_wins:
                x1, y1, x2, y2 = win
                patch = _crop_resize(image, x1, y1, x2, y2, in_h, in_w)

                # Early rejection: variance filter
                if use_prefilter and not _passes_variance_filter(
                    patch, prefilter_var_thresh
                ):
                    continue

                feat = self.preprocess_fn(patch)
                patches.append(feat.ravel())
                valid_wins.append(win)

            if not patches:
                continue

            batch_input = np.stack(patches, axis=0)
            scores = self.backend.predict(batch_input)  # (N, C)

            # Softmax-normalise if scores are raw logits (all non-negative → skip)
            # We just use raw scores as-is; callers should ensure they are probabilities
            # or at least monotone with confidence.

            for i, win in enumerate(valid_wins):
                x1, y1, x2, y2 = win
                probs = scores[i]  # (C,)

                pred_class = int(np.argmax(probs))
                if pred_class == self.background_class:
                    continue

                score = float(probs[pred_class])
                thresh = (
                    score_thresh.get(pred_class, 0.0)
                    if isinstance(score_thresh, dict)
                    else score_thresh
                )
                if score < thresh:
                    continue

                raw_dets.append({
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "class_id": pred_class,
                    "score": score,
                })

        # Optional: keep only topk per class
        if topk_per_class is not None:
            raw_dets = _topk(raw_dets, topk_per_class)

        return raw_dets

    def _merge(self, dets: List[Detection]) -> List[Detection]:
        if not dets:
            return []
        if self.merge_method == "nms":
            return nms(dets, iou_thresh=self.nms_iou_thresh)
        elif self.merge_method == "soft_nms":
            return soft_nms(dets, iou_thresh=self.nms_iou_thresh)
        elif self.merge_method == "wbf":
            return weighted_box_fusion(dets, iou_thresh=self.nms_iou_thresh)
        else:
            raise ValueError(f"Unknown merge_method: {self.merge_method!r}")

    def _coarse_to_fine(
        self,
        image: np.ndarray,
        scales: List[float],
        stride: Union[float, int],
        score_thresh: Union[float, Dict],
        topk_per_class: Optional[int],
        batch_size: int,
        aspect_ratios: Optional[List[float]],
        max_detections: Optional[int],
        use_prefilter: bool,
        prefilter_var_thresh: float,
        coarse_stride_factor: float,
        refine: bool,
    ) -> List[Detection]:
        """Two-pass coarse-to-fine scanning."""
        img_h, img_w = image.shape[:2]

        # --- Coarse pass ---
        coarse_stride = (
            stride * coarse_stride_factor
            if isinstance(stride, float)
            else int(stride * coarse_stride_factor)
        )
        win_gen_coarse = WindowGenerator(
            input_size=self.input_size,
            scales=scales,
            stride=coarse_stride,
            aspect_ratios=aspect_ratios,
        )
        coarse_windows = win_gen_coarse.generate(img_h, img_w)
        coarse_dets = self._classify_windows(
            image, coarse_windows, score_thresh, topk_per_class,
            batch_size, use_prefilter, prefilter_var_thresh,
        )

        if not coarse_dets:
            return []

        # --- Fine pass: refine around high-score coarse regions ---
        fine_windows = _expand_windows_around_dets(
            coarse_dets, img_h, img_w, expand=0.5
        )
        # Add fine-stride windows restricted to those regions
        fine_win_gen = WindowGenerator(
            input_size=self.input_size,
            scales=scales,
            stride=stride,
            aspect_ratios=aspect_ratios,
        )
        all_fine = fine_win_gen.generate(img_h, img_w)
        fine_in_region = _filter_windows_in_regions(all_fine, coarse_dets, expand=0.5)

        all_windows = list(set(coarse_windows) | set(fine_in_region))
        raw_dets = self._classify_windows(
            image, all_windows, score_thresh, topk_per_class,
            batch_size, use_prefilter, prefilter_var_thresh,
        )

        merged = self._merge(raw_dets)

        if refine and merged:
            merged = refine_detections(
                merged, image, self.backend, self.preprocess_fn, self.input_size
            )

        if max_detections is not None:
            merged = sorted(merged, key=lambda d: -d["score"])[:max_detections]

        return merged


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _passes_variance_filter(patch: np.ndarray, var_thresh: float) -> bool:
    """Return True if the patch has sufficient variance to be worth classifying."""
    f = patch.astype(np.float32)
    if f.max() > 1.0:
        f = f / 255.0
    return float(f.var()) >= var_thresh


def _topk(dets: List[Detection], k: int) -> List[Detection]:
    """Keep at most k detections per class sorted by score descending."""
    from collections import defaultdict
    buckets: Dict[int, List[Detection]] = defaultdict(list)
    for d in dets:
        buckets[d["class_id"]].append(d)
    result = []
    for cid, bucket in buckets.items():
        bucket.sort(key=lambda d: -d["score"])
        result.extend(bucket[:k])
    return result


def _expand_windows_around_dets(
    dets: List[Detection], img_h: int, img_w: int, expand: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """Return expanded boxes around each detection as a list of windows."""
    regions = []
    for d in dets:
        w = d["x2"] - d["x1"]
        h = d["y2"] - d["y1"]
        x1 = max(0, int(d["x1"] - expand * w))
        y1 = max(0, int(d["y1"] - expand * h))
        x2 = min(img_w, int(d["x2"] + expand * w))
        y2 = min(img_h, int(d["y2"] + expand * h))
        if x2 > x1 and y2 > y1:
            regions.append((x1, y1, x2, y2))
    return regions


def _filter_windows_in_regions(
    windows: List[Tuple[int, int, int, int]],
    dets: List[Detection],
    expand: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """Keep only windows that overlap with at least one detection region."""
    from .postprocess import _iou
    det_boxes = np.array(
        [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets], dtype=np.float32
    )
    result = []
    for win in windows:
        box = np.array(win, dtype=np.float32)
        for db in det_boxes:
            if _iou(box, db) > 0:
                result.append(win)
                break
    return result
