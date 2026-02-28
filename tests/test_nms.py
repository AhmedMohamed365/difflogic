"""
Tests for detector.postprocess: NMS, Soft-NMS, and Weighted Box Fusion.

Validates:
- NMS removes overlapping boxes and keeps the highest-score one.
- NMS on non-overlapping boxes keeps all.
- NMS is per-class (different classes are not suppressed together).
- Soft-NMS reduces scores of overlapping boxes rather than removing them.
- WBF merges overlapping boxes by weighted average.
- All functions return empty list for empty input.
- Deterministic output for fixed input.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from detector.postprocess import nms, soft_nms, weighted_box_fusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_det(x1, y1, x2, y2, class_id, score):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_id": class_id, "score": score}


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def test_nms_empty():
    assert nms([]) == []


def test_nms_single():
    dets = [make_det(0, 0, 10, 10, 1, 0.9)]
    result = nms(dets, iou_thresh=0.5)
    assert len(result) == 1


def test_nms_two_overlapping_same_class():
    """High-score box should survive; low-score overlapping box suppressed."""
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(1, 1, 11, 11, 1, 0.5),  # large overlap with first
    ]
    result = nms(dets, iou_thresh=0.5)
    assert len(result) == 1
    assert abs(result[0]["score"] - 0.9) < 1e-5


def test_nms_two_non_overlapping():
    """Non-overlapping boxes of the same class should both survive."""
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(50, 50, 60, 60, 1, 0.8),  # far away
    ]
    result = nms(dets, iou_thresh=0.5)
    assert len(result) == 2


def test_nms_different_classes_not_suppressed():
    """Overlapping boxes of different classes should both survive in per-class mode."""
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(0, 0, 10, 10, 2, 0.8),  # same location, different class
    ]
    result = nms(dets, iou_thresh=0.5, class_agnostic=False)
    assert len(result) == 2


def test_nms_class_agnostic_suppresses_across_classes():
    """In class-agnostic mode overlapping boxes from different classes are suppressed."""
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(0, 0, 10, 10, 2, 0.8),
    ]
    result = nms(dets, iou_thresh=0.5, class_agnostic=True)
    assert len(result) == 1


def test_nms_output_sorted_by_score():
    dets = [
        make_det(0, 0, 10, 10, 1, 0.6),
        make_det(50, 50, 60, 60, 1, 0.9),
        make_det(100, 100, 110, 110, 1, 0.75),
    ]
    result = nms(dets, iou_thresh=0.5)
    scores = [d["score"] for d in result]
    assert scores == sorted(scores, reverse=True)


def test_nms_deterministic():
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(1, 1, 11, 11, 1, 0.5),
        make_det(50, 50, 60, 60, 2, 0.8),
    ]
    r1 = nms(dets, iou_thresh=0.5)
    r2 = nms(dets, iou_thresh=0.5)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Soft-NMS
# ---------------------------------------------------------------------------

def test_soft_nms_empty():
    assert soft_nms([]) == []


def test_soft_nms_single():
    dets = [make_det(0, 0, 10, 10, 1, 0.9)]
    result = soft_nms(dets)
    assert len(result) == 1


def test_soft_nms_decays_overlapping_score():
    """Overlapping box should have a *lower* score after soft-NMS than before."""
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(1, 1, 11, 11, 1, 0.85),  # high overlap
    ]
    result = soft_nms(dets, iou_thresh=0.5, method="linear", score_thresh=0.0)
    # All boxes still present (none removed), second box score should be decayed
    assert len(result) == 2
    scores = sorted([d["score"] for d in result], reverse=True)
    assert scores[1] < 0.85  # decayed


def test_soft_nms_gaussian():
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(1, 1, 11, 11, 1, 0.85),
    ]
    result = soft_nms(dets, method="gaussian", score_thresh=0.0)
    assert len(result) == 2


def test_soft_nms_score_thresh_removes_boxes():
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(1, 1, 11, 11, 1, 0.85),
    ]
    result = soft_nms(dets, iou_thresh=0.5, method="linear", score_thresh=0.9)
    # The decayed box should have been removed
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Weighted Box Fusion
# ---------------------------------------------------------------------------

def test_wbf_empty():
    assert weighted_box_fusion([]) == []


def test_wbf_single():
    dets = [make_det(0, 0, 10, 10, 1, 0.9)]
    result = weighted_box_fusion(dets)
    assert len(result) == 1
    assert abs(result[0]["x1"] - 0) < 1e-3


def test_wbf_merges_overlapping():
    """Two overlapping boxes of the same class should be merged into one."""
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(2, 2, 12, 12, 1, 0.7),
    ]
    result = weighted_box_fusion(dets, iou_thresh=0.3)
    assert len(result) == 1


def test_wbf_merged_box_is_weighted_average():
    """The merged box coordinates should be a score-weighted average."""
    dets = [
        make_det(0.0, 0.0, 10.0, 10.0, 1, 1.0),
        make_det(0.0, 0.0, 10.0, 10.0, 1, 1.0),  # identical boxes → same average
    ]
    result = weighted_box_fusion(dets, iou_thresh=0.3)
    assert len(result) == 1
    assert abs(result[0]["x1"] - 0.0) < 1e-3
    assert abs(result[0]["x2"] - 10.0) < 1e-3


def test_wbf_keeps_non_overlapping():
    """Non-overlapping boxes should not be merged."""
    dets = [
        make_det(0, 0, 10, 10, 1, 0.9),
        make_det(100, 100, 110, 110, 1, 0.8),
    ]
    result = weighted_box_fusion(dets, iou_thresh=0.5)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Known toy case – NMS
# ---------------------------------------------------------------------------

def test_nms_known_case():
    """
    Three boxes. Box A and B overlap heavily; B and C do not.
    After NMS at iou_thresh=0.5:
      - A (0.9) kept, B (0.5) suppressed (overlaps A), C (0.8) kept.
    """
    A = make_det(0, 0, 100, 100, 1, 0.9)
    B = make_det(5, 5, 105, 105, 1, 0.5)   # large overlap with A
    C = make_det(200, 200, 300, 300, 1, 0.8)  # no overlap

    result = nms([A, B, C], iou_thresh=0.5)

    assert len(result) == 2
    assert any(abs(d["score"] - 0.9) < 1e-5 for d in result)
    assert any(abs(d["score"] - 0.8) < 1e-5 for d in result)
    assert all(abs(d["score"] - 0.5) > 1e-5 for d in result)
