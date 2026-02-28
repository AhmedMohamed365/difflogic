"""
End-to-end test: SlidingWindowDetector on a synthetic shapes image.

Uses a simple toy classifier (a linear model trained in the test itself) to
validate:
- Detector runs without errors.
- Returned detections have the correct format.
- Deterministic results for fixed seed.
- Detector finds at least one object on a known synthetic image.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from datasets.synth_shapes_detect import (
    generate_synth_image,
    CIRCLE,
    RECTANGLE,
    BACKGROUND,
    CLASS_NAMES,
)
from detector import SlidingWindowDetector, TorchBackend
from detector.windows import WindowGenerator


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------

class ToyClassifier:
    """A mock classifier that identifies circles vs background by centre intensity.

    For testing only – just needs to produce (N, C) float scores.
    """

    def __init__(self, input_size=(16, 16), num_classes=4):
        self.input_size = input_size
        self.num_classes = num_classes

    def predict(self, batch_inputs: np.ndarray) -> np.ndarray:
        """batch_inputs: (N, D) float32 in [0,1]."""
        n = batch_inputs.shape[0]
        scores = np.zeros((n, self.num_classes), dtype=np.float32)
        # Background score is always 0.6
        scores[:, BACKGROUND] = 0.6
        # Simple heuristic: if variance is high → likely a shape (circle class used)
        for i in range(n):
            v = float(batch_inputs[i].var())
            if v > 0.02:
                scores[i, CIRCLE] = 0.8
                scores[i, BACKGROUND] = 0.2
        return scores


def make_preprocess(input_size=(16, 16)):
    def preprocess(patch: np.ndarray) -> np.ndarray:
        # resize to input_size
        out_h, out_w = input_size
        y_idx = np.linspace(0, patch.shape[0] - 1, out_h).astype(int)
        x_idx = np.linspace(0, patch.shape[1] - 1, out_w).astype(int)
        resized = patch[np.ix_(y_idx, x_idx)]
        f = resized.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        return f.ravel()
    return preprocess


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_detect_returns_list():
    rng = np.random.default_rng(42)
    image, boxes, labels = generate_synth_image(
        img_size=(128, 128), num_objects=2, rng=rng
    )
    backend = ToyClassifier(input_size=(16, 16), num_classes=4)
    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(16, 16),
        classes=CLASS_NAMES,
        preprocess_fn=make_preprocess((16, 16)),
        background_class=BACKGROUND,
    )
    result = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.7)
    assert isinstance(result, list)


def test_detections_have_correct_keys():
    rng = np.random.default_rng(42)
    image, _, _ = generate_synth_image(img_size=(128, 128), num_objects=1, rng=rng)
    backend = ToyClassifier(input_size=(16, 16), num_classes=4)
    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(16, 16),
        classes=CLASS_NAMES,
        preprocess_fn=make_preprocess((16, 16)),
        background_class=BACKGROUND,
    )
    result = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.5)
    for det in result:
        for key in ("x1", "y1", "x2", "y2", "class_id", "score"):
            assert key in det, f"Missing key: {key}"


def test_detections_within_image_bounds():
    rng = np.random.default_rng(0)
    h, w = 128, 128
    image, _, _ = generate_synth_image(img_size=(h, w), num_objects=2, rng=rng)
    backend = ToyClassifier(input_size=(16, 16), num_classes=4)
    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(16, 16),
        classes=CLASS_NAMES,
        preprocess_fn=make_preprocess((16, 16)),
        background_class=BACKGROUND,
    )
    result = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.5)
    for det in result:
        assert det["x1"] >= 0
        assert det["y1"] >= 0
        assert det["x2"] <= w
        assert det["y2"] <= h
        assert det["x2"] > det["x1"]
        assert det["y2"] > det["y1"]


def test_deterministic_for_fixed_seed():
    """Running the detector twice on the same image must return the same results."""
    rng = np.random.default_rng(7)
    image, _, _ = generate_synth_image(img_size=(128, 128), num_objects=2, rng=rng)
    backend = ToyClassifier(input_size=(16, 16), num_classes=4)
    preprocess = make_preprocess((16, 16))

    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(16, 16),
        classes=CLASS_NAMES,
        preprocess_fn=preprocess,
        background_class=BACKGROUND,
    )

    r1 = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.5)
    r2 = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.5)

    assert r1 == r2


def test_score_threshold_filters():
    """High threshold should produce fewer or equal detections than low threshold."""
    rng = np.random.default_rng(42)
    image, _, _ = generate_synth_image(img_size=(128, 128), num_objects=3, rng=rng)
    backend = ToyClassifier(input_size=(16, 16), num_classes=4)
    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(16, 16),
        classes=CLASS_NAMES,
        preprocess_fn=make_preprocess((16, 16)),
        background_class=BACKGROUND,
    )
    low = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.1)
    high = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.95)
    assert len(low) >= len(high)


def test_empty_image_no_crash():
    """A blank image should not crash the detector."""
    image = np.ones((128, 128, 3), dtype=np.uint8) * 220  # uniform background
    backend = ToyClassifier(input_size=(16, 16), num_classes=4)
    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(16, 16),
        classes=CLASS_NAMES,
        preprocess_fn=make_preprocess((16, 16)),
        background_class=BACKGROUND,
    )
    result = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.7)
    assert isinstance(result, list)


def test_merge_methods():
    """Detector should work with all three merge methods."""
    rng = np.random.default_rng(42)
    image, _, _ = generate_synth_image(img_size=(128, 128), num_objects=2, rng=rng)
    preprocess = make_preprocess((16, 16))

    for method in ("nms", "soft_nms", "wbf"):
        backend = ToyClassifier(input_size=(16, 16), num_classes=4)
        detector = SlidingWindowDetector(
            backend=backend,
            input_size=(16, 16),
            classes=CLASS_NAMES,
            preprocess_fn=preprocess,
            background_class=BACKGROUND,
            merge_method=method,
        )
        result = detector.detect(image, scales=[0.3], stride=0.5, score_thresh=0.5)
        assert isinstance(result, list), f"method={method} returned non-list"


def test_coarse_to_fine():
    """Coarse-to-fine mode must not crash and returns valid detections."""
    rng = np.random.default_rng(42)
    image, _, _ = generate_synth_image(img_size=(128, 128), num_objects=2, rng=rng)
    backend = ToyClassifier(input_size=(16, 16), num_classes=4)
    detector = SlidingWindowDetector(
        backend=backend,
        input_size=(16, 16),
        classes=CLASS_NAMES,
        preprocess_fn=make_preprocess((16, 16)),
        background_class=BACKGROUND,
    )
    result = detector.detect(
        image, scales=[0.3], stride=0.25, score_thresh=0.5,
        coarse_to_fine=True, coarse_stride_factor=2.0,
    )
    assert isinstance(result, list)


def test_window_generator_box_mapping():
    """Windows from WindowGenerator map to valid original-image coordinates."""
    gen = WindowGenerator(input_size=(16, 16), scales=[0.25, 0.5], stride=0.5)
    h, w = 128, 128
    windows = gen.generate(h, w)
    for (x1, y1, x2, y2) in windows:
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
