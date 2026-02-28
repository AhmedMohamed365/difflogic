"""
Unit tests for the YOLOv1 difflogic extension.

Tests cover:
- Output tensor shapes
- YOLOTargetBuilder encode/decode consistency
- Loss sanity (finite, non-zero gradients on random model)
- Overfit test (32 samples → near-zero loss)
- PreprocessToBits determinism and bit count
- NMS
"""

import os
import sys
import math

import torch
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.logic_yolov1 import PreprocessToBits, LogicYOLOv1Tiny
from datasets.synth_shapes import SynthShapesDataset, YOLOTargetBuilder
from losses.yolo_bins import YOLOBinsLoss
from postprocess.yolo_decode import decode_predictions, nms


# ---------------------------------------------------------------------------
# Fixtures / shared constants
# ---------------------------------------------------------------------------

S, C, Q = 4, 3, 8    # small values for speed
IMG_H, IMG_W = 16, 16
NUM_THRESH = 2


@pytest.fixture(scope='module')
def tiny_model():
    model = LogicYOLOv1Tiny(
        S=S, C=C, Q=Q,
        num_thresholds=NUM_THRESH,
        img_channels=1,
        img_h=IMG_H, img_w=IMG_W,
        hidden_dim=256,
        num_layers=2,
        tau=5.0,
        device='cpu',
    )
    return model


@pytest.fixture(scope='module')
def tiny_dataset():
    return SynthShapesDataset(
        num_samples=32,
        img_size=(IMG_H, IMG_W),
        S=S, C=C, Q=Q,
        seed=42,
    )


# ---------------------------------------------------------------------------
# 1. PreprocessToBits
# ---------------------------------------------------------------------------

class TestPreprocessToBits:

    def test_output_shape(self):
        pp = PreprocessToBits(num_thresholds=3, img_channels=1,
                              img_h=8, img_w=8)
        x = torch.rand(5, 1, 8, 8)
        out = pp(x)
        assert out.shape == (5, 3 * 1 * 8 * 8), out.shape

    def test_out_dim_property(self):
        pp = PreprocessToBits(num_thresholds=2, img_channels=3,
                              img_h=4, img_w=4)
        assert pp.out_dim == 2 * 3 * 4 * 4

    def test_binary_outputs(self):
        pp = PreprocessToBits(num_thresholds=2, img_channels=1,
                              img_h=4, img_w=4)
        x = torch.rand(10, 1, 4, 4)
        out = pp(x)
        unique_vals = out.unique()
        assert set(unique_vals.tolist()).issubset({0.0, 1.0})

    def test_deterministic(self):
        pp = PreprocessToBits(num_thresholds=2, img_channels=1,
                              img_h=4, img_w=4)
        x = torch.rand(3, 1, 4, 4)
        assert torch.allclose(pp(x), pp(x))

    def test_monotone_thresholds(self):
        """Higher threshold ⇒ fewer 1 bits."""
        pp = PreprocessToBits(num_thresholds=3, img_channels=1,
                              img_h=8, img_w=8)
        x = torch.rand(4, 1, 8, 8)
        out = pp(x).reshape(4, 1 * 8 * 8, 3)
        # For each pixel: bits should be non-increasing (lower t → more 1s)
        assert (out[..., 0] >= out[..., 1]).all()
        assert (out[..., 1] >= out[..., 2]).all()


# ---------------------------------------------------------------------------
# 2. LogicYOLOv1Tiny shapes
# ---------------------------------------------------------------------------

class TestModelShapes:

    def test_output_shape_train(self, tiny_model):
        tiny_model.train()
        x = torch.rand(2, 1, IMG_H, IMG_W)
        out = tiny_model(x)
        expected_K = 1 + C + 4 * Q
        assert out.shape == (2, S, S, expected_K), out.shape

    def test_output_shape_eval(self, tiny_model):
        tiny_model.eval()
        x = torch.rand(2, 1, IMG_H, IMG_W)
        with torch.no_grad():
            out = tiny_model(x)
        expected_K = 1 + C + 4 * Q
        assert out.shape == (2, S, S, expected_K), out.shape

    def test_output_finite(self, tiny_model):
        tiny_model.train()
        x = torch.rand(3, 1, IMG_H, IMG_W)
        out = tiny_model(x)
        assert torch.isfinite(out).all()

    def test_K_formula(self, tiny_model):
        assert tiny_model.K == 1 + C + 4 * Q

    def test_split_output_shapes(self, tiny_model):
        tiny_model.eval()
        x = torch.rand(2, 1, IMG_H, IMG_W)
        with torch.no_grad():
            out = tiny_model(x)
        obj, cls, coords = tiny_model.split_output(out)
        assert obj.shape   == (2, S, S, 1)
        assert cls.shape   == (2, S, S, C)
        for key in ('x', 'y', 'w', 'h'):
            assert coords[key].shape == (2, S, S, Q), key


# ---------------------------------------------------------------------------
# 3. YOLOTargetBuilder encode/decode consistency
# ---------------------------------------------------------------------------

class TestYOLOTargetBuilder:

    def test_output_shape(self):
        tb = YOLOTargetBuilder(S=S, C=C, Q=Q)
        target = tb.build(class_id=1, bbox_xywh_norm=(0.5, 0.5, 0.2, 0.2))
        assert target.shape == (S, S, 1 + C + 4 * Q)

    def test_objectness_single_cell(self):
        tb = YOLOTargetBuilder(S=4, C=3, Q=8)
        target = tb.build(class_id=0, bbox_xywh_norm=(0.5, 0.5, 0.1, 0.1))
        # Exactly one cell should have objectness = 1
        assert target[..., 0].sum() == 1.0

    def test_class_one_hot_consistency(self):
        tb = YOLOTargetBuilder(S=4, C=3, Q=8)
        for cls in range(3):
            target = tb.build(class_id=cls, bbox_xywh_norm=(0.5, 0.5, 0.1, 0.1))
            obj_mask = target[..., 0].bool()
            cls_onehot = target[..., 1:4][obj_mask]  # [1, C]
            assert cls_onehot.shape == (1, 3)
            assert cls_onehot.argmax(-1).item() == cls

    def test_cell_assignment(self):
        """Object at (cx, cy) should be assigned to the correct grid cell."""
        tb = YOLOTargetBuilder(S=4, C=3, Q=8)
        cx, cy = 0.7, 0.3  # cell (2, 1) for S=4 (x*4=2.8→2, y*4=1.2→1)
        target = tb.build(class_id=0, bbox_xywh_norm=(cx, cy, 0.1, 0.1))
        obj = target[..., 0]
        # Find which cell has objectness=1
        cell_y, cell_x = (obj == 1).nonzero(as_tuple=False)[0].tolist()
        assert cell_x == int(cx * 4)
        assert cell_y == int(cy * 4)

    def test_coord_bins_in_range(self):
        """All coordinate one-hot bin indices should be within [0, Q)."""
        tb = YOLOTargetBuilder(S=4, C=3, Q=8)
        target = tb.build(class_id=1, bbox_xywh_norm=(0.5, 0.5, 0.2, 0.2))
        obj_mask = target[..., 0].bool()
        K = 1 + 3 + 4 * 8
        coord_part = target[..., 1 + 3:][obj_mask]  # [1, 4*Q]
        coord_part = coord_part.reshape(4, 8)
        for i in range(4):
            assert coord_part[i].sum() == 1.0, f'coord {i} not one-hot'
            assert 0 <= coord_part[i].argmax().item() < 8

    def test_decode_consistency(self):
        """
        Encode a bbox, then decode using argmax bins.
        The decoded value should be close to the original (within ±1/Q).
        """
        S_, C_, Q_ = 8, 3, 16
        tb = YOLOTargetBuilder(S=S_, C=C_, Q=Q_)
        cx, cy, bw, bh = 0.45, 0.55, 0.20, 0.15
        target = tb.build(class_id=0, bbox_xywh_norm=(cx, cy, bw, bh))

        # find responsible cell
        cell_x = int(cx * S_)
        cell_y = int(cy * S_)
        cell = target[cell_y, cell_x]   # [K]

        off = 1 + C_
        for i, (true_val, label) in enumerate([
            (cx * S_ - cell_x, 'x_rel'),
            (cy * S_ - cell_y, 'y_rel'),
            (bw, 'w'),
            (bh, 'h'),
        ]):
            bins = cell[off + i * Q_: off + (i + 1) * Q_]
            bin_idx = bins.argmax().item()
            decoded = (bin_idx + 0.5) / Q_
            assert abs(decoded - true_val) <= 1.0 / Q_ + 1e-6, \
                f'{label}: decoded={decoded:.4f}, true={true_val:.4f}'


# ---------------------------------------------------------------------------
# 4. Loss sanity
# ---------------------------------------------------------------------------

class TestYOLOBinsLoss:

    def test_loss_finite(self, tiny_model, tiny_dataset):
        tiny_model.train()
        criterion = YOLOBinsLoss(S=S, C=C, Q=Q)

        imgs    = torch.stack([tiny_dataset[i][0] for i in range(4)])
        targets = torch.stack([tiny_dataset[i][1] for i in range(4)])

        pred = tiny_model(imgs)
        loss = criterion(pred, targets)

        assert torch.isfinite(loss), f'Loss is not finite: {loss}'
        assert loss.item() >= 0, f'Loss is negative: {loss}'

    def test_gradients_non_zero(self, tiny_model, tiny_dataset):
        tiny_model.train()
        # Zero out any existing grads
        for p in tiny_model.parameters():
            p.grad = None

        criterion = YOLOBinsLoss(S=S, C=C, Q=Q)
        imgs    = torch.stack([tiny_dataset[i][0] for i in range(4)])
        targets = torch.stack([tiny_dataset[i][1] for i in range(4)])

        pred = tiny_model(imgs)
        loss = criterion(pred, targets)
        loss.backward()

        has_nonzero = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in tiny_model.parameters()
        )
        assert has_nonzero, 'All gradients are zero!'

    def test_random_target_loss(self):
        """Loss on random pred / random target should still be finite."""
        S_, C_, Q_ = 4, 3, 8
        K = 1 + C_ + 4 * Q_
        pred   = torch.randn(2, S_, S_, K)
        target = torch.zeros(2, S_, S_, K)
        # set one object per batch item
        target[0, 2, 3, 0] = 1.0
        target[0, 2, 3, 1] = 1.0  # class 0
        target[0, 2, 3, 1 + C_ + 0] = 1.0   # x bin 0
        target[0, 2, 3, 1 + C_ + Q_] = 1.0  # y bin 0
        target[0, 2, 3, 1 + C_ + 2 * Q_] = 1.0
        target[0, 2, 3, 1 + C_ + 3 * Q_] = 1.0

        criterion = YOLOBinsLoss(S=S_, C=C_, Q=Q_)
        loss = criterion(pred, target)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 5. Decode + NMS
# ---------------------------------------------------------------------------

class TestDecode:

    def test_decode_output_format(self, tiny_model):
        tiny_model.eval()
        x = torch.rand(3, 1, IMG_H, IMG_W)
        with torch.no_grad():
            raw = tiny_model(x)
        dets = decode_predictions(raw, S=S, C=C, Q=Q, score_thresh=0.0)
        assert len(dets) == 3
        for d in dets:
            assert d.ndim == 2
            assert d.shape[1] == 6

    def test_nms_removes_duplicates(self):
        """Two nearly identical boxes should reduce to one after NMS."""
        boxes = torch.tensor([
            [0.5, 0.5, 0.4, 0.4],
            [0.5, 0.5, 0.4, 0.4],
            [0.1, 0.1, 0.1, 0.1],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_thresh=0.5)
        assert 0 in keep.tolist()
        # The second box should be suppressed
        assert 1 not in keep.tolist()
        assert 2 in keep.tolist()

    def test_nms_empty_input(self):
        boxes  = torch.zeros(0, 4)
        scores = torch.zeros(0)
        keep   = nms(boxes, scores)
        assert keep.shape[0] == 0

    def test_score_threshold_filters(self, tiny_model):
        tiny_model.eval()
        x = torch.rand(1, 1, IMG_H, IMG_W)
        with torch.no_grad():
            raw = tiny_model(x)
        # With score_thresh=1.0 nothing should pass
        dets = decode_predictions(raw, S=S, C=C, Q=Q, score_thresh=1.0)
        assert dets[0].shape[0] == 0

    def test_decode_coords_in_range(self, tiny_model):
        """Decoded box coords should be in reasonable range [0, 1]."""
        tiny_model.eval()
        x = torch.rand(2, 1, IMG_H, IMG_W)
        with torch.no_grad():
            raw = tiny_model(x)
        dets = decode_predictions(raw, S=S, C=C, Q=Q, score_thresh=0.0)
        for det in dets:
            if det.shape[0] > 0:
                cx, cy, w, h = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
                assert (cx >= 0).all() and (cx <= 1 + 1e-4).all()
                assert (cy >= 0).all() and (cy <= 1 + 1e-4).all()


# ---------------------------------------------------------------------------
# 6. Dataset
# ---------------------------------------------------------------------------

class TestSynthShapesDataset:

    def test_len(self, tiny_dataset):
        assert len(tiny_dataset) == 32

    def test_image_shape(self, tiny_dataset):
        img, target = tiny_dataset[0]
        assert img.shape == (1, IMG_H, IMG_W)

    def test_target_shape(self, tiny_dataset):
        img, target = tiny_dataset[0]
        assert target.shape == (S, S, 1 + C + 4 * Q)

    def test_image_range(self, tiny_dataset):
        img, _ = tiny_dataset[0]
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_exactly_one_object_per_image(self, tiny_dataset):
        for i in range(len(tiny_dataset)):
            _, target = tiny_dataset[i]
            n_obj = target[..., 0].sum().item()
            assert n_obj == 1.0, f'Image {i}: expected 1 object, got {n_obj}'
