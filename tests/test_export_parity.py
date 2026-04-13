"""
Export parity tests: verify that CompiledLogicNet matches PyTorch-eval
for LogicYOLOv1Tiny.

These tests skip automatically when gcc/clang is not available or when the
model is too large to compile quickly in CI.  They exercise the full
export → inference → argmax-comparison pipeline.
"""

import os
import sys
import shutil

import numpy as np
import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.logic_yolov1 import LogicYOLOv1Tiny
from export.export_compiled_yolo import export_compiled, binarize_inputs

# Skip entire module if gcc is not available
pytestmark = pytest.mark.skipif(
    shutil.which('gcc') is None and shutil.which('clang') is None,
    reason='Neither gcc nor clang found on PATH; skipping compiled-net tests',
)


# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def compiled_model_pair():
    """Returns (pytorch_model_eval, compiled_net, random_imgs)."""
    # Use very small dims for fast compilation in CI
    model = LogicYOLOv1Tiny(
        S=2, C=2, Q=4,
        num_thresholds=1,
        img_channels=1,
        img_h=8, img_w=8,
        hidden_dim=64,
        num_layers=2,
        tau=4.0,
        device='cpu',
    )
    model.eval()

    compiler = 'gcc' if shutil.which('gcc') else 'clang'
    compiled = export_compiled(model, num_bits=64, cpu_compiler=compiler,
                               opt_level=0, verbose=False)

    torch.manual_seed(7)
    imgs = torch.rand(8, 1, 8, 8)
    return model, compiled, imgs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportParity:

    def test_compiled_output_shape(self, compiled_model_pair):
        model, compiled, imgs = compiled_model_pair
        bits_np = binarize_inputs(imgs, model)
        out = compiled(bits_np)
        assert out.shape == (imgs.shape[0], model.total_out), out.shape

    def test_pytorch_output_shape(self, compiled_model_pair):
        model, compiled, imgs = compiled_model_pair
        with torch.no_grad():
            bits = model.preprocess(imgs)
            flat = model.backbone(bits)
        assert flat.shape == (imgs.shape[0], model.total_out)

    def test_class_argmax_agreement(self, compiled_model_pair):
        """
        Class-argmax of compiled output should match PyTorch-eval output for
        ≥95% of cells (allow small differences due to tie-breaking).
        """
        model, compiled, imgs = compiled_model_pair

        bits_np = binarize_inputs(imgs, model)

        with torch.no_grad():
            bits_t = model.preprocess(imgs)
            pt_flat = model.backbone(bits_t).numpy()

        comp_flat = compiled(bits_np).float().numpy()

        S, C, Q, K = model.S, model.C, model.Q, model.K
        N = imgs.shape[0]

        pt   = pt_flat.reshape(N, S * S, K)
        comp = comp_flat.reshape(N, S * S, K)

        # class argmax  [N, S*S]
        pt_cls   = pt[...,   1:1 + C].argmax(-1)
        comp_cls = comp[..., 1:1 + C].argmax(-1)
        match_rate = (pt_cls == comp_cls).mean()
        assert match_rate >= 0.95, \
            f'Class argmax agreement {match_rate * 100:.1f}% < 95%'

    def test_coord_argmax_agreement(self, compiled_model_pair):
        """
        Each coordinate bin-argmax should agree for ≥95% of object cells.
        """
        model, compiled, imgs = compiled_model_pair
        bits_np = binarize_inputs(imgs, model)

        with torch.no_grad():
            bits_t = model.preprocess(imgs)
            pt_flat = model.backbone(bits_t).numpy()
        comp_flat = compiled(bits_np).float().numpy()

        S, C, Q, K = model.S, model.C, model.Q, model.K
        N = imgs.shape[0]
        pt   = pt_flat.reshape(N, S * S, K)
        comp = comp_flat.reshape(N, S * S, K)

        for ci, name in enumerate('xywh'):
            off = 1 + C + ci * Q
            pt_arg   = pt[...,   off:off + Q].argmax(-1)
            comp_arg = comp[..., off:off + Q].argmax(-1)
            rate = (pt_arg == comp_arg).mean()
            assert rate >= 0.95, \
                f'{name} argmax agreement {rate * 100:.1f}% < 95%'

    def test_objectness_sign_agreement(self, compiled_model_pair):
        """
        The sign of the objectness score (positive / zero / negative) should
        be consistent between PyTorch-eval and compiled.
        """
        model, compiled, imgs = compiled_model_pair
        bits_np = binarize_inputs(imgs, model)

        with torch.no_grad():
            bits_t = model.preprocess(imgs)
            pt_flat = model.backbone(bits_t).numpy()
        comp_flat = compiled(bits_np).float().numpy()

        S, K = model.S, model.K
        N = imgs.shape[0]
        pt   = pt_flat.reshape(N, S * S, K)
        comp = comp_flat.reshape(N, S * S, K)

        pt_obj_pos   = pt[...,   0] > 0
        comp_obj_pos = comp[..., 0] > 0
        rate = (pt_obj_pos == comp_obj_pos).mean()
        assert rate >= 0.90, \
            f'Objectness sign agreement {rate * 100:.1f}% < 90%'
