"""
Sliding-window detection framework for difflogic classification models.

This package converts a trained difflogic classifier into an object detector
by scanning images with many windows/tiles, classifying each tile, then
merging hits with NMS or other post-processing methods.
"""

from .backends import TorchBackend, CompiledBackend
from .windows import WindowGenerator
from .postprocess import nms, soft_nms, weighted_box_fusion
from .sliding_window import SlidingWindowDetector

__all__ = [
    "TorchBackend",
    "CompiledBackend",
    "WindowGenerator",
    "nms",
    "soft_nms",
    "weighted_box_fusion",
    "SlidingWindowDetector",
]
