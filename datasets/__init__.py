"""
Dataset utilities for the sliding-window detection framework.
"""

from .synth_shapes_detect import SynthShapesDataset, generate_synth_image
from .patch_sampler import PatchSampler

__all__ = [
    "SynthShapesDataset",
    "generate_synth_image",
    "PatchSampler",
]
