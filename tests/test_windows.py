"""
Tests for detector.windows.WindowGenerator.

Validates:
- Correct window generation for a known image size and scale.
- Correct mapping of windows to original image coordinates.
- Deterministic output for identical parameters.
- Multi-scale output contains windows at every requested scale.
- Stride control works correctly.
"""

import sys
import os

# Make the detector package importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from detector.windows import WindowGenerator


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------

def test_windows_are_tuples_of_four():
    gen = WindowGenerator(input_size=(28, 28), scales=[0.5], stride=0.5)
    windows = gen.generate(64, 64)
    assert len(windows) > 0
    for w in windows:
        assert len(w) == 4, "Each window must be a 4-tuple (x1, y1, x2, y2)"


def test_windows_within_image_bounds():
    h, w = 100, 120
    gen = WindowGenerator(input_size=(28, 28), scales=[0.3, 0.5], stride=0.25)
    windows = gen.generate(h, w)
    for (x1, y1, x2, y2) in windows:
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= w
        assert y2 <= h
        assert x2 > x1
        assert y2 > y1


def test_window_size_matches_scale_absolute():
    """When scale > 1 it is interpreted as absolute pixel size."""
    gen = WindowGenerator(input_size=(28, 28), scales=[32.0], stride=0.5)
    windows = gen.generate(64, 64)
    for (x1, y1, x2, y2) in windows:
        assert (x2 - x1) == 32
        assert (y2 - y1) == 32


def test_window_size_matches_scale_relative():
    """When scale <= 1 it is relative to the shorter image side."""
    gen = WindowGenerator(input_size=(28, 28), scales=[0.5], stride=0.5)
    windows = gen.generate(100, 100)
    for (x1, y1, x2, y2) in windows:
        assert (x2 - x1) == 50
        assert (y2 - y1) == 50


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_output():
    gen1 = WindowGenerator(input_size=(28, 28), scales=[0.3, 0.5], stride=0.25)
    gen2 = WindowGenerator(input_size=(28, 28), scales=[0.3, 0.5], stride=0.25)
    wins1 = gen1.generate(100, 100)
    wins2 = gen2.generate(100, 100)
    assert wins1 == wins2


# ---------------------------------------------------------------------------
# Multi-scale
# ---------------------------------------------------------------------------

def test_multiscale_has_multiple_sizes():
    gen = WindowGenerator(input_size=(28, 28), scales=[0.2, 0.4, 0.6], stride=0.5)
    windows = gen.generate(200, 200)
    sizes = set((x2 - x1) for x1, y1, x2, y2 in windows)
    # Expect windows of at least 2 different sizes
    assert len(sizes) >= 2


# ---------------------------------------------------------------------------
# Stride
# ---------------------------------------------------------------------------

def test_stride_controls_density():
    """Smaller stride → more windows."""
    gen_coarse = WindowGenerator(input_size=(28, 28), scales=[0.5], stride=0.5)
    gen_fine = WindowGenerator(input_size=(28, 28), scales=[0.5], stride=0.1)
    wins_coarse = gen_coarse.generate(200, 200)
    wins_fine = gen_fine.generate(200, 200)
    assert len(wins_fine) > len(wins_coarse)


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def test_box_coordinates_are_integers():
    gen = WindowGenerator(input_size=(28, 28), scales=[0.3], stride=0.5)
    windows = gen.generate(128, 128)
    for (x1, y1, x2, y2) in windows:
        assert isinstance(x1, int)
        assert isinstance(y1, int)
        assert isinstance(x2, int)
        assert isinstance(y2, int)


# ---------------------------------------------------------------------------
# Aspect ratios
# ---------------------------------------------------------------------------

def test_aspect_ratios():
    gen = WindowGenerator(
        input_size=(28, 28), scales=[0.4], stride=0.5, aspect_ratios=[1.0, 2.0]
    )
    windows = gen.generate(200, 200)
    widths = set((x2 - x1) for x1, y1, x2, y2 in windows)
    # Expect both 1:1 and 2:1 windows
    assert len(widths) >= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_scale_larger_than_image_produces_no_windows():
    """A scale larger than the image should produce no windows."""
    gen = WindowGenerator(input_size=(28, 28), scales=[2.0], stride=0.5)
    windows = gen.generate(50, 50)
    # scale 2.0 means 100px window on 50px image → should be skipped
    assert len(windows) == 0


def test_no_duplicate_windows():
    gen = WindowGenerator(input_size=(28, 28), scales=[0.5], stride=0.25)
    windows = gen.generate(100, 100)
    assert len(windows) == len(set(windows)), "Windows must be unique"
