#!/usr/bin/env python3
"""
Benchmark the SlidingWindowDetector on synthetic images.

Reports:
- Number of windows/sec
- ms/image at each scale
- Comparison across merge methods

Usage::

    python bench_detector.py \\
        --num-images 20 \\
        --scales 0.2 0.3 0.4 \\
        --stride 0.25 \\
        --batch-size 64
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SlidingWindowDetector.")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument("--stride", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=128,
                        help="Image side length (default: 128)")
    return parser.parse_args()


class _MockBackend:
    """Trivial backend that returns random scores (for speed benchmarking only)."""
    def __init__(self, num_classes=4, seed=0):
        self.num_classes = num_classes
        self._rng = np.random.default_rng(seed)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._rng.random((x.shape[0], self.num_classes)).astype(np.float32)


def build_preprocess(patch_size=16):
    h = w = patch_size

    def preprocess(patch):
        y_idx = np.linspace(0, patch.shape[0] - 1, h).astype(int)
        x_idx = np.linspace(0, patch.shape[1] - 1, w).astype(int)
        resized = patch[np.ix_(y_idx, x_idx)]
        f = resized.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        if f.ndim == 3:
            f = f.mean(axis=2)
        return f.ravel()

    return preprocess


def main():
    args = parse_args()

    from datasets.synth_shapes_detect import SynthShapesDataset, CLASS_NAMES
    from detector import SlidingWindowDetector
    from detector.windows import WindowGenerator

    dataset = SynthShapesDataset(
        num_images=args.num_images,
        img_size=(args.img_size, args.img_size),
        num_objects_range=(1, 4),
        seed=args.seed,
    )

    backend = _MockBackend(num_classes=4, seed=args.seed)
    preprocess = build_preprocess(patch_size=16)
    input_size = (16, 16)

    # --- Count windows ---
    gen = WindowGenerator(
        input_size=input_size,
        scales=args.scales,
        stride=args.stride,
    )
    sample_img, _, _ = dataset[0]
    h, w = sample_img.shape[:2]
    windows = gen.generate(h, w)
    print(f"\nImage size: {h}×{w}")
    print(f"Scales: {args.scales}, stride: {args.stride}")
    print(f"Windows per image: {len(windows)}")

    # --- Benchmark per merge method ---
    merge_methods = ["nms", "soft_nms", "wbf"]
    print(f"\n{'Method':<12}  {'ms/image':>10}  {'windows/sec':>14}  {'dets/image':>12}")
    print("-" * 54)

    for method in merge_methods:
        detector = SlidingWindowDetector(
            backend=backend,
            input_size=input_size,
            classes=CLASS_NAMES,
            preprocess_fn=preprocess,
            background_class=0,
            merge_method=method,
            nms_iou_thresh=0.5,
        )

        t0 = time.perf_counter()
        total_dets = 0
        for i in range(args.num_images):
            image, _, _ = dataset[i]
            dets = detector.detect(
                image,
                scales=args.scales,
                stride=args.stride,
                score_thresh=args.thresh,
                batch_size=args.batch_size,
            )
            total_dets += len(dets)

        elapsed = time.perf_counter() - t0
        ms_per_image = elapsed / args.num_images * 1000
        wins_per_sec = (len(windows) * args.num_images) / elapsed
        avg_dets = total_dets / args.num_images

        print(
            f"{method:<12}  {ms_per_image:>10.1f}  {wins_per_sec:>14.0f}  {avg_dets:>12.1f}"
        )

    # --- Per-scale timing ---
    print(f"\nPer-scale timing (nms, stride={args.stride}):")
    print(f"{'Scale':<8}  {'windows':>8}  {'ms/image':>10}")
    print("-" * 32)

    for scale in args.scales:
        det = SlidingWindowDetector(
            backend=backend,
            input_size=input_size,
            classes=CLASS_NAMES,
            preprocess_fn=preprocess,
            background_class=0,
            merge_method="nms",
        )
        wins = WindowGenerator(
            input_size=input_size, scales=[scale], stride=args.stride
        ).generate(h, w)

        t0 = time.perf_counter()
        for i in range(args.num_images):
            image, _, _ = dataset[i]
            det.detect(
                image, scales=[scale], stride=args.stride,
                score_thresh=args.thresh, batch_size=args.batch_size,
            )
        elapsed = time.perf_counter() - t0
        ms_per_image = elapsed / args.num_images * 1000
        print(f"{scale:<8.2f}  {len(wins):>8}  {ms_per_image:>10.1f}")

    print()


if __name__ == "__main__":
    main()
