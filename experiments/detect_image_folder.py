#!/usr/bin/env python3
"""
Run sliding-window detection on a folder of images.

Usage::

    python detect_image_folder.py \\
        --classifier classifier_synth.pt \\
        --image-dir /path/to/images \\
        --scales 0.2 0.3 0.4 \\
        --stride 0.25 \\
        --thresh 0.6 \\
        --output detections.json

Each image in ``--image-dir`` (jpg/png) is processed and detections are
written to a JSON file.  If ``--save-images`` is passed, annotated images are
saved alongside the JSON.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SlidingWindowDetector on a folder of images."
    )
    parser.add_argument("--classifier", type=str, required=True,
                        help="Path to saved classifier .pt file.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing input images.")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.2, 0.3, 0.4],
                        help="Detection scales (relative to short side).")
    parser.add_argument("--stride", type=float, default=0.25,
                        help="Stride as fraction of window size.")
    parser.add_argument("--thresh", type=float, default=0.6,
                        help="Score threshold.")
    parser.add_argument("--merge", type=str, default="nms",
                        choices=["nms", "soft_nms", "wbf"])
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="detections.json",
                        help="Output JSON path.")
    parser.add_argument("--save-images", action="store_true",
                        help="Save annotated images alongside JSON.")
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    try:
        from PIL import Image as PILImage  # type: ignore
        img = PILImage.open(path).convert("RGB")
        return np.array(img)
    except ImportError:
        raise ImportError("Pillow is required to load images. Install with: pip install pillow")


def load_backend_and_preprocess(classifier_path: str):
    import torch
    import torch.nn as nn
    from detector import TorchBackend

    ckpt = torch.load(classifier_path, map_location="cpu")
    in_dim = ckpt["in_dim"]
    num_classes = ckpt["num_classes"]
    patch_size = ckpt["patch_size"]
    num_thresholds = ckpt["num_thresholds"]

    hidden = 128
    model = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    backend = TorchBackend(model, device="cpu")

    h = w = patch_size

    def preprocess(patch: np.ndarray) -> np.ndarray:
        y_idx = np.linspace(0, patch.shape[0] - 1, h).astype(int)
        x_idx = np.linspace(0, patch.shape[1] - 1, w).astype(int)
        resized = patch[np.ix_(y_idx, x_idx)]
        f = resized.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        if f.ndim == 3:
            f = f.mean(axis=2)
        bits = np.concatenate([
            (f > (i + 1) / (num_thresholds + 1)).astype(np.float32).ravel()
            for i in range(num_thresholds)
        ])
        return bits

    return backend, preprocess, (patch_size, patch_size), num_classes


def main():
    args = parse_args()

    backend, preprocess, input_size, num_classes = load_backend_and_preprocess(
        args.classifier
    )

    from detector import SlidingWindowDetector

    CLASS_NAMES = [f"class_{i}" for i in range(num_classes)]
    CLASS_NAMES[0] = "background"

    detector = SlidingWindowDetector(
        backend=backend,
        input_size=input_size,
        classes=CLASS_NAMES,
        preprocess_fn=preprocess,
        background_class=0,
        merge_method=args.merge,
        nms_iou_thresh=args.iou_thresh,
    )

    image_dir = args.image_dir
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in exts
    )
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    results = {}
    save_dir = os.path.join(os.path.dirname(args.output), "annotated_images")
    if args.save_images:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {len(image_files)} images…")
    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        try:
            image = load_image(fpath)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
            continue

        dets = detector.detect(
            image,
            scales=args.scales,
            stride=args.stride,
            score_thresh=args.thresh,
            batch_size=args.batch_size,
        )

        results[fname] = dets
        print(f"  {fname}: {len(dets)} detections")

        if args.save_images:
            _annotate_and_save(image, dets, os.path.join(save_dir, fname))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detections written to {args.output}")


def _annotate_and_save(image, dets, out_path):
    try:
        from PIL import Image as PILImage, ImageDraw
    except ImportError:
        return
    pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for det in dets:
        draw.rectangle(
            [det["x1"], det["y1"], det["x2"], det["y2"]],
            outline=(255, 0, 0), width=2,
        )
        draw.text(
            (det["x1"], det["y1"] - 12),
            f"c{det['class_id']}:{det['score']:.2f}",
            fill=(255, 0, 0),
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pil.save(out_path)


if __name__ == "__main__":
    main()
