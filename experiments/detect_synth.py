#!/usr/bin/env python3
"""
Run sliding-window detection on the synthetic shapes dataset and print / save
results.

Usage::

    python detect_synth.py \\
        --classifier classifier_synth.pt \\
        --num-images 20 \\
        --scales 0.2 0.3 0.4 \\
        --stride 0.25 \\
        --thresh 0.6

Pass ``--no-classifier`` to use the built-in variance-based toy backend
(for quick smoke-testing without a trained model).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Detect shapes in synthetic images.")
    parser.add_argument("--classifier", type=str, default=None,
                        help="Path to saved classifier .pt file.")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Use a trivial variance-based mock classifier.")
    parser.add_argument("--num-images", type=int, default=10,
                        help="Number of test images.")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.2, 0.3, 0.4],
                        help="Detection scales (relative to short side).")
    parser.add_argument("--stride", type=float, default=0.25,
                        help="Stride as fraction of window size.")
    parser.add_argument("--thresh", type=float, default=0.6,
                        help="Score threshold.")
    parser.add_argument("--merge", type=str, default="nms",
                        choices=["nms", "soft_nms", "wbf"],
                        help="Post-processing method.")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for NMS / WBF.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Dataset seed.")
    parser.add_argument("--save-images", action="store_true",
                        help="Save annotated images to ./results_synth/.")
    return parser.parse_args()


def _variance_backend(batch_inputs, num_classes=4):
    """Trivial mock backend that fires on high-variance patches."""
    from detector.postprocess import BACKGROUND  # noqa – just use constant
    BACKGROUND_IDX = 0
    CIRCLE_IDX = 1
    n = batch_inputs.shape[0]
    scores = np.zeros((n, num_classes), dtype=np.float32)
    scores[:, BACKGROUND_IDX] = 0.6
    for i in range(n):
        v = float(batch_inputs[i].var())
        if v > 0.02:
            scores[i, CIRCLE_IDX] = 0.8
            scores[i, BACKGROUND_IDX] = 0.2
    return scores


class _MockBackend:
    def predict(self, x):
        return _variance_backend(x)


def load_backend(args):
    if args.no_classifier or args.classifier is None:
        print("Using variance-based mock backend.")
        return _MockBackend(), None, 4

    import torch
    from detector import TorchBackend

    ckpt = torch.load(args.classifier, map_location="cpu")
    in_dim = ckpt["in_dim"]
    num_classes = ckpt["num_classes"]
    patch_size = ckpt["patch_size"]
    num_thresholds = ckpt["num_thresholds"]

    # Reconstruct model
    hidden = 128
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    backend = TorchBackend(model, device="cpu")
    return backend, (patch_size, num_thresholds), num_classes


def build_preprocess(patch_size, num_thresholds):
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
        bits = np.concatenate([
            (f > (i + 1) / (num_thresholds + 1)).astype(np.float32).ravel()
            for i in range(num_thresholds)
        ])
        return bits

    return preprocess


def compute_iou(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    ab = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def evaluate(all_dets, all_gt_boxes, all_gt_labels, iou_thresh=0.5):
    tp = fp = fn = 0
    for dets, gt_boxes, gt_labels in zip(all_dets, all_gt_boxes, all_gt_labels):
        matched = set()
        for det in dets:
            best_iou = 0.0
            best_j = -1
            for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if j in matched:
                    continue
                iou = compute_iou(
                    (det["x1"], det["y1"], det["x2"], det["y2"]), gb
                )
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, fp, fn


def main():
    args = parse_args()

    backend, classifier_cfg, num_classes = load_backend(args)

    if classifier_cfg is not None:
        patch_size, num_thresholds = classifier_cfg
        preprocess = build_preprocess(patch_size, num_thresholds)
        input_size = (patch_size, patch_size)
    else:
        input_size = (16, 16)
        patch_size, num_thresholds = 16, 3
        preprocess = build_preprocess(patch_size, num_thresholds)

    from datasets.synth_shapes_detect import SynthShapesDataset, CLASS_NAMES
    from detector import SlidingWindowDetector

    dataset = SynthShapesDataset(
        num_images=args.num_images,
        img_size=(128, 128),
        num_objects_range=(1, 3),
        seed=args.seed,
    )

    detector = SlidingWindowDetector(
        backend=backend,
        input_size=input_size,
        classes=CLASS_NAMES,
        preprocess_fn=preprocess,
        background_class=0,
        merge_method=args.merge,
        nms_iou_thresh=args.iou_thresh,
    )

    all_dets = []
    all_gt_boxes = []
    all_gt_labels = []

    print(f"\nRunning detection on {args.num_images} images…")
    for i in range(args.num_images):
        image, boxes, labels = dataset[i]
        dets = detector.detect(
            image,
            scales=args.scales,
            stride=args.stride,
            score_thresh=args.thresh,
            batch_size=args.batch_size,
        )
        all_dets.append(dets)
        all_gt_boxes.append(boxes)
        all_gt_labels.append(labels)

        if i < 3:
            print(f"  Image {i}: GT={len(boxes)} objects, "
                  f"detected={len(dets)} boxes")

    precision, recall, tp, fp, fn = evaluate(all_dets, all_gt_boxes, all_gt_labels)
    print(f"\nResults (IoU@0.5):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")

    if args.save_images:
        _save_images(dataset, all_dets, args)


def _save_images(dataset, all_dets, args):
    os.makedirs("results_synth", exist_ok=True)
    try:
        from PIL import Image as PILImage, ImageDraw
    except ImportError:
        print("Pillow not available – skipping image save.")
        return

    for i in range(min(len(all_dets), args.num_images)):
        image, boxes, labels = dataset[i]
        pil = PILImage.fromarray(image)
        draw = ImageDraw.Draw(pil)
        # GT in green
        for box in boxes:
            draw.rectangle(list(box), outline=(0, 200, 0), width=2)
        # Detections in red
        for det in all_dets[i]:
            draw.rectangle(
                [det["x1"], det["y1"], det["x2"], det["y2"]],
                outline=(200, 0, 0), width=2,
            )
            draw.text((det["x1"], det["y1"] - 10),
                      f"{det['score']:.2f}", fill=(200, 0, 0))
        path = f"results_synth/img_{i:04d}.png"
        pil.save(path)
    print(f"Images saved to results_synth/")


if __name__ == "__main__":
    main()
