#!/usr/bin/env python3
"""
Train a patch classifier on the synthetic shapes dataset for use with
SlidingWindowDetector.

The model is a simple PyTorch classifier (linear layers) that takes a
flattened, binarised patch and predicts one of:
  0 = background, 1 = circle, 2 = rectangle, 3 = triangle

Usage::

    python train_patch_classifier_synth.py \\
        --patch-size 16 \\
        --num-images 500 \\
        --epochs 20 \\
        --output classifier.pt

After training, the saved model can be loaded by detect_synth.py.
"""

import argparse
import os
import sys

# Allow importing from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Train patch classifier on synthetic shapes.")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Side length of square patches (default: 16)")
    parser.add_argument("--num-images", type=int, default=300,
                        help="Number of training images (default: 300)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output", type=str, default="classifier_synth.pt",
                        help="Output path for saved model (default: classifier_synth.pt)")
    parser.add_argument("--num-thresholds", type=int, default=3,
                        help="Number of binarisation thresholds per pixel (default: 3)")
    return parser.parse_args()


def build_preprocess(patch_size, num_thresholds):
    """Return a preprocess function and the input dimensionality."""
    h = w = patch_size

    def preprocess(patch: np.ndarray) -> np.ndarray:
        # Resize to patch_size × patch_size
        y_idx = np.linspace(0, patch.shape[0] - 1, h).astype(int)
        x_idx = np.linspace(0, patch.shape[1] - 1, w).astype(int)
        resized = patch[np.ix_(y_idx, x_idx)]
        f = resized.astype(np.float32)
        if f.max() > 1.0:
            f = f / 255.0
        # Convert to grey if needed
        if f.ndim == 3:
            f = f.mean(axis=2)
        # Multi-threshold binarisation (matches difflogic MNIST encoding)
        bits = np.concatenate([
            (f > (i + 1) / (num_thresholds + 1)).astype(np.float32).ravel()
            for i in range(num_thresholds)
        ])
        return bits

    in_dim = h * w * num_thresholds
    return preprocess, in_dim


def build_model(in_dim: int, num_classes: int = 4):
    """A simple two-layer MLP classifier."""
    hidden = 128
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from datasets.synth_shapes_detect import SynthShapesDataset
    from datasets.patch_sampler import PatchSampler

    print(f"Building synthetic dataset ({args.num_images} images)…")
    dataset = SynthShapesDataset(
        num_images=args.num_images,
        img_size=(128, 128),
        num_objects_range=(1, 4),
        seed=args.seed,
    )

    preprocess, in_dim = build_preprocess(args.patch_size, args.num_thresholds)

    print("Sampling patches…")
    sampler = PatchSampler(
        detection_dataset=dataset,
        patch_size=(args.patch_size, args.patch_size),
        positives_per_image=4,
        negatives_per_image=6,
        preprocess_fn=preprocess,
        seed=args.seed,
    )
    patches, labels = sampler.build()

    print(f"  {patches.shape[0]} patches, input_dim={in_dim}")
    print(f"  class distribution: {np.bincount(labels)}")

    # Shuffle
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(patches))
    patches = patches[idx]
    labels = labels[idx]

    X = torch.from_numpy(patches).float()
    Y = torch.from_numpy(labels).long()

    model = build_model(in_dim, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    n = len(X)
    bs = args.batch_size

    print(f"\nTraining for {args.epochs} epochs…")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        correct = 0
        for start in range(0, n, bs):
            batch_idx = perm[start: start + bs]
            xb = X[batch_idx]
            yb = Y[batch_idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        acc = correct / n
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss/n:.4f}  acc={acc:.3f}")

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "in_dim": in_dim,
        "num_classes": 4,
        "patch_size": args.patch_size,
        "num_thresholds": args.num_thresholds,
    }, args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
