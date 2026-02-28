"""
Train LogicYOLOv1Tiny on the synthetic shapes dataset.

Quick start
-----------
From the repository root::

    python experiments/train_yolo_synth.py

All hyperparameters have sensible defaults.  The script prints loss values
every epoch and saves a checkpoint to ``checkpoints/yolo_synth_best.pt``.
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, random_split

# Ensure repo root is on the path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.synth_shapes import SynthShapesDataset
from losses.yolo_bins import YOLOBinsLoss
from models.logic_yolov1 import LogicYOLOv1Tiny


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train LogicYOLOv1Tiny on SynthShapes')
    # Dataset
    p.add_argument('--num-samples',  type=int, default=2000)
    p.add_argument('--img-size',     type=int, default=32,
                   help='Square image size (img_h = img_w = img_size)')
    p.add_argument('--seed',         type=int, default=42)
    # YOLO spec
    p.add_argument('--S',  type=int, default=8,  help='Grid size')
    p.add_argument('--C',  type=int, default=3,  help='Number of classes')
    p.add_argument('--Q',  type=int, default=16, help='Bins per coord')
    # Model
    p.add_argument('--num-thresholds', type=int,   default=2)
    p.add_argument('--hidden-dim',     type=int,   default=1024)
    p.add_argument('--num-layers',     type=int,   default=4)
    p.add_argument('--tau',            type=float, default=10.0)
    p.add_argument('--grad-factor',    type=float, default=1.0)
    # Training
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--batch-size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=0.01)
    p.add_argument('--val-split',  type=float, default=0.1)
    # Loss
    p.add_argument('--lambda-coord', type=float, default=5.0)
    p.add_argument('--lambda-noobj', type=float, default=0.5)
    # Misc
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--device',         default='cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = SynthShapesDataset(
        num_samples=args.num_samples,
        img_size=(args.img_size, args.img_size),
        S=args.S, C=args.C, Q=args.Q,
        seed=args.seed,
    )

    val_len   = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = dict(
        S=args.S, C=args.C, Q=args.Q,
        num_thresholds=args.num_thresholds,
        img_channels=1,
        img_h=args.img_size, img_w=args.img_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        tau=args.tau,
        device=args.device,
        grad_factor=args.grad_factor,
    )
    model = LogicYOLOv1Tiny(**model_cfg).to(args.device)

    # ------------------------------------------------------------------
    # Optimizer & loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = YOLOBinsLoss(
        S=args.S, C=args.C, Q=args.Q,
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
    )

    best_val_loss = float('inf')
    ckpt_path = os.path.join(args.checkpoint_dir, 'yolo_synth_best.pt')

    print(f'Training LogicYOLOv1Tiny  '
          f'(S={args.S}, C={args.C}, Q={args.Q}, '
          f'hidden={args.hidden_dim}×{args.num_layers})')
    print(f'  Input bits: {model.preprocess.out_dim}  '
          f'→  GroupSum k={model.total_out}')
    print(f'  Train samples: {train_len}  Val: {val_len}')
    print()

    for epoch in range(1, args.epochs + 1):
        # ---- train ---------------------------------------------------
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs    = imgs.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            pred  = model(imgs)
            loss  = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- validation ----------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs    = imgs.to(args.device)
                targets = targets.to(args.device)
                pred    = model(imgs)
                val_loss += criterion(pred, targets).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'({elapsed:.1f}s)')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_cfg,
            }, ckpt_path)

    print(f'\nBest val loss: {best_val_loss:.4f}')
    print(f'Checkpoint saved to: {ckpt_path}')


if __name__ == '__main__':
    main()
