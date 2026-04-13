"""
Train LogicYOLOv1Tiny on mini-COCO (coco128) using YOLO txt annotations.

Example:
    python experiments/train_yolo_coco.py \
      --download-mini-coco \
      --data-root data/coco128 \
      --class-ids 15 16 \
      --img-size 96 \
      --S 8 --Q 16 \
      --epochs 20 --batch-size 16
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, random_split

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.mini_coco import COCO80_NAMES, MiniCOCOYOLODataset
from losses.yolo_bins import YOLOBinsLoss
from models.logic_yolov1 import LogicYOLOv1Tiny


def parse_args():
    p = argparse.ArgumentParser(description='Train LogicYOLOv1Tiny on mini-COCO (coco128)')
    # data
    p.add_argument('--data-root', default='data/coco128')
    p.add_argument('--split', default='train2017')
    p.add_argument('--download-mini-coco', action='store_true')
    p.add_argument('--class-ids', type=int, nargs='+', default=[15, 16],  # cat, dog
                   help='COCO global class IDs to keep')
    p.add_argument('--max-samples', type=int, default=0,
                   help='0=no cap, else random subset size')
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep-empty', action='store_true',
                   help='Keep images with no selected classes')
    # model
    p.add_argument('--img-size', type=int, default=96)
    p.add_argument('--S', type=int, default=8)
    p.add_argument('--Q', type=int, default=16)
    p.add_argument('--num-thresholds', type=int, default=2)
    p.add_argument('--hidden-dim', type=int, default=16384)
    p.add_argument('--num-layers', type=int, default=4)
    p.add_argument('--tau', type=float, default=10.0)
    p.add_argument('--grad-factor', type=float, default=1.0)
    # train
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lambda-coord', type=float, default=5.0)
    p.add_argument('--lambda-noobj', type=float, default=0.5)
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--checkpoint-name', default='yolo_coco128_best.pt')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset = MiniCOCOYOLODataset(
        data_root=args.data_root,
        split=args.split,
        img_size=(args.img_size, args.img_size),
        S=args.S,
        Q=args.Q,
        class_ids=args.class_ids,
        max_samples=args.max_samples,
        seed=args.seed,
        download=args.download_mini_coco,
        keep_empty=args.keep_empty,
    )

    C = dataset.C
    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # LogicLayer requires out_dim*2 >= in_dim for each layer.
    in_dim = args.img_size * args.img_size * 3 * args.num_thresholds
    min_hidden = (in_dim + 1) // 2
    if args.hidden_dim < min_hidden:
        raise ValueError(
            f'hidden-dim={args.hidden_dim} is too small for img-size={args.img_size} and '
            f'num-thresholds={args.num_thresholds}. Need hidden-dim >= {min_hidden}.'
        )

    model_cfg = dict(
        S=args.S, C=C, Q=args.Q,
        num_thresholds=args.num_thresholds,
        img_channels=3,
        img_h=args.img_size, img_w=args.img_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        tau=args.tau,
        device=args.device,
        grad_factor=args.grad_factor,
    )
    model = LogicYOLOv1Tiny(**model_cfg).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = YOLOBinsLoss(
        S=args.S, C=C, Q=args.Q,
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
    )

    class_names = [COCO80_NAMES[cid] if 0 <= cid < len(COCO80_NAMES) else f'class_{cid}'
                   for cid in args.class_ids]
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    best_val_loss = float('inf')

    print(f'Training LogicYOLOv1Tiny on mini-COCO')
    print(f'  Classes ({C}): {class_names}')
    print(f'  Samples: train={train_len}, val={val_len}')
    print(f'  Model: S={args.S}, Q={args.Q}, hidden={args.hidden_dim}x{args.num_layers}')
    print(f'  Input bits: {model.preprocess.out_dim} -> GroupSum k={model.total_out}')
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(args.device)
                targets = targets.to(args.device)
                pred = model(imgs)
                val_loss += float(criterion(pred, targets).item())
        val_loss /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  ({dt:.1f}s)')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_cfg,
                'dataset': {
                    'name': 'mini-coco',
                    'class_ids': args.class_ids,
                    'class_names': class_names,
                },
            }, ckpt_path)

    print(f'\nBest val loss: {best_val_loss:.4f}')
    print(f'Checkpoint saved to: {ckpt_path}')


if __name__ == '__main__':
    main()
