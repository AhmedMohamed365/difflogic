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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.mini_coco import COCO80_NAMES, MiniCOCOYOLODataset
from models.logic_yolov1 import LogicYOLOv1Tiny


# -----------------------------------------------------------------------------
# Enhanced Loss Function with Focal Objectness, Label Smoothing, and Soft Bins
# -----------------------------------------------------------------------------
class EnhancedYOLOBinsLoss(nn.Module):
    """
    Enhanced loss for LogicYOLO grid predictions with binned coordinates.
    
    Features:
        - Focal loss for objectness (auto down-weights easy background cells)
        - Label smoothing for classification (prevents overconfidence)
        - Optional soft bin assignment (Gaussian around true bin for smoother gradients)
        - Coordinate term scaling
    
    Args:
        S (int): Grid size (e.g., 8)
        C (int): Number of classes
        Q (int): Number of quantization bins per coordinate
        lambda_coord (float): Weight for coordinate losses (default 5.0)
        lambda_noobj (float): Weight for no-object confidence (kept for compatibility,
                              but focal loss makes it less critical)
        focal_gamma (float): Gamma parameter for focal loss on objectness (2.0 works well)
        label_smoothing (float): Smoothing factor for bin classification (0.1)
        soft_bin_sigma (float): If >0, apply Gaussian soft assignment around true bin
        coord_weight (float): Additional scaling for coordinate terms relative to class
    """
    def __init__(
        self,
        S: int,
        C: int,
        Q: int,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        soft_bin_sigma: float = 0.5,   # set to 0 to disable soft bins
        coord_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.S = S
        self.C = C
        self.Q = Q
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.soft_bin_sigma = soft_bin_sigma
        self.coord_weight = coord_weight
        self.reduction = reduction
        
        # Precompute bin centers for soft assignment
        if soft_bin_sigma > 0:
            bin_centers = torch.linspace(0, 1, Q)
            self.register_buffer('bin_centers', bin_centers)
        
        # For logging
        self.last_losses = {}

    def forward(
        self, 
        pred: torch.Tensor,      # [B, S, S, 1 + C + 4*Q]
        target: torch.Tensor,    # [B, S, S, 1 + C + 4*Q] (one-hot bins)
        mask: Optional[torch.Tensor] = None  # optional ignore mask (e.g., for empty cells)
    ) -> torch.Tensor:
        """
        Compute loss.
        """
        B = pred.shape[0]
        
        # Split predictions and targets
        pred_obj = pred[..., 0]                     # [B, S, S]
        pred_cls = pred[..., 1:1+self.C]            # [B, S, S, C]
        pred_x = pred[..., 1+self.C : 1+self.C+self.Q]
        pred_y = pred[..., 1+self.C+self.Q : 1+self.C+2*self.Q]
        pred_w = pred[..., 1+self.C+2*self.Q : 1+self.C+3*self.Q]
        pred_h = pred[..., 1+self.C+3*self.Q : 1+self.C+4*self.Q]
        
        target_obj = target[..., 0]                  # [B, S, S]
        target_cls = target[..., 1:1+self.C]
        target_x = target[..., 1+self.C : 1+self.C+self.Q]
        target_y = target[..., 1+self.C+self.Q : 1+self.C+2*self.Q]
        target_w = target[..., 1+self.C+2*self.Q : 1+self.C+3*self.Q]
        target_h = target[..., 1+self.C+3*self.Q : 1+self.C+4*self.Q]
        
        # Object mask: cells that contain an object center
        obj_mask = target_obj > 0.5                  # [B, S, S]
        noobj_mask = ~obj_mask
        
        # If external mask provided (e.g., padded cells), apply it
        if mask is not None:
            obj_mask = obj_mask & mask
            noobj_mask = noobj_mask & mask
        
        # ---------------------------
        # 1. Objectness Loss (Focal)
        # ---------------------------
        obj_logits = pred_obj  # assuming sigmoid applied in model? 
        obj_probs = torch.sigmoid(obj_logits)
        
        # Focal weight: (1 - pt)^gamma
        pt = torch.where(obj_mask, obj_probs, 1 - obj_probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            obj_logits, target_obj, reduction='none'
        )
        
        # Apply focal weight and separate object/no-object scaling
        obj_loss = focal_weight * bce_loss
        obj_loss = torch.where(
            obj_mask,
            obj_loss,                       # object cells: weight=1
            self.lambda_noobj * obj_loss    # no-object cells: downweight
        )
        
        # ---------------------------
        # 2. Class Loss (only where object)
        # ---------------------------
        if obj_mask.any():
            pred_cls_obj = pred_cls[obj_mask]          # [N_obj, C]
            target_cls_obj = target_cls[obj_mask]      # [N_obj, C]
            cls_loss = self._cross_entropy_with_smoothing(
                pred_cls_obj, target_cls_obj, self.label_smoothing
            )
        else:
            cls_loss = torch.tensor(0.0, device=pred.device)
        
        # ---------------------------
        # 3. Coordinate Bin Losses
        # ---------------------------
        coord_losses = []
        for pred_bin, target_bin in [
            (pred_x, target_x), (pred_y, target_y),
            (pred_w, target_w), (pred_h, target_h)
        ]:
            if obj_mask.any():
                pred_bin_obj = pred_bin[obj_mask]      # [N_obj, Q]
                target_bin_obj = target_bin[obj_mask]  # [N_obj, Q]
                
                if self.soft_bin_sigma > 0:
                    # Convert one-hot to index
                    bin_idx = target_bin_obj.argmax(dim=-1)  # [N_obj]
                    soft_target = self._gaussian_soft_bins(bin_idx, self.Q, self.soft_bin_sigma)
                    soft_target = soft_target.to(pred_bin_obj.device)
                    log_probs = F.log_softmax(pred_bin_obj, dim=-1)
                    ce = -(soft_target * log_probs).sum(dim=-1).mean()
                else:
                    ce = self._cross_entropy_with_smoothing(
                        pred_bin_obj, target_bin_obj, self.label_smoothing
                    )
                coord_losses.append(ce)
            else:
                coord_losses.append(torch.tensor(0.0, device=pred.device))
        
        coord_loss = sum(coord_losses) * self.coord_weight
        
        # ---------------------------
        # 4. Combine Losses
        # ---------------------------
        obj_loss = obj_loss.mean() if self.reduction == 'mean' else obj_loss.sum()
        total_loss = obj_loss + cls_loss + self.lambda_coord * coord_loss
        
        # Store components for logging
        self.last_losses = {
            'obj': obj_loss.item(),
            'cls': cls_loss.item() if torch.is_tensor(cls_loss) else cls_loss,
            'coord': coord_loss.item() if torch.is_tensor(coord_loss) else coord_loss,
            'total': total_loss.item()
        }
        
        return total_loss
    
    def _cross_entropy_with_smoothing(self, pred, target, smoothing):
        """
        Cross entropy with label smoothing.
        pred: [N, K] logits
        target: [N, K] one-hot
        """
        n_classes = pred.size(-1)
        target_idx = target.argmax(dim=-1)
        
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target_idx.unsqueeze(-1)).squeeze(-1)
        
        if smoothing > 0:
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
        else:
            loss = nll_loss
            
        return loss.mean()
    
    def _gaussian_soft_bins(self, bin_idx, Q, sigma):
        """
        Create soft bin targets using Gaussian distribution around true bin.
        bin_idx: [N] integer bin indices
        Q: total bins
        sigma: standard deviation in bin units
        Returns: [N, Q] soft probability distribution
        """
        device = bin_idx.device
        bins = torch.arange(Q, device=device).float()
        diff = bins.unsqueeze(0) - bin_idx.unsqueeze(1).float()  # [N, Q]
        weights = torch.exp(-0.5 * (diff / sigma) ** 2)
        soft_targets = weights / weights.sum(dim=1, keepdim=True)
        return soft_targets


# -----------------------------------------------------------------------------
# Utility: Compute bin accuracy for monitoring
# -----------------------------------------------------------------------------
def compute_bin_accuracy(pred: torch.Tensor, target: torch.Tensor, S: int, C: int, Q: int):
    """
    Compute per-coordinate bin accuracy (only on object cells).
    Returns dict with accuracy for objectness, class, x, y, w, h.
    """
    obj_mask = target[..., 0] > 0.5  # [B, S, S]
    if not obj_mask.any():
        return {'obj': 0.0, 'cls': 0.0, 'x': 0.0, 'y': 0.0, 'w': 0.0, 'h': 0.0}
    
    # Objectness accuracy
    pred_obj_bin = (torch.sigmoid(pred[..., 0]) > 0.5).float()
    obj_acc = (pred_obj_bin == obj_mask.float()).float().mean().item()
    
    # Class accuracy
    pred_cls = pred[..., 1:1+C].argmax(dim=-1)      # [B, S, S]
    true_cls = target[..., 1:1+C].argmax(dim=-1)    # [B, S, S]
    cls_acc = (pred_cls[obj_mask] == true_cls[obj_mask]).float().mean().item()
    
    # Coordinate accuracies
    coord_names = ['x', 'y', 'w', 'h']
    acc_dict = {'obj': obj_acc, 'cls': cls_acc}
    for i, name in enumerate(coord_names):
        start = 1 + C + i*Q
        pred_bin = pred[..., start:start+Q].argmax(dim=-1)   # [B, S, S]
        true_bin = target[..., start:start+Q].argmax(dim=-1)
        acc = (pred_bin[obj_mask] == true_bin[obj_mask]).float().mean().item()
        acc_dict[name] = acc
    return acc_dict


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Train LogicYOLOv1Tiny on mini-COCO (coco128)')
    # Data
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
    # Model
    p.add_argument('--img-size', type=int, default=96)
    p.add_argument('--S', type=int, default=8)
    p.add_argument('--Q', type=int, default=16)
    p.add_argument('--num-thresholds', type=int, default=2)
    p.add_argument('--hidden-dim', type=int, default=16384)
    p.add_argument('--num-layers', type=int, default=4)
    p.add_argument('--tau', type=float, default=10.0)
    p.add_argument('--grad-factor', type=float, default=1.0)
    # Training
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lambda-coord', type=float, default=5.0)
    p.add_argument('--lambda-noobj', type=float, default=0.5)
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--soft-bin-sigma', type=float, default=0.5)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--checkpoint-name', default='yolo_coco128_best.pt')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
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

    # Model dimension validation
    in_dim = args.img_size * args.img_size * 3 * args.num_thresholds
    min_hidden = (in_dim + 1) // 2
    if args.hidden_dim < min_hidden:
        raise ValueError(
            f'hidden-dim={args.hidden_dim} is too small for img-size={args.img_size} and '
            f'num-thresholds={args.num_thresholds}. Need hidden-dim >= {min_hidden}.'
        )

    # Model
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

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-5)

    # Enhanced Loss
    criterion = EnhancedYOLOBinsLoss(
        S=args.S, C=C, Q=args.Q,
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        soft_bin_sigma=args.soft_bin_sigma,
        coord_weight=1.0,
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
    print(f'  Loss: Focal γ={args.focal_gamma}, LabelSmooth={args.label_smoothing}, SoftBinσ={args.soft_bin_sigma}')
    print()

    # Warmup scheduler helper
    def get_lr(epoch):
        if epoch < args.warmup_epochs:
            return args.lr * (epoch + 1) / args.warmup_epochs
        else:
            return scheduler.get_last_lr()[0]

    for epoch in range(1, args.epochs + 1):
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch - 1)

        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_obj_loss = 0.0
        train_cls_loss = 0.0
        train_coord_loss = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, targets)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            train_obj_loss += criterion.last_losses.get('obj', 0)
            train_cls_loss += criterion.last_losses.get('cls', 0)
            train_coord_loss += criterion.last_losses.get('coord', 0)

        num_batches = len(train_loader)
        train_loss /= num_batches
        train_obj_loss /= num_batches
        train_cls_loss /= num_batches
        train_coord_loss /= num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = {'obj': 0.0, 'cls': 0.0, 'x': 0.0, 'y': 0.0, 'w': 0.0, 'h': 0.0}
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(args.device)
                targets = targets.to(args.device)
                pred = model(imgs)
                val_loss += criterion(pred, targets).item()
                
                # Compute bin accuracy (only for first few batches to save time)
                if len(val_acc) < 10:  # heuristic
                    batch_acc = compute_bin_accuracy(pred, targets, args.S, C, args.Q)
                    for k in val_acc:
                        val_acc[k] += batch_acc[k]

        val_loss /= len(val_loader)
        for k in val_acc:
            val_acc[k] /= len(val_loader)

        dt = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs} | LR: {get_lr(epoch-1):.2e} | '
              f'Train Loss: {train_loss:.4f} (obj={train_obj_loss:.3f}, cls={train_cls_loss:.3f}, coord={train_coord_loss:.3f}) | '
              f'Val Loss: {val_loss:.4f} | Acc(x/y/w/h): {val_acc["x"]:.3f}/{val_acc["y"]:.3f}/{val_acc["w"]:.3f}/{val_acc["h"]:.3f} | '
              f'Time: {dt:.1f}s')

        # Step scheduler after warmup
        if epoch > args.warmup_epochs:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_cfg,
                'args': args,
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