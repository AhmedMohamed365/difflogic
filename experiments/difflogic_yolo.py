import argparse
import os
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from difflogic import GroupSum, LogicLayer


@dataclass
class DetectionConfig:
    image_size: int = 32
    grid_size: int = 8
    classes: int = 80

    @property
    def outputs_per_cell(self) -> int:
        # Efficient alternative to coordinate bin classification:
        # objectness + classes + direct box regression (x, y, w, h)
        return 1 + self.classes + 4

    @property
    def total_outputs(self) -> int:
        return self.grid_size * self.grid_size * self.outputs_per_cell


class PreprocessToBits(torch.nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.threshold).to(torch.float32)


class DifflogicYoloLite(torch.nn.Module):
    """
    CPU-friendly difflogic detector with a lightweight regression head.
    """

    def __init__(self, cfg: DetectionConfig, tau: float = 8.0):
        super().__init__()
        in_dim = cfg.image_size * cfg.image_size * 3

        self.cfg = cfg
        self.model = torch.nn.Sequential(
            PreprocessToBits(0.5),
            torch.nn.Flatten(),
            LogicLayer(in_dim=in_dim, out_dim=2048, device='cpu', implementation='python', connections='unique'),
            LogicLayer(in_dim=2048, out_dim=1024, device='cpu', implementation='python', connections='unique'),
            LogicLayer(in_dim=1024, out_dim=1024, device='cpu', implementation='python', connections='unique'),
            LogicLayer(in_dim=1024, out_dim=cfg.total_outputs, device='cpu', implementation='python', connections='unique'),
            GroupSum(k=cfg.total_outputs, tau=tau, device='cpu'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return y.view(x.shape[0], self.cfg.grid_size, self.cfg.grid_size, self.cfg.outputs_per_cell)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_coco128_dataset(data_root: Path) -> Path:
    dataset_root = data_root / 'coco128'
    train_images = dataset_root / 'images' / 'train2017'
    train_labels = dataset_root / 'labels' / 'train2017'
    if train_images.exists() and train_labels.exists():
        return dataset_root

    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / 'coco128.zip'
    url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'
    print(f'Downloading COCO128 subset from {url} ...')
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_root)

    extracted = data_root / 'coco128'
    if not extracted.exists():
        raise RuntimeError('Failed to extract coco128 dataset archive.')
    return extracted


class CocoYoloLabelDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 32):
        self.image_dir = root / 'images' / 'train2017'
        self.label_dir = root / 'labels' / 'train2017'
        self.image_size = image_size
        self.image_paths = sorted(self.image_dir.glob('*.jpg'))
        if len(self.image_paths) == 0:
            raise RuntimeError(f'No images found in {self.image_dir}')

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_labels(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        labels: List[Tuple[int, float, float, float, float]] = []
        if not label_path.exists():
            return labels
        for line in label_path.read_text().splitlines():
            fields = line.strip().split()
            if len(fields) != 5:
                continue
            cls, x, y, w, h = fields
            labels.append((int(float(cls)), float(x), float(y), float(w), float(h)))
        return labels

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.label_dir / f'{image_path.stem}.txt'

        image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        image_t = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        labels = self._read_labels(label_path)
        return image_t, labels


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    return images, labels


def encode_targets(batch_labels, cfg: DetectionConfig, device: torch.device):
    bsz = len(batch_labels)
    obj = torch.zeros((bsz, cfg.grid_size, cfg.grid_size), dtype=torch.float32, device=device)
    cls = torch.full((bsz, cfg.grid_size, cfg.grid_size), -1, dtype=torch.long, device=device)
    box = torch.zeros((bsz, cfg.grid_size, cfg.grid_size, 4), dtype=torch.float32, device=device)

    for bi, labels in enumerate(batch_labels):
        for (c, x, y, w, h) in labels:
            if c >= cfg.classes:
                continue
            gx = min(cfg.grid_size - 1, int(x * cfg.grid_size))
            gy = min(cfg.grid_size - 1, int(y * cfg.grid_size))

            obj[bi, gy, gx] = 1.0
            cls[bi, gy, gx] = int(c)

            x_cell = (x * cfg.grid_size) - gx
            y_cell = (y * cfg.grid_size) - gy
            box[bi, gy, gx, 0] = x_cell
            box[bi, gy, gx, 1] = y_cell
            box[bi, gy, gx, 2] = min(0.999, w)
            box[bi, gy, gx, 3] = min(0.999, h)

    return obj, cls, box


def generalized_box_iou_loss(pred_xywh: torch.Tensor, tgt_xywh: torch.Tensor, eps: float = 1e-6):
    # input shapes: [N, 4], format cx, cy, w, h in normalized [0,1]
    px1 = pred_xywh[:, 0] - pred_xywh[:, 2] / 2
    py1 = pred_xywh[:, 1] - pred_xywh[:, 3] / 2
    px2 = pred_xywh[:, 0] + pred_xywh[:, 2] / 2
    py2 = pred_xywh[:, 1] + pred_xywh[:, 3] / 2

    tx1 = tgt_xywh[:, 0] - tgt_xywh[:, 2] / 2
    ty1 = tgt_xywh[:, 1] - tgt_xywh[:, 3] / 2
    tx2 = tgt_xywh[:, 0] + tgt_xywh[:, 2] / 2
    ty2 = tgt_xywh[:, 1] + tgt_xywh[:, 3] / 2

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)

    inter_w = torch.clamp(ix2 - ix1, min=0.0)
    inter_h = torch.clamp(iy2 - iy1, min=0.0)
    inter = inter_w * inter_h

    area_p = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    area_t = torch.clamp(tx2 - tx1, min=0.0) * torch.clamp(ty2 - ty1, min=0.0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c_area = torch.clamp(cx2 - cx1, min=0.0) * torch.clamp(cy2 - cy1, min=0.0) + eps

    giou = iou - (c_area - union) / c_area
    return 1.0 - giou.mean()


def detection_loss(pred: torch.Tensor, obj_t, cls_t, box_t, cfg: DetectionConfig):
    obj_logits = pred[..., 0]
    cls_logits = pred[..., 1:1 + cfg.classes]
    box_raw = pred[..., 1 + cfg.classes:1 + cfg.classes + 4]
    box_pred = torch.sigmoid(box_raw)

    obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(obj_logits, obj_t)
    positive = obj_t > 0.5

    if positive.any():
        cls_loss = torch.nn.functional.cross_entropy(cls_logits[positive], cls_t[positive])

        # Smooth L1 for stable early training + GIoU for geometry quality
        l1_loss = torch.nn.functional.smooth_l1_loss(box_pred[positive], box_t[positive])
        giou_loss = generalized_box_iou_loss(box_pred[positive], box_t[positive])
        box_loss = 0.5 * l1_loss + 0.5 * giou_loss
    else:
        cls_loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        box_loss = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

    total = obj_loss + cls_loss + 2.0 * box_loss
    return total, {'obj': obj_loss.item(), 'cls': cls_loss.item(), 'box': box_loss.item()}


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    losses = []
    for images, labels in loader:
        images = images.to(device)
        obj_t, cls_t, box_t = encode_targets(labels, cfg, device)

        pred = model(images)
        loss, _ = detection_loss(pred, obj_t, cls_t, box_t, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CPU-only difflogic detector on COCO128 with direct box regression.')
    parser.add_argument('--data-root', type=str, default='./data-coco')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--classes', type=int, default=80)
    parser.add_argument('--tau', type=float, default=8.0)
    parser.add_argument('--max-samples', type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    device = torch.device('cpu')

    cfg = DetectionConfig(classes=args.classes)
    dataset_root = ensure_coco128_dataset(Path(args.data_root))
    dataset = CocoYoloLabelDataset(dataset_root, image_size=cfg.image_size)

    if args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(args.max_samples)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = DifflogicYoloLite(cfg=cfg, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print('Starting CPU training on COCO128 subset (direct box regression head)...')
    history = []
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, loader, optimizer, cfg, device)
        history.append(avg_loss)
        print(f'Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}')

    if len(history) >= 2:
        trend = 'decreased' if history[-1] < history[0] else 'did not decrease'
        print(f'Loss trend across epochs: {history[0]:.6f} -> {history[-1]:.6f} ({trend}).')


if __name__ == '__main__':
    main()
