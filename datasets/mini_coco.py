"""
Mini-COCO (coco128) dataset adapter for LogicYOLOv1Tiny training.

The dataset is expected in YOLO txt format:
    <class_id> <cx> <cy> <w> <h>
where coordinates are normalised to [0, 1].

This adapter maps YOLO labels to difflogic YOLO-bin targets:
    [S, S, 1 + C + 4*Q]
with at most one object per cell (YOLOv1-style).
"""

from __future__ import annotations

import os
import random
import urllib.request
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.synth_shapes import YOLOTargetBuilder


COCO128_URL = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'

COCO80_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def _resolve_coco128_root(data_root: str) -> str:
    # If caller points to extracted root directly
    if os.path.isdir(os.path.join(data_root, 'images')) and os.path.isdir(os.path.join(data_root, 'labels')):
        return data_root
    # If caller points to parent directory containing coco128/
    nested = os.path.join(data_root, 'coco128')
    if os.path.isdir(os.path.join(nested, 'images')) and os.path.isdir(os.path.join(nested, 'labels')):
        return nested
    return data_root


def download_coco128(data_root: str) -> str:
    """
    Download and extract the mini-COCO (coco128) dataset if it is not present.

    Returns the resolved dataset root containing `images/` and `labels/`.
    """
    os.makedirs(data_root, exist_ok=True)
    resolved = _resolve_coco128_root(data_root)
    if os.path.isdir(os.path.join(resolved, 'images')) and os.path.isdir(os.path.join(resolved, 'labels')):
        return resolved

    zip_path = os.path.join(data_root, 'coco128.zip')
    if not os.path.isfile(zip_path):
        print(f'Downloading coco128 from {COCO128_URL}')
        urllib.request.urlretrieve(COCO128_URL, zip_path)

    print(f'Extracting {zip_path}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_root)

    resolved = _resolve_coco128_root(data_root)
    if not (os.path.isdir(os.path.join(resolved, 'images')) and os.path.isdir(os.path.join(resolved, 'labels'))):
        raise RuntimeError(f'Could not locate extracted coco128 dataset under {data_root}')
    return resolved


class MiniCOCOYOLODataset(Dataset):
    """
    Mini-COCO (YOLO txt) dataset adapter for LogicYOLOv1Tiny.

    Args:
        data_root: root containing coco128 files (or parent containing coco128/).
        split: data split folder name under images/ and labels/ (default train2017).
        img_size: model image size (H, W).
        S, Q: YOLO grid size and quantization bins.
        class_ids: selected COCO class IDs (global 0..79). Local class indices are
                   assigned in the same order as this list.
        max_samples: optional cap on dataset size (0 = no cap).
        seed: random seed used when sub-sampling max_samples.
        download: if True, download/extract coco128 automatically.
        keep_empty: if False, skip images with no selected classes.
    """

    def __init__(
        self,
        data_root: str = 'data/coco128',
        split: str = 'train2017',
        img_size: Tuple[int, int] = (96, 96),
        S: int = 8,
        Q: int = 16,
        class_ids: Sequence[int] = (15, 16),  # cat, dog
        max_samples: int = 0,
        seed: int = 42,
        download: bool = False,
        keep_empty: bool = False,
    ):
        if download:
            data_root = download_coco128(data_root)
        data_root = _resolve_coco128_root(data_root)

        self.data_root = data_root
        self.split = split
        self.img_h, self.img_w = img_size
        self.S = S
        self.Q = Q
        self.class_ids = list(class_ids)
        self.C = len(self.class_ids)
        self.class_id_to_local = {cid: i for i, cid in enumerate(self.class_ids)}
        self.class_names = [COCO80_NAMES[cid] if 0 <= cid < len(COCO80_NAMES) else f'class_{cid}'
                            for cid in self.class_ids]

        self.target_builder = YOLOTargetBuilder(S=S, C=self.C, Q=Q)

        images_dir = os.path.join(data_root, 'images', split)
        labels_dir = os.path.join(data_root, 'labels', split)
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            raise FileNotFoundError(
                f'Expected dataset folders not found: {images_dir} and {labels_dir}. '
                f'Use --download-mini-coco or set correct --data-root.'
            )

        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        image_files.sort()

        samples = []
        for image_name in image_files:
            stem, _ = os.path.splitext(image_name)
            label_path = os.path.join(labels_dir, f'{stem}.txt')
            anns = self._read_yolo_labels(label_path)
            anns = [a for a in anns if a['class_id'] in self.class_id_to_local]
            if not anns and not keep_empty:
                continue
            samples.append({
                'image_path': os.path.join(images_dir, image_name),
                'labels': anns,
            })

        if max_samples > 0 and len(samples) > max_samples:
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_samples]

        if len(samples) == 0:
            raise RuntimeError('No usable samples found for selected class_ids.')

        self.samples = samples

    @staticmethod
    def _read_yolo_labels(path: str) -> List[Dict[str, float]]:
        if not os.path.isfile(path):
            return []
        anns: List[Dict[str, float]] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = parts
                anns.append({
                    'class_id': int(float(cls)),
                    'cx': float(cx),
                    'cy': float(cy),
                    'bw': float(bw),
                    'bh': float(bh),
                })
        return anns

    def __len__(self) -> int:
        return len(self.samples)

    def _build_target(self, labels: List[Dict[str, float]]) -> torch.Tensor:
        """
        Build [S,S,K] target with one object per cell.
        If multiple objects map to one cell, keep the larger area.
        """
        target = torch.zeros(self.S, self.S, 1 + self.C + 4 * self.Q, dtype=torch.float32)
        cell_area = torch.full((self.S, self.S), -1.0, dtype=torch.float32)

        for ann in labels:
            class_global = ann['class_id']
            if class_global not in self.class_id_to_local:
                continue
            class_local = self.class_id_to_local[class_global]
            cx = float(np.clip(ann['cx'], 0.0, 1.0 - 1e-6))
            cy = float(np.clip(ann['cy'], 0.0, 1.0 - 1e-6))
            bw = float(np.clip(ann['bw'], 1e-6, 1.0))
            bh = float(np.clip(ann['bh'], 1e-6, 1.0))
            area = bw * bh

            cell_x = min(int(cx * self.S), self.S - 1)
            cell_y = min(int(cy * self.S), self.S - 1)
            if area < float(cell_area[cell_y, cell_x]):
                continue

            single = self.target_builder.build(class_local, (cx, cy, bw, bh))
            target[cell_y, cell_x] = single[cell_y, cell_x]
            cell_area[cell_y, cell_x] = area

        return target

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample['image_path']).convert('RGB')
        img = img.resize((self.img_w, self.img_h), resample=Image.Resampling.BILINEAR)
        img_np = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
        img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1)))  # [3,H,W]

        target = self._build_target(sample['labels'])
        return img_tensor, target
