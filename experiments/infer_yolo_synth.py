"""
Run inference with a trained LogicYOLOv1Tiny on synthetic shapes images and
print decoded detections per image.

Quick start
-----------
::

    python experiments/infer_yolo_synth.py --checkpoint checkpoints/yolo_synth_best.pt
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.synth_shapes import SynthShapesDataset, _SHAPE_NAMES
from models.logic_yolov1 import LogicYOLOv1Tiny
from postprocess.yolo_decode import decode_predictions


def parse_args():
    p = argparse.ArgumentParser(description='Inference with LogicYOLOv1Tiny')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-images', type=int, default=8)
    p.add_argument('--score-thresh', type=float, default=0.3)
    p.add_argument('--iou-thresh',   type=float, default=0.5)
    p.add_argument('--device',  default='cpu')
    p.add_argument('--seed',    type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg  = ckpt['config']
    model = LogicYOLOv1Tiny(**cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(args.device)

    print(f'Loaded checkpoint from {args.checkpoint}')
    print(f'  S={cfg["S"]}, C={cfg["C"]}, Q={cfg["Q"]}')

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = SynthShapesDataset(
        num_samples=args.num_images,
        img_size=(cfg['img_h'], cfg['img_w']),
        S=cfg['S'], C=cfg['C'], Q=cfg['Q'],
        seed=args.seed,
    )

    class_names = _SHAPE_NAMES[:cfg['C']]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    for i in range(args.num_images):
        img, target = dataset[i]
        ann = dataset.get_annotation(i)

        img_batch = img.unsqueeze(0).to(args.device)

        with torch.no_grad():
            raw = model(img_batch)

        dets = decode_predictions(
            raw, S=cfg['S'], C=cfg['C'], Q=cfg['Q'],
            score_thresh=args.score_thresh,
            iou_thresh=args.iou_thresh,
        )[0]

        gt_label = class_names[ann['class_id']]
        gt_box   = (ann['cx_norm'], ann['cy_norm'], ann['bw'], ann['bh'])

        print(f'--- Image {i} ---')
        print(f'  GT : class={gt_label}  box=(cx={gt_box[0]:.3f}, cy={gt_box[1]:.3f}, '
              f'w={gt_box[2]:.3f}, h={gt_box[3]:.3f})')

        if dets.shape[0] == 0:
            print('  Pred: (no detections above threshold)')
        else:
            for d in dets:
                cx, cy, w, h, score, cls_id = d.tolist()
                name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else '?'
                print(f'  Pred: class={name}  score={score:.3f}  '
                      f'box=(cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f})')


if __name__ == '__main__':
    main()
