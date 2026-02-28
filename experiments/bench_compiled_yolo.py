"""
Benchmark PyTorch-eval vs CompiledLogicNet throughput for LogicYOLOv1Tiny.

Quick start
-----------
::

    python experiments/bench_compiled_yolo.py --checkpoint checkpoints/yolo_synth_best.pt

The script:
1. Loads a trained checkpoint.
2. Exports the backbone to a ``CompiledLogicNet``.
3. Runs both the PyTorch-eval model and the compiled model on the same
   random inputs and checks output consistency.
4. Reports per-image throughput (ms / image) for both paths.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.logic_yolov1 import LogicYOLOv1Tiny
from export.export_compiled_yolo import export_compiled, binarize_inputs


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark LogicYOLOv1Tiny PyTorch vs compiled')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-batches', type=int, default=5,
                   help='Number of batches to time (first is warmup)')
    p.add_argument('--num-bits', type=int, default=64)
    p.add_argument('--compiler', default='gcc', choices=['gcc', 'clang'])
    p.add_argument('--opt-level', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg  = ckpt['config']
    model = LogicYOLOv1Tiny(**cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f'Model: S={cfg["S"]}, C={cfg["C"]}, Q={cfg["Q"]}, '
          f'K={model.K}, total_out={model.total_out}')

    # ------------------------------------------------------------------
    # Compile backbone
    # ------------------------------------------------------------------
    print('\nCompiling backbone ...')
    t0 = time.time()
    compiled = export_compiled(
        model,
        num_bits=args.num_bits,
        cpu_compiler=args.compiler,
        opt_level=args.opt_level,
        verbose=False,
    )
    print(f'Compiled in {time.time() - t0:.1f}s')

    # ------------------------------------------------------------------
    # Build random test batch
    # ------------------------------------------------------------------
    imgs = torch.rand(args.batch_size, cfg['img_channels'],
                      cfg['img_h'], cfg['img_w'])

    # PyTorch preprocessing (float) → shape [N, in_dim]
    with torch.no_grad():
        bits_torch = model.preprocess(imgs)          # float {0,1}

    # Bool numpy for compiled
    bits_np = bits_torch.numpy().astype(bool)        # [N, in_dim]

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------
    with torch.no_grad():
        pt_flat   = model.backbone(bits_torch)       # [N, total_out]
    compiled_out = compiled(bits_np)                  # [N, total_out]  int32 sums

    # Both should be monotonically consistent: higher sum → model prefers that
    # output neuron.  We compare argmax of each cell's component.
    pt_np   = pt_flat.numpy()
    comp_np = compiled_out.numpy().astype(float)

    # Reshape to [N, S*S, K]
    S, K = model.S, model.K
    pt_reshaped   = pt_np.reshape(args.batch_size, S * S, K)
    comp_reshaped = comp_np.reshape(args.batch_size, S * S, K)

    # Compare argmax for class & coords
    C, Q = model.C, model.Q
    cls_pt   = pt_reshaped[..., 1:1 + C].argmax(-1)
    cls_comp = comp_reshaped[..., 1:1 + C].argmax(-1)
    cls_match = (cls_pt == cls_comp).mean()

    coord_matches = []
    for ci in range(4):
        off = 1 + C + ci * Q
        a = pt_reshaped[..., off:off + Q].argmax(-1)
        b = comp_reshaped[..., off:off + Q].argmax(-1)
        coord_matches.append((a == b).mean())

    print(f'\nCorrectness check (argmax agreement):')
    print(f'  class argmax match : {cls_match * 100:.1f}%')
    for ci, name in enumerate('xywh'):
        print(f'  {name} coord argmax  : {coord_matches[ci] * 100:.1f}%')

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    def time_pytorch(n_reps):
        times = []
        for _ in range(n_reps):
            t = time.time()
            with torch.no_grad():
                model.backbone(bits_torch)
            times.append(time.time() - t)
        return times

    def time_compiled(n_reps):
        times = []
        for _ in range(n_reps):
            t = time.time()
            compiled(bits_np)
            times.append(time.time() - t)
        return times

    print(f'\nTiming ({args.batch_size} images/batch, '
          f'{args.num_batches} batches, first is warmup)')

    t_pt   = time_pytorch(args.num_batches)
    t_comp = time_compiled(args.num_batches)

    # drop warmup
    t_pt   = t_pt[1:]
    t_comp = t_comp[1:]

    ms_pt   = 1000.0 * sum(t_pt)   / len(t_pt)   / args.batch_size
    ms_comp = 1000.0 * sum(t_comp) / len(t_comp)  / args.batch_size

    print(f'  PyTorch-eval       : {ms_pt:.3f} ms / image')
    print(f'  CompiledLogicNet   : {ms_comp:.3f} ms / image')
    speedup = ms_pt / ms_comp if ms_comp > 0 else float('inf')
    print(f'  Speedup (compiled) : {speedup:.1f}×')


if __name__ == '__main__':
    main()
