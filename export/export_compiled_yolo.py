"""
Export a trained LogicYOLOv1Tiny backbone to a CompiledLogicNet.

The backbone (``model.backbone``) is a ``torch.nn.Sequential`` containing
only ``LogicLayer``, ``GroupSum``, and ``torch.nn.Flatten`` layers — which
is exactly what ``CompiledLogicNet`` supports.

The compiled model outputs ``[batch, S*S*K]`` raw integer sums (GroupSum
counts).  These should be reshaped to ``[batch, S, S, K]`` in Python for
decoding, just like the PyTorch eval model.

Usage
-----
::

    python -m export.export_compiled_yolo \\
        --checkpoint path/to/checkpoint.pt \\
        --save-lib    path/to/yolo_logic.so

See ``--help`` for all options.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Allow running from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from difflogic import CompiledLogicNet
from models.logic_yolov1 import LogicYOLOv1Tiny


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def binarize_inputs(imgs: torch.Tensor, model: LogicYOLOv1Tiny) -> np.ndarray:
    """
    Apply ``model.preprocess`` and convert to a bool numpy array suitable
    for ``CompiledLogicNet``.

    Returns: bool ndarray [N, in_dim].
    """
    with torch.no_grad():
        bits = model.preprocess(imgs)  # [N, in_dim] float32 in {0., 1.}
    return bits.numpy().astype(bool)


def export_compiled(
    model: LogicYOLOv1Tiny,
    save_lib_path: str = None,
    num_bits: int = 64,
    cpu_compiler: str = 'gcc',
    opt_level: int = 0,
    verbose: bool = False,
) -> CompiledLogicNet:
    """
    Export ``model.backbone`` to a ``CompiledLogicNet``.

    Args:
        model        : trained ``LogicYOLOv1Tiny`` in eval mode.
        save_lib_path: optional path to save the compiled ``.so`` file.
        num_bits     : packing bits for the compiled net (8/16/32/64).
        cpu_compiler : ``'gcc'`` or ``'clang'``.
        opt_level    : C compiler optimisation level (0–3).
        verbose      : print compilation details.

    Returns:
        compiled ``CompiledLogicNet`` instance ready for inference.
    """
    assert not model.training, 'model must be in eval() mode before export'

    compiled = CompiledLogicNet(
        model=model.backbone,
        device='cpu',
        num_bits=num_bits,
        cpu_compiler=cpu_compiler,
        verbose=verbose,
    )
    compiled.compile(opt_level=opt_level, save_lib_path=save_lib_path,
                     verbose=verbose)
    return compiled


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Export LogicYOLOv1Tiny backbone to CompiledLogicNet'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to checkpoint .pt file saved by train_yolo_synth.py')
    parser.add_argument('--save-lib', default=None,
                        help='Optional path to save the compiled .so library')
    parser.add_argument('--num-bits', type=int, default=64, choices=[8, 16, 32, 64],
                        help='Packing bits for the compiled net')
    parser.add_argument('--compiler', default='gcc', choices=['gcc', 'clang'])
    parser.add_argument('--opt-level', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt['config']

    model = LogicYOLOv1Tiny(**cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    compiled = export_compiled(
        model,
        save_lib_path=args.save_lib,
        num_bits=args.num_bits,
        cpu_compiler=args.compiler,
        opt_level=args.opt_level,
        verbose=args.verbose,
    )

    print('Export complete.')
    if args.save_lib:
        print(f'Compiled library saved to: {args.save_lib}')

    return compiled


if __name__ == '__main__':
    main()
