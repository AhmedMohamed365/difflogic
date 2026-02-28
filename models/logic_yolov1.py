"""
LogicYOLOv1Tiny: a YOLOv1-style object detector built on difflogic primitives.

Design decisions:
- Input images are binarized via multiple pixel-value thresholds (PreprocessToBits).
- The backbone is a stack of LogicLayer modules (flat MLP-style logic-gate network).
- The head is a GroupSum that aggregates neurons into S*S*(1 + C + 4*Q) outputs.
- Output shape: [N, S, S, 1 + C + 4*Q]
  - objectness : 1     (raw logit; use sigmoid at decode time)
  - class      : C     (raw logits; use softmax at decode time)
  - coord bins : 4*Q   (x_bin, y_bin, w_bin, h_bin each with Q bins; use CE)
- B=1 box per cell (can be extended to B>1).
- The sequential (backbone + GroupSum) is export-compatible with CompiledLogicNet
  when PostprocessReshape is placed OUTSIDE the compiled model.
"""

import torch
import torch.nn as nn
import sys
import os

# Ensure difflogic is importable from any working directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from difflogic import LogicLayer, GroupSum


# ---------------------------------------------------------------------------
# Input binarization
# ---------------------------------------------------------------------------

class PreprocessToBits(nn.Module):
    """
    Convert a floating-point image tensor in [0, 1] to a binary (0/1 float)
    vector using multiple thresholds per channel.

    Args:
        num_thresholds: number of thresholds per pixel/channel.
            Threshold values are evenly spaced in (0, 1):
            t_k = (k+1) / (num_thresholds + 1), k = 0..num_thresholds-1.
        img_channels: number of image channels (1 for greyscale, 3 for RGB).
        img_h: image height in pixels.
        img_w: image width in pixels.

    Input:  [N, C, H, W]  (float32, range [0, 1])
    Output: [N, C * H * W * num_thresholds]  (float32, values 0 or 1)
    """

    def __init__(self, num_thresholds: int, img_channels: int, img_h: int, img_w: int):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.img_channels = img_channels
        self.img_h = img_h
        self.img_w = img_w
        thresholds = [(k + 1) / (num_thresholds + 1) for k in range(num_thresholds)]
        # shape [num_thresholds]
        self.register_buffer('thresholds', torch.tensor(thresholds, dtype=torch.float32))

    @property
    def out_dim(self) -> int:
        return self.img_channels * self.img_h * self.img_w * self.num_thresholds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        x_flat = x.reshape(x.shape[0], -1)  # [N, C*H*W]
        # compare against each threshold → [N, C*H*W, num_thresholds]
        bits = (x_flat.unsqueeze(-1) > self.thresholds).float()
        # flatten → [N, C*H*W*num_thresholds]
        return bits.reshape(x.shape[0], -1)

    def extra_repr(self) -> str:
        return (
            f'num_thresholds={self.num_thresholds}, '
            f'channels={self.img_channels}, h={self.img_h}, w={self.img_w}, '
            f'out_dim={self.out_dim}'
        )


# ---------------------------------------------------------------------------
# LogicYOLOv1Tiny
# ---------------------------------------------------------------------------

class LogicYOLOv1Tiny(nn.Module):
    """
    A tiny YOLOv1-style object detector using differentiable logic gate networks.

    The backbone is a pure ``torch.nn.Sequential`` of ``LogicLayer`` + ``GroupSum``
    and is therefore exportable to ``CompiledLogicNet`` (see
    ``export/export_compiled_yolo.py``).  ``PreprocessToBits`` must be applied
    *before* passing data to the sequential backbone, and the output must be
    reshaped from ``[N, S*S*K]`` → ``[N, S, S, K]`` *after* inference.

    Args:
        S            : grid size (S×S cells).
        C            : number of object classes.
        Q            : quantization bins per bbox coordinate.
        num_thresholds: thresholds per pixel for binary encoding.
        img_channels : number of input image channels.
        img_h / img_w: input image spatial size (after any pre-downsampling).
        hidden_dim   : width of hidden LogicLayer neurons.
        num_layers   : number of hidden LogicLayer blocks.
        tau          : GroupSum temperature (sum divisor).
        device       : 'cpu' or 'cuda'.
        grad_factor  : gradient scaling for deep nets (>6 layers → use ~2).
    """

    # Per-cell output layout
    # [obj(1) | class(C) | x_bin(Q) | y_bin(Q) | w_bin(Q) | h_bin(Q)]
    OBJ_OFFSET = 0
    OBJ_LEN = 1

    def __init__(
        self,
        S: int = 8,
        C: int = 3,
        Q: int = 16,
        num_thresholds: int = 2,
        img_channels: int = 1,
        img_h: int = 32,
        img_w: int = 32,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        tau: float = 10.0,
        device: str = 'cpu',
        grad_factor: float = 1.0,
    ):
        super().__init__()

        self.S = S
        self.C = C
        self.Q = Q
        self.tau = tau
        self.device = device

        # per-cell output size: objectness + class + 4 coord bins
        self.K = 1 + C + 4 * Q  # output channels per cell

        # total output neurons for GroupSum
        self.total_out = S * S * self.K

        # --- binarization (NOT part of the exported backbone) ---
        self.preprocess = PreprocessToBits(
            num_thresholds=num_thresholds,
            img_channels=img_channels,
            img_h=img_h,
            img_w=img_w,
        )
        in_dim = self.preprocess.out_dim

        # --- exported sequential backbone ---
        # GroupSum requires: last_logic_layer.out_dim % total_out == 0
        # We set the last hidden dim to be a multiple of total_out.
        expansion = max(1, hidden_dim // self.total_out)
        last_dim = expansion * self.total_out

        layers = [torch.nn.Flatten()]
        prev_dim = in_dim
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out = last_dim if is_last else hidden_dim
            layers.append(
                LogicLayer(
                    in_dim=prev_dim,
                    out_dim=out,
                    device=device,
                    grad_factor=grad_factor,
                )
            )
            prev_dim = out

        layers.append(GroupSum(k=self.total_out, tau=tau, device=device))

        self.backbone = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, C, H, W] float image in [0, 1].
        Returns:
            [N, S, S, K] raw output (objectness logit + class logits + coord sums).
        """
        bits = self.preprocess(x)          # [N, in_dim]
        flat = self.backbone(bits)         # [N, S*S*K]
        N = x.shape[0]
        return flat.reshape(N, self.S, self.S, self.K)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def class_offset(self) -> int:
        return self.OBJ_LEN

    def coord_offset(self, coord_idx: int) -> int:
        """coord_idx: 0=x, 1=y, 2=w, 3=h."""
        return self.OBJ_LEN + self.C + coord_idx * self.Q

    def split_output(self, out: torch.Tensor):
        """
        Split [N, S, S, K] into component tensors.

        Returns:
            obj_logit : [N, S, S, 1]
            cls_logits: [N, S, S, C]
            coord_logits: dict with keys 'x','y','w','h', each [N, S, S, Q]
        """
        obj = out[..., :self.OBJ_LEN]
        cls = out[..., self.class_offset():self.class_offset() + self.C]
        x_bins = out[..., self.coord_offset(0):self.coord_offset(0) + self.Q]
        y_bins = out[..., self.coord_offset(1):self.coord_offset(1) + self.Q]
        w_bins = out[..., self.coord_offset(2):self.coord_offset(2) + self.Q]
        h_bins = out[..., self.coord_offset(3):self.coord_offset(3) + self.Q]
        return obj, cls, {'x': x_bins, 'y': y_bins, 'w': w_bins, 'h': h_bins}
