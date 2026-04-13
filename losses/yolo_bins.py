"""
YOLOBinsLoss: multi-task loss for the binned YOLOv1-style difflogic detector.

Loss components
---------------
- **objectness**: binary cross-entropy on the single objectness logit per cell.
- **class**: cross-entropy on the C-way class logits (only for object cells).
- **coords**: cross-entropy on the Q-way bin logits for each of x, y, w, h
              (only for object cells).

All losses are averaged over the batch and spatial cells.

Args
----
    S          : grid size.
    C          : number of classes.
    Q          : quantisation bins per coordinate.
    lambda_coord: weight applied to the coordinate loss (default 5.0,
                  following YOLOv1).
    lambda_noobj: weight applied to the no-object BCE loss (default 0.5,
                  following YOLOv1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOBinsLoss(nn.Module):

    def __init__(
        self,
        S: int = 8,
        C: int = 3,
        Q: int = 16,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        super().__init__()
        self.S = S
        self.C = C
        self.Q = Q
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.K = 1 + C + 4 * Q

    # ------------------------------------------------------------------
    # helpers to slice the output / target tensors
    # ------------------------------------------------------------------

    @staticmethod
    def _obj(t):      return t[..., :1]
    def _cls(self, t): return t[..., 1:1 + self.C]
    def _coord(self, t, i):
        """i=0→x, 1→y, 2→w, 3→h — returns Q-dimensional slice."""
        off = 1 + self.C + i * self.Q
        return t[..., off:off + self.Q]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred  : model output, shape [N, S, S, K].
            target: target tensor from YOLOTargetBuilder, shape [N, S, S, K].
        Returns:
            scalar loss.
        """
        # obj_mask: [N, S, S] — 1 where the cell contains an object
        obj_mask = target[..., 0]               # [N, S, S]
        noobj_mask = 1.0 - obj_mask

        # ---- objectness BCE ----------------------------------------
        obj_pred = self._obj(pred).squeeze(-1)  # [N, S, S]
        obj_tgt  = obj_mask                     # [N, S, S]

        bce_obj   = F.binary_cross_entropy_with_logits(
            obj_pred, obj_tgt, reduction='none')   # [N, S, S]
        loss_obj  = (obj_mask * bce_obj).sum() / (obj_mask.sum() + 1e-6)
        loss_noobj = (noobj_mask * bce_obj).sum() / (noobj_mask.sum() + 1e-6)

        # ---- class CE (only for object cells) ----------------------
        cls_pred  = self._cls(pred)             # [N, S, S, C]
        cls_tgt   = self._cls(target)           # [N, S, S, C]  (one-hot)

        # flatten over spatial dims
        N = pred.shape[0]
        cls_pred_flat = cls_pred.reshape(-1, self.C)    # [N*S*S, C]
        cls_tgt_flat  = cls_tgt.reshape(-1, self.C)     # [N*S*S, C]
        mask_flat     = obj_mask.reshape(-1)            # [N*S*S]

        if mask_flat.sum() > 0:
            cls_ce = F.cross_entropy(
                cls_pred_flat[mask_flat.bool()],
                cls_tgt_flat[mask_flat.bool()].argmax(-1),
                reduction='mean',
            )
        else:
            cls_ce = pred.new_zeros(1).squeeze()

        # ---- coord CE (only for object cells) ----------------------
        loss_coord = pred.new_zeros(1).squeeze()
        for i in range(4):
            coord_pred = self._coord(pred, i)       # [N, S, S, Q]
            coord_tgt  = self._coord(target, i)     # [N, S, S, Q]

            cp_flat = coord_pred.reshape(-1, self.Q)
            ct_flat = coord_tgt.reshape(-1, self.Q)

            if mask_flat.sum() > 0:
                loss_coord = loss_coord + F.cross_entropy(
                    cp_flat[mask_flat.bool()],
                    ct_flat[mask_flat.bool()].argmax(-1),
                    reduction='mean',
                )

        # ---- total loss -------------------------------------------
        loss = (
            loss_obj
            + self.lambda_noobj * loss_noobj
            + cls_ce
            + self.lambda_coord * loss_coord
        )
        return loss
