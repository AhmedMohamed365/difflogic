"""
Decode raw LogicYOLOv1Tiny outputs into bounding-box predictions.

Output per cell layout (K channels):
    [obj(1) | class(C) | x_bin(Q) | y_bin(Q) | w_bin(Q) | h_bin(Q)]

Decoding steps
--------------
1. Sigmoid on objectness → confidence score.
2. Softmax on class logits → class probabilities.
3. Argmax on each coordinate bin → float coordinate via
       value = (bin_index + 0.5) / Q
   Then:
       cx_image = (cell_x + x_rel) / S
       cy_image = (cell_y + y_rel) / S
       w_image  = w_bin_value   (already relative to full image)
       h_image  = h_bin_value
4. Apply score threshold.
5. Per-class NMS.

Public API
----------
decode_predictions(raw, S, C, Q, score_thresh, iou_thresh)
    raw: [N, S, S, K] float tensor
    returns list (one per batch item) of (N_det, 6) tensors:
        [cx_norm, cy_norm, w_norm, h_norm, score, class_id]

nms(boxes_xywh_norm, scores, iou_thresh)
    boxes: [M, 4]  (cx, cy, w, h) normalised
    scores: [M]
    returns keep: LongTensor of kept indices
"""

import torch
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2] (all normalised)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def _box_iou(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes in xyxy format.
    b1: [M, 4], b2: [N, 4] → [M, N]
    """
    x1 = torch.max(b1[:, 0].unsqueeze(1), b2[:, 0].unsqueeze(0))
    y1 = torch.max(b1[:, 1].unsqueeze(1), b2[:, 1].unsqueeze(0))
    x2 = torch.min(b1[:, 2].unsqueeze(1), b2[:, 2].unsqueeze(0))
    y2 = torch.min(b1[:, 3].unsqueeze(1), b2[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = a1.unsqueeze(1) + a2.unsqueeze(0) - inter
    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(
    boxes_xywh: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float = 0.5,
) -> torch.Tensor:
    """
    Class-agnostic NMS (caller handles per-class if desired).

    Args:
        boxes_xywh: [M, 4] (cx, cy, w, h) normalised.
        scores    : [M] confidence scores.
        iou_thresh: IoU threshold for suppression.
    Returns:
        keep: [K] indices of surviving detections.
    """
    if boxes_xywh.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = _box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[rest]).squeeze(0)
        order = rest[iou <= iou_thresh]

    return torch.tensor(keep, dtype=torch.long)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode_predictions(
    raw: torch.Tensor,
    S: int,
    C: int,
    Q: int,
    score_thresh: float = 0.3,
    iou_thresh: float = 0.5,
) -> List[torch.Tensor]:
    """
    Convert raw model output to a list of detections.

    Args:
        raw        : [N, S, S, 1 + C + 4*Q] — model output (float).
        S, C, Q    : grid size, classes, quantisation bins.
        score_thresh: minimum confidence score.
        iou_thresh : IoU threshold for NMS.

    Returns:
        List of length N.  Each element is a float tensor of shape [D, 6]:
            [cx_norm, cy_norm, w_norm, h_norm, score, class_id]
        or an empty tensor of shape [0, 6] if no detections.
    """
    N = raw.shape[0]
    device = raw.device

    # Build grid cell offsets  [S, S]
    gy, gx = torch.meshgrid(
        torch.arange(S, device=device, dtype=torch.float32),
        torch.arange(S, device=device, dtype=torch.float32),
        indexing='ij',
    )  # both [S, S]

    # objectness  [N, S, S]
    obj_score = torch.sigmoid(raw[..., 0])

    # class probabilities  [N, S, S, C]
    cls_probs = F.softmax(raw[..., 1:1 + C], dim=-1)

    # per-class score = obj_score * class_prob
    # best class and score  [N, S, S]
    best_cls_score, best_cls_id = cls_probs.max(dim=-1)
    score = obj_score * best_cls_score   # [N, S, S]

    # decode coords via argmax over bins
    def _decode_coord(logits, offset_cells, scale):
        """
        logits     : [N, S, S, Q]
        offset_cells: [S, S] offset for that axis (cell_x or cell_y)
        scale      : S (for cell-relative x/y) or 1 (for w/h)
        Returns [N, S, S] normalised float.
        """
        bin_idx = logits.argmax(dim=-1).float()   # [N, S, S]
        val = (bin_idx + 0.5) / Q                 # in [0, 1]
        if scale == S:
            # cell-relative → image-relative
            return (offset_cells + val) / S
        else:
            return val

    off_x = 1 + C
    off_y = 1 + C + Q
    off_w = 1 + C + 2 * Q
    off_h = 1 + C + 3 * Q

    cx_norm = _decode_coord(raw[..., off_x:off_x + Q], gx, S)  # [N,S,S]
    cy_norm = _decode_coord(raw[..., off_y:off_y + Q], gy, S)
    w_norm  = _decode_coord(raw[..., off_w:off_w + Q], None, 1)
    h_norm  = _decode_coord(raw[..., off_h:off_h + Q], None, 1)

    results = []
    for n in range(N):
        # Flatten over grid  [S*S]
        sc  = score[n].reshape(-1)         # [S*S]
        cid = best_cls_id[n].reshape(-1).float()
        cx  = cx_norm[n].reshape(-1)
        cy  = cy_norm[n].reshape(-1)
        ww  = w_norm[n].reshape(-1)
        hh  = h_norm[n].reshape(-1)

        # Score threshold
        keep_mask = sc >= score_thresh
        if keep_mask.sum() == 0:
            results.append(torch.zeros(0, 6, device=device))
            continue

        sc  = sc[keep_mask]
        cid = cid[keep_mask]
        cx  = cx[keep_mask]
        cy  = cy[keep_mask]
        ww  = ww[keep_mask]
        hh  = hh[keep_mask]

        boxes = torch.stack([cx, cy, ww, hh], dim=1)  # [M, 4]

        # NMS
        keep_idx = nms(boxes, sc, iou_thresh)
        dets = torch.stack([
            cx[keep_idx], cy[keep_idx],
            ww[keep_idx], hh[keep_idx],
            sc[keep_idx], cid[keep_idx],
        ], dim=1)
        results.append(dets)

    return results
