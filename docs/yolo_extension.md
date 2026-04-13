# YOLOv1-Style Object Detection with difflogic

This document describes the extension of `difflogic` to support
**YOLOv1-style object detection** using differentiable logic gate networks.
The design keeps the entire model compatible with `CompiledLogicNet` for
fast CPU inference.

---

## Overview

`difflogic` was originally designed for classification tasks using
`LogicLayer` blocks and a `GroupSum(k=num_classes)` head.  This extension
adds:

| Module | Purpose |
|--------|---------|
| `models/logic_yolov1.py` | `PreprocessToBits` + `LogicYOLOv1Tiny` |
| `datasets/synth_shapes.py` | Synthetic shapes dataset + YOLO target builder |
| `losses/yolo_bins.py` | Binned multi-task loss |
| `postprocess/yolo_decode.py` | Decode predictions → boxes + NMS |
| `export/export_compiled_yolo.py` | Export backbone to `CompiledLogicNet` |
| `experiments/train_yolo_synth.py` | End-to-end training demo |
| `experiments/infer_yolo_synth.py` | Inference demo |
| `experiments/bench_compiled_yolo.py` | PyTorch vs compiled throughput |
| `tests/test_yolo_shapes.py` | Unit tests |
| `tests/test_export_parity.py` | Export parity tests |

---

## Key Design Decision: Regression via Binning

Logic nets output boolean or [0, 1] gate-relaxed values.  YOLOv1 needs
continuous bounding-box coordinates.  To keep everything compatible with
logic outputs (and with `CompiledLogicNet`), all four bbox coordinates
`(x, y, w, h)` are **discretized into Q bins** and treated as
multi-class classification problems.

**Encoding:**
```
bin_index = clamp(floor(value * Q), 0, Q-1)
target    = one_hot(bin_index, Q)
```

**Decoding (argmax bins → float):**
```
value = (argmax(bins) + 0.5) / Q
```

---

## Output Tensor Layout

Output shape: `[N, S, S, K]` where `K = 1 + C + 4*Q`.

```
index 0        : objectness logit (sigmoid → confidence)
index 1..C     : class logits     (softmax → class probs)
index 1+C      : x bin logits     (Q values, argmax → x_rel in cell)
index 1+C+Q    : y bin logits     (Q values, argmax → y_rel in cell)
index 1+C+2*Q  : w bin logits     (Q values, argmax → w relative to image)
index 1+C+3*Q  : h bin logits     (Q values, argmax → h relative to image)
```

---

## Model Architecture

```
Input [N,1,H,W]
    │
    ▼
PreprocessToBits                # threshold → binary bits [N, C*H*W*T]
    │  (T thresholds per pixel)
    ▼
Flatten
    │
LogicLayer(in_dim, hidden_dim)
    │
   ...  (num_layers blocks)
    │
LogicLayer(hidden_dim, last_dim)     # last_dim = expansion * S*S*K
    │
GroupSum(k=S*S*K, tau=tau)           # [N, S*S*K]
    │
Reshape (post-processing only)       # [N, S, S, K]
```

The `Flatten → LogicLayer × n → GroupSum` part is a `torch.nn.Sequential`
stored as `model.backbone`, and is directly exportable to `CompiledLogicNet`.

---

## Quick Start

### Installation

```bash
pip install torch numpy
# difflogic python implementation works on CPU without CUDA extensions
```

### Train on synthetic shapes

```bash
python experiments/train_yolo_synth.py \
    --num-samples 2000 --epochs 20 \
    --S 8 --C 3 --Q 16 --img-size 32
```

### Run inference

```bash
python experiments/infer_yolo_synth.py \
    --checkpoint checkpoints/yolo_synth_best.pt \
    --num-images 5
```

### Export + benchmark

```bash
python experiments/bench_compiled_yolo.py \
    --checkpoint checkpoints/yolo_synth_best.pt
```

### Run tests

```bash
# From the repository root
python -m pytest tests/ -v
```

---

## Loss Functions

`YOLOBinsLoss` in `losses/yolo_bins.py` implements three components:

| Component | Formula | Weight |
|-----------|---------|--------|
| Objectness | BCE on sigmoid logit | 1.0 (obj cells) / λ_noobj (no-obj cells) |
| Class | Cross-entropy | 1.0 (obj cells only) |
| Coordinates | 4× Cross-entropy over Q bins | λ_coord (obj cells only) |

Default weights: `λ_coord = 5.0`, `λ_noobj = 0.5` (following YOLOv1).

---

## Export to CompiledLogicNet

`CompiledLogicNet` compiles a `torch.nn.Sequential` of `LogicLayer` +
`GroupSum` blocks to a C shared library for fast CPU inference.

```python
from models.logic_yolov1 import LogicYOLOv1Tiny
from export.export_compiled_yolo import export_compiled, binarize_inputs

model = LogicYOLOv1Tiny(...)
model.eval()

# Export backbone
compiled = export_compiled(model, num_bits=64, cpu_compiler='gcc')

# Prepare inputs (binarize outside compiled model)
bits_np = binarize_inputs(imgs, model)   # bool ndarray [N, in_dim]

# Inference (returns raw GroupSum counts)
raw_counts = compiled(bits_np)           # [N, S*S*K] int32

# Reshape and decode
import torch
raw_tensor = torch.tensor(raw_counts, dtype=torch.float32)
raw_tensor = raw_tensor.reshape(-1, model.S, model.S, model.K)
from postprocess.yolo_decode import decode_predictions
dets = decode_predictions(raw_tensor, S=model.S, C=model.C, Q=model.Q)
```

**Limitation:** `PreprocessToBits` runs outside the compiled model (it's a
PyTorch module).  For a fully compiled pipeline one would manually threshold
the input bits in the application code before calling `compiled(bits_np)`.

---

## Hyperparameter Guide

| Parameter | Default | Notes |
|-----------|---------|-------|
| `S` | 8 | Grid cells per side. Start with S=4 for experiments. |
| `C` | 3 | Classes. Limited to 3 (rectangle/circle/triangle) for synth dataset. |
| `Q` | 16 | Bins per coordinate. Higher → better precision, larger model. |
| `num_thresholds` | 2 | Bits per pixel. 1–3 is typical. |
| `hidden_dim` | 1024 | Width of hidden LogicLayer. Keep multiple of `S*S*K`. |
| `num_layers` | 4 | Depth. For >6 layers set `grad_factor>1`. |
| `tau` | 10.0 | GroupSum temperature. Divide by to keep logits in ≈[0,1]. |

---

## File Layout

```
difflogic/
├── difflogic/          # core library (LogicLayer, GroupSum, CompiledLogicNet)
├── models/
│   └── logic_yolov1.py
├── datasets/
│   └── synth_shapes.py
├── losses/
│   └── yolo_bins.py
├── postprocess/
│   └── yolo_decode.py
├── export/
│   └── export_compiled_yolo.py
├── experiments/
│   ├── train_yolo_synth.py
│   ├── infer_yolo_synth.py
│   └── bench_compiled_yolo.py
├── tests/
│   ├── test_yolo_shapes.py
│   └── test_export_parity.py
└── docs/
    └── yolo_extension.md   ← this file
```
