# Sliding-Window Detector for difflogic

A detection framework that reuses an existing **difflogic classification model**
and converts it into an object detector by scanning images with many
windows/tiles, classifying each tile, then merging hits with NMS or other
post-processing.

---

## Package layout

```
detector/
  __init__.py         – public API
  backends.py         – TorchBackend / CompiledBackend
  windows.py          – WindowGenerator
  postprocess.py      – nms / soft_nms / weighted_box_fusion
  sliding_window.py   – SlidingWindowDetector
  refine.py           – inference-time box refinement

datasets/
  __init__.py
  synth_shapes_detect.py  – synthetic shapes dataset + image generator
  patch_sampler.py        – build patch dataset from detection labels

experiments/
  train_patch_classifier_synth.py  – train a simple classifier on patches
  detect_synth.py                  – run detector on synthetic images
  detect_image_folder.py           – run detector on a folder of real images
  bench_detector.py                – timing benchmarks

tests/
  test_windows.py                  – WindowGenerator unit tests
  test_nms.py                      – postprocess unit tests
  test_detector_end2end_synth.py   – end-to-end detector tests
```

---

## Quick-start

### 1. Define a `preprocess_fn`

The pre-process function must match **exactly** how the classifier was trained.
For a difflogic model trained with multi-threshold binarisation on 16×16 patches:

```python
import numpy as np

PATCH_SIZE = 16
NUM_THRESHOLDS = 3

def preprocess(patch: np.ndarray) -> np.ndarray:
    # Resize to 16×16
    h = w = PATCH_SIZE
    y_idx = np.linspace(0, patch.shape[0] - 1, h).astype(int)
    x_idx = np.linspace(0, patch.shape[1] - 1, w).astype(int)
    resized = patch[np.ix_(y_idx, x_idx)]
    f = resized.astype(np.float32) / 255.0
    if f.ndim == 3:          # (H, W, C) → greyscale
        f = f.mean(axis=2)
    # Multi-threshold binarisation
    bits = np.concatenate([
        (f > (i + 1) / (NUM_THRESHOLDS + 1)).astype(np.float32).ravel()
        for i in range(NUM_THRESHOLDS)
    ])
    return bits  # shape: (PATCH_SIZE*PATCH_SIZE*NUM_THRESHOLDS,)
```

### 2. Create a backend

```python
import torch
from detector import TorchBackend

model = torch.load("my_model.pt")   # trained difflogic classifier
model.eval()
backend = TorchBackend(model, device="cpu")
```

For a compiled model:

```python
from detector import CompiledBackend
from difflogic import CompiledLogicNet

compiled = CompiledLogicNet.load("my_model.so", num_classes=4, num_bits=64)
backend = CompiledBackend(compiled)
```

### 3. Create the detector and run detection

```python
from detector import SlidingWindowDetector

CLASS_NAMES = ["background", "circle", "rectangle", "triangle"]

detector = SlidingWindowDetector(
    backend=backend,
    input_size=(16, 16),       # (H, W) expected by classifier
    classes=CLASS_NAMES,
    preprocess_fn=preprocess,
    background_class=0,        # class index that means "no object"
    merge_method="nms",        # "nms" | "soft_nms" | "wbf"
    nms_iou_thresh=0.5,
)

detections = detector.detect(
    image=my_image_array,      # (H, W) or (H, W, C) uint8 / float32
    scales=[0.2, 0.3, 0.4],   # relative to short image side
    stride=0.25,               # fraction of window size
    score_thresh=0.6,
    batch_size=64,
)

for det in detections:
    print(det)
    # {'x1': 12.0, 'y1': 8.0, 'x2': 38.0, 'y2': 34.0, 'class_id': 1, 'score': 0.87}
```

---

## API Reference

### `SlidingWindowDetector`

**Constructor parameters**

| Parameter         | Type                 | Description |
|-------------------|----------------------|-------------|
| `backend`         | `TorchBackend` \| `CompiledBackend` | Classifier backend |
| `input_size`      | `(H, W)`             | Patch size expected by classifier |
| `classes`         | `List[str]`          | Class names (index 0 = background) |
| `preprocess_fn`   | `Callable`           | Raw patch → flat feature vector |
| `background_class`| `int`                | Class index for "no object" (default 0) |
| `merge_method`    | `str`                | `"nms"` / `"soft_nms"` / `"wbf"` |
| `nms_iou_thresh`  | `float`              | IoU threshold for NMS |

**`detect()` parameters**

| Parameter              | Default  | Description |
|------------------------|----------|-------------|
| `image`                | –        | Input image array |
| `scales`               | –        | List of scales (≤1 = relative, >1 = absolute px) |
| `stride`               | `0.25`   | Step between windows (fraction of win size) |
| `score_thresh`         | `0.5`    | Min score; can be `dict {class_id: thresh}` |
| `topk_per_class`       | `None`   | Keep at most K dets per class before NMS |
| `batch_size`           | `64`     | Patches per forward pass |
| `use_prefilter`        | `True`   | Skip low-variance patches |
| `prefilter_var_thresh` | `1e-3`   | Variance threshold for prefilter |
| `coarse_to_fine`       | `False`  | Two-pass coarse-then-fine scanning |
| `coarse_stride_factor` | `2.0`    | Stride multiplier for coarse pass |
| `refine`               | `False`  | Run multi-window snapping after NMS |
| `max_detections`       | `None`   | Cap on returned detections |

### `WindowGenerator`

```python
from detector import WindowGenerator

gen = WindowGenerator(
    input_size=(16, 16),
    scales=[0.2, 0.4, 0.6],
    stride=0.25,
    aspect_ratios=[1.0],   # optional: e.g. [0.5, 1.0, 2.0]
)
windows = gen.generate(img_h=128, img_w=128)
# list of (x1, y1, x2, y2) tuples
```

### Post-processing functions

```python
from detector.postprocess import nms, soft_nms, weighted_box_fusion

kept = nms(detections, iou_thresh=0.5)
kept = soft_nms(detections, method="linear", iou_thresh=0.5, score_thresh=0.01)
kept = weighted_box_fusion(detections, iou_thresh=0.55)
```

---

## Tuning recipe

1. **Start wide** – use many scales and small stride to find objects at all sizes:
   - `scales = [0.15, 0.25, 0.35, 0.5]`
   - `stride = 0.25`
2. **Set thresholds from validation data**
   - Run detector on a small validation set
   - Plot score distribution of true vs false positives
   - Pick threshold at the 80–95th percentile of true-positive scores
3. **NMS IoU** – use `0.5` as default; raise to `0.6` for crowded scenes
4. **Prefilter** – keep enabled (`use_prefilter=True`, `var_thresh=1e-3`) for speed

---

## Speed optimisations

The detector implements three speed optimisations:

| Optimisation           | How to enable |
|------------------------|---------------|
| **Batch inference**    | Always on; control with `batch_size` |
| **Coarse-to-fine**     | `coarse_to_fine=True` in `detect()` |
| **Variance prefilter** | `use_prefilter=True` (default) |

Typical speedup on 128×128 images (16×16 patches, random backend):
- Brute-force NMS: ~600 windows/image → ~5 ms/image
- Coarse-to-fine:  ~40% fewer windows on low-content images

---

## Known limitations

- Sliding-window struggles with **very small objects** (need tiny scales → many
  windows).
- **Crowded scenes** produce many overlapping boxes; WBF helps here.
- Objects **not centred in the patch** may score lower unless trained for
  positional invariance.
- If `CompiledLogicNet` cannot batch-process efficiently, pure brute-force
  sliding-window is still relatively slow on large images.
- Score thresholds are critical; **always calibrate on a held-out set**.

---

## Demo scripts

### Train a patch classifier on synthetic shapes

```bash
cd experiments
python train_patch_classifier_synth.py \
    --patch-size 16 \
    --num-images 500 \
    --epochs 20 \
    --output classifier_synth.pt
```

### Run detection on synthetic images

```bash
python detect_synth.py \
    --classifier classifier_synth.pt \
    --scales 0.2 0.3 0.4 \
    --stride 0.25 \
    --thresh 0.6 \
    --save-images
```

### Run detection on a folder of images

```bash
python detect_image_folder.py \
    --classifier classifier_synth.pt \
    --image-dir /path/to/images \
    --output detections.json
```

### Benchmark

```bash
python bench_detector.py \
    --num-images 50 \
    --scales 0.2 0.3 0.4 \
    --stride 0.25
```

---

## Running tests

```bash
cd tests
python -m pytest test_windows.py test_nms.py test_detector_end2end_synth.py -v
```
