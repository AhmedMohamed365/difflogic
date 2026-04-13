"""
Run YOLOv1-style difflogic inference on a video and save an annotated video.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from difflogic import CompiledLogicNet
from models.logic_yolov1 import LogicYOLOv1Tiny
from postprocess.yolo_decode import decode_predictions


def parse_args():
    p = argparse.ArgumentParser(description='Run LogicYOLOv1Tiny on a video and save annotated output')
    p.add_argument('--checkpoint', required=True, help='Path to training checkpoint (.pt)')
    p.add_argument('--input-video', required=True, help='Input video path')
    p.add_argument('--output-video', required=True, help='Output annotated video path')
    p.add_argument('--device', default='cpu', help='PyTorch device (used for non-compiled path)')
    p.add_argument('--score-thresh', type=float, default=0.30, help='Detection score threshold')
    p.add_argument('--iou-thresh', type=float, default=0.50, help='NMS IoU threshold')
    p.add_argument('--max-frames', type=int, default=0, help='0=all frames, else process at most this many')
    p.add_argument('--compiled-lib', default=None, help='Optional compiled YOLO backbone .so path')
    p.add_argument('--num-bits', type=int, default=64, choices=[8, 16, 32, 64], help='Bit packing used in compiled model')
    p.add_argument('--class-names', nargs='*', default=None, help='Optional class names override')
    return p.parse_args()


def _probe_video(path: str):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json', path,
    ]
    out = subprocess.check_output(cmd).decode('utf-8')
    data = json.loads(out)
    stream = data['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    rate = stream.get('r_frame_rate', '25/1')
    if '/' in rate:
        num, den = rate.split('/')
        fps = float(num) / max(float(den), 1.0)
    else:
        fps = float(rate)
    if fps <= 0:
        fps = 25.0
    return width, height, fps


class FFMPEGReader:
    def __init__(self, input_video: str, width: int, height: int):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.proc = subprocess.Popen(
            [
                'ffmpeg', '-loglevel', 'error', '-i', input_video,
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', '0', '-',
            ],
            stdout=subprocess.PIPE,
        )

    def read(self) -> Optional[np.ndarray]:
        assert self.proc.stdout is not None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)

    def close(self):
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        self.proc.wait()


class FFMPEGWriter:
    def __init__(self, output_video: str, width: int, height: int, fps: float):
        os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
        codec = 'mpeg4'
        self.proc = subprocess.Popen(
            [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{width}x{height}', '-r', f'{fps:.6f}', '-i', '-',
                '-an', '-c:v', codec, '-pix_fmt', 'yuv420p',
                output_video,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray):
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame_rgb.tobytes())

    def close(self):
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait()


def _prepare_input(frame_rgb: np.ndarray, cfg: dict) -> torch.Tensor:
    img_h, img_w = int(cfg['img_h']), int(cfg['img_w'])
    img_channels = int(cfg['img_channels'])
    image = Image.fromarray(frame_rgb, mode='RGB').resize((img_w, img_h), resample=Image.Resampling.BILINEAR)

    if img_channels == 1:
        gray = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
        x = gray
        x = np.expand_dims(x, axis=(0, 1))  # [1,1,H,W]
    elif img_channels == 3:
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]  # [1,3,H,W]
    else:
        raise ValueError(f'Unsupported img_channels={img_channels}. Expected 1 or 3.')
    return torch.from_numpy(x)


def _draw_detections(
    frame_rgb: np.ndarray,
    detections: np.ndarray,
    class_names: Sequence[str],
) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    image = Image.fromarray(frame_rgb, mode='RGB')
    draw = ImageDraw.Draw(image)

    for det in detections:
        cx, cy, bw, bh, score, cls_id = det.tolist()
        x1 = int(max(0.0, (cx - bw / 2.0) * w))
        y1 = int(max(0.0, (cy - bh / 2.0) * h))
        x2 = int(min(w - 1.0, (cx + bw / 2.0) * w))
        y2 = int(min(h - 1.0, (cy + bh / 2.0) * h))
        cls_idx = int(cls_id)
        label_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        label = f'{label_name}:{score:.2f}'

        draw.rectangle([(x1, y1), (x2, y2)], outline=(40, 230, 40), width=2)
        draw.text((x1, max(0, y1 - 14)), label, fill=(40, 230, 40))

    return np.asarray(image, dtype=np.uint8)


def _load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt['config']
    model = LogicYOLOv1Tiny(**cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)
    return ckpt, cfg, model


def _predict_with_pytorch(model: LogicYOLOv1Tiny, inp: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(inp)


def _predict_with_compiled(
    compiled_model: CompiledLogicNet,
    model: LogicYOLOv1Tiny,
    inp: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        bits = model.preprocess(inp).numpy().astype(bool)  # [1, in_dim]
    out = compiled_model(bits).float()  # [1, S*S*K]
    return out.reshape(inp.shape[0], model.S, model.S, model.K)


def main():
    args = parse_args()

    ckpt, cfg, model = _load_model(args.checkpoint, args.device)
    class_count = int(cfg['C'])
    class_names = args.class_names if args.class_names else [f'class_{i}' for i in range(class_count)]
    if len(class_names) < class_count:
        raise ValueError(f'Need at least {class_count} class names, got {len(class_names)}')

    compiled_model: Optional[CompiledLogicNet] = None
    if args.compiled_lib:
        if not os.path.isfile(args.compiled_lib):
            raise FileNotFoundError(f'Compiled library not found: {args.compiled_lib}')
        compiled_model = CompiledLogicNet.load(args.compiled_lib, model.total_out, args.num_bits)

    out_w, out_h, fps = _probe_video(args.input_video)
    reader = FFMPEGReader(args.input_video, out_w, out_h)
    writer = FFMPEGWriter(args.output_video, out_w, out_h, fps)

    frame_idx = 0
    total_dets = 0
    while True:
        frame = reader.read()
        if frame is None:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        inp = _prepare_input(frame, cfg).to(args.device)
        if compiled_model is None:
            raw = _predict_with_pytorch(model, inp)
        else:
            raw = _predict_with_compiled(compiled_model, model, inp)

        dets = decode_predictions(
            raw,
            S=int(cfg['S']),
            C=int(cfg['C']),
            Q=int(cfg['Q']),
            score_thresh=args.score_thresh,
            iou_thresh=args.iou_thresh,
        )[0].cpu().numpy()

        total_dets += int(dets.shape[0])
        annotated = _draw_detections(frame, dets, class_names)
        writer.write(annotated)
        frame_idx += 1

    reader.close()
    writer.close()

    mode = 'compiled' if compiled_model is not None else 'pytorch'
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Model cfg: S={cfg["S"]} C={cfg["C"]} Q={cfg["Q"]} K={model.K}')
    print(f'Inference mode: {mode}')
    print(f'Frames processed: {frame_idx}')
    print(f'Total detections drawn: {total_dets}')
    print(f'Saved annotated video: {args.output_video}')


if __name__ == '__main__':
    main()
