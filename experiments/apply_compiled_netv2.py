import argparse
import os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

from difflogic import CompiledLogicNet


def get_threshold_transform(dataset_name: str):
    if dataset_name == 'cifar-10-3-thresholds':
        threshold_count = 3
        transform = lambda x: torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=0)
    elif dataset_name == 'cifar-10-31-thresholds':
        threshold_count = 31
        transform = lambda x: torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=0)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(transform),
    ])
    return transforms, threshold_count


def build_thresholded_batch(images, dataset_name):
    tensors = [torchvision.transforms.ToTensor()(img) for img in images]
    x = torch.stack(tensors, dim=0)
    if dataset_name == 'cifar-10-3-thresholds':
        x = torch.cat([(x > (i + 1) / 4).float() for i in range(3)], dim=1)
    elif dataset_name == 'cifar-10-31-thresholds':
        x = torch.cat([(x > (i + 1) / 32).float() for i in range(31)], dim=1)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    return x


def save_prediction_visualization(image, gt_label, pred_label, out_path):
    base = image.copy().convert('RGB').resize((256, 256), resample=Image.Resampling.NEAREST)
    caption_h = 56
    vis = torchvision.transforms.functional.pad(base, padding=[0, 0, 0, caption_h], fill=(0, 0, 0))
    draw = ImageDraw.Draw(vis)

    gt_text = f'ground truth: {gt_label}'
    pred_text = f'predicted: {pred_label}'

    draw.text((8, 262), gt_text, fill=(255, 255, 255))
    draw.text((8, 284), pred_text, fill=(180, 220, 255))
    vis.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description='Evaluate compiled logic model on CIFAR-10 thresholds dataset.')
    parser.add_argument('--dataset', type=str, default='cifar-10-3-thresholds', choices=['cifar-10-3-thresholds', 'cifar-10-31-thresholds'])
    parser.add_argument('--data-root', type=str, default='./data-cifar')
    parser.add_argument('--batch-size', type=int, default=1_000)
    parser.add_argument('--num-bits', type=int, nargs='+', default=[64])
    parser.add_argument('--experiment-id', type=int, default=0)
    parser.add_argument('--save-count', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='predictions')
    parser.add_argument('--download', action='store_true', help='Download CIFAR-10 if missing.')
    args = parser.parse_args()

    torch.set_num_threads(1)

    _, threshold_count = get_threshold_transform(args.dataset)

    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=args.download,
        transform=None,
    )
    class_names = test_set.classes

    input_dim = 3 * 32 * 32 * threshold_count

    for num_bits in args.num_bits:
        save_lib_path = f'lib/{args.experiment_id:08d}_{num_bits}.so'
        compiled_model = CompiledLogicNet.load(save_lib_path, 10, num_bits)

        os.makedirs(args.save_dir, exist_ok=True)
        correct, total = 0, 0
        saved = 0

        for start_idx in range(0, len(test_set), args.batch_size):
            end_idx = min(start_idx + args.batch_size, len(test_set))
            batch = [test_set[i] for i in range(start_idx, end_idx)]

            images = [item[0] for item in batch]
            labels = np.array([item[1] for item in batch], dtype=np.int64)

            thresholded = build_thresholded_batch(images, args.dataset)
            data = thresholded.reshape(thresholded.shape[0], input_dim).bool().numpy()
            output = np.asarray(compiled_model.forward(data))
            if output.ndim == 0:
                preds = np.array([int(output)], dtype=np.int64)
            elif output.ndim == 1:
                if output.shape[0] == labels.shape[0]:
                    preds = output.astype(np.int64)
                else:
                    preds = np.array([int(output.argmax())], dtype=np.int64)
            else:
                preds = output.argmax(axis=-1).astype(np.int64).reshape(-1)

            if preds.shape[0] != labels.shape[0]:
                preds = np.resize(preds, labels.shape[0])

            correct += (preds == labels).sum()
            total += labels.shape[0]

            if saved < args.save_count:
                for image, gt_idx, pred_idx, sample_idx in zip(images, labels, preds, range(start_idx, end_idx)):
                    if saved >= args.save_count:
                        break
                    gt_label = class_names[int(gt_idx)]
                    pred_label = class_names[int(pred_idx)]
                    out_path = os.path.join(args.save_dir, f'sample_{sample_idx:05d}.jpg')
                    save_prediction_visualization(image, gt_label, pred_label, out_path)
                    saved += 1

        accuracy = correct / total
        print("number of samples evaluated:", total)
        print(f'saved {saved} labeled predictions in {args.save_dir}')
        print(f'COMPILED MODEL bits={num_bits} acc={accuracy:.6f}')


if __name__ == '__main__':
    main()
