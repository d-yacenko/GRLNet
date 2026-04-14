from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from grlnet.inference import load_model
from grlnet.transforms import imagenet_eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GRLNet/StabHRec40 on an ImageFolder validation split.")
    parser.add_argument("--data-root", type=Path, required=True, help="ImageFolder root, e.g. ImageNet val directory.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _topk_hits(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    topk = logits.topk(min(k, logits.shape[1]), dim=1).indices
    return int(topk.eq(targets.view(-1, 1)).any(dim=1).sum().item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = ImageFolder(
        args.data_root,
        transform=imagenet_eval_transform(image_size=args.image_size, resize_size=args.resize_size),
    )
    if args.limit is not None and args.limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(int(args.limit))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    num_classes = len(dataset.dataset.classes) if isinstance(dataset, torch.utils.data.Subset) else len(dataset.classes)
    model = load_model(
        checkpoint=args.checkpoint,
        weights=args.weights,
        num_classes=num_classes,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss()
    loss_sum = 0.0
    top1_sum = 0
    top5_sum = 0
    total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            batch_size = int(targets.numel())
            loss_sum += float(loss.item()) * batch_size
            top1_sum += int((logits.argmax(dim=1) == targets).sum().item())
            top5_sum += _topk_hits(logits, targets, 5)
            total += batch_size

    print(
        json.dumps(
            {
                "loss": loss_sum / max(total, 1),
                "acc1": top1_sum / max(total, 1),
                "acc5": top5_sum / max(total, 1),
                "num_samples": total,
                "num_classes": num_classes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
