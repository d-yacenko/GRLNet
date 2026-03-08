from __future__ import annotations

"""Reference ImageNet-style training entrypoint. / Точка входа для reference-обучения в стиле ImageNet."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder

from grl_model.models import grl_base, grl_tiny
from grl_model.utils import ReferenceTrainConfig, fit_reference_imagefolders, set_reference_seed


def build_model(name: str, num_classes: int, track_length: int):
    if name == "grl_tiny":
        return grl_tiny(num_classes=num_classes, track_length=track_length)
    if name == "grl_base":
        return grl_base(num_classes=num_classes, track_length=track_length)
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-subdir", default="train")
    parser.add_argument("--val-subdir", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["grl_tiny", "grl_base"], default="grl_base")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--track-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_reference_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_root = args.data_root / args.train_subdir
    eval_root = args.data_root / args.val_subdir if args.val_subdir else None
    num_classes = len(ImageFolder(train_root).classes)
    model = build_model(args.model, num_classes=num_classes, track_length=args.track_length)
    config = ReferenceTrainConfig(epochs=args.epochs)

    result = fit_reference_imagefolders(
        model,
        data_root=train_root,
        eval_root=eval_root,
        track_length=args.track_length,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        center_crop=args.center_crop,
        device=device,
        config=config,
        output_dir=args.output_dir,
    )

    print(json.dumps({
        "best_val_acc": result.best_val_acc,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "elapsed_sec": result.elapsed_sec,
    }))
