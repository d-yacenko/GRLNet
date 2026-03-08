from __future__ import annotations

"""Reference evaluation entrypoint. / Точка входа для reference-оценки модели."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from grl_model.data.datasets import ImageFolderPseudoTrackDataset
from grl_model.models import grl_base, grl_tiny


def build_model(name: str, num_classes: int, track_length: int):
    if name == "grl_tiny":
        return grl_tiny(num_classes=num_classes, track_length=track_length)
    if name == "grl_base":
        return grl_base(num_classes=num_classes, track_length=track_length)
    raise ValueError(f"Unknown model: {name}")


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    eval_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolderPseudoTrackDataset(
        Path(args.data_root) / args.val_subdir,
        track_length=args.track_length,
        image_transform=eval_transform,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, num_classes=len(dataset.classes), track_length=args.track_length)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for tracks, targets in loader:
            if args.apply_gold:
                tracks = tracks.clone()
                model.prep_batch(tracks)
            tracks = tracks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(tracks)
                loss = criterion(logits, targets)
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    print(json.dumps({
        "phase": "gold" if args.apply_gold else "val",
        "loss": loss_sum / total,
        "acc": correct / total,
        "num_samples": total,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--val-subdir", default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["grl_tiny", "grl_base"], default="grl_base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--track-length", type=int, default=10)
    parser.add_argument("--apply-gold", action="store_true")
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
