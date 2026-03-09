from __future__ import annotations

"""Academic/reference evaluation entrypoint.

The script mirrors reference semantics with explicit roots:
- ``val`` is evaluated on ``eval_root`` without gold preprocessing
- ``gold`` is evaluated on ``eval_root`` with notebook-compatible ``prep_batch()``
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (repo_root, repo_root / "src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

from grl_model.data.datasets import SequenceFolderDataset
from grl_model.models import grl_base, grl_tiny
from grl_model.utils import set_reference_seed
from grl_model.utils.training import _build_default_transforms


def build_model(name: str, num_classes: int, track_length: int):
    if name == "grl_tiny":
        return grl_tiny(num_classes=num_classes, track_length=track_length)
    if name == "grl_base":
        return grl_base(num_classes=num_classes, track_length=track_length)
    raise ValueError(f"Unknown model: {name}")


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state)!r}")
    if any(key.startswith("module.") for key in state):
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint in academic/reference mode on val and gold.",
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["grl_tiny", "grl_base"], default="grl_base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--track-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def build_eval_loaders(
    *,
    train_root: Path,
    eval_root: Path,
    track_length: int,
    image_size: int,
    center_crop: bool,
    batch_size: int,
    workers: int,
    pin_memory: bool,
):
    _, eval_transform = _build_default_transforms(image_size=image_size, center_crop=center_crop)

    # The reference trainer constructs train -> val -> gold datasets in this order.
    # We mimic that order here so that val/gold grouped sampling stays aligned with
    # the reference semantics when the same seed is used.
    train_dataset_for_rng_alignment = SequenceFolderDataset(train_root, track_length)
    del train_dataset_for_rng_alignment

    val_dataset = SequenceFolderDataset(eval_root, track_length, transform=eval_transform)
    gold_dataset = SequenceFolderDataset(eval_root, track_length, transform=eval_transform)
    return {
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
        "gold": DataLoader(gold_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
    }


def evaluate_phase(
    *,
    model: nn.Module,
    loader: DataLoader,
    phase: str,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    use_amp = device.type == "cuda"
    loss_sum = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            if phase == "gold":
                inputs = inputs.clone()
                model.prep_batch(inputs)

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "num_samples": total,
    }


def main() -> None:
    args = parse_args()

    if not args.train_root.exists():
        raise FileNotFoundError(args.train_root)
    if not args.eval_root.exists():
        raise FileNotFoundError(args.eval_root)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    set_reference_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_classes = ImageFolder(args.train_root).classes
    eval_classes = ImageFolder(args.eval_root).classes
    if train_classes != eval_classes:
        raise ValueError("train_root and eval_root must expose the same class order")

    model = build_model(args.model, num_classes=len(eval_classes), track_length=args.track_length)
    model.load_state_dict(load_checkpoint_state(args.checkpoint))
    model.to(device)

    loaders = build_eval_loaders(
        train_root=args.train_root,
        eval_root=args.eval_root,
        track_length=args.track_length,
        image_size=args.image_size,
        center_crop=args.center_crop,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()

    summary = {
        "script": "scripts/eval_reference.py",
        "train_root": str(args.train_root),
        "eval_root": str(args.eval_root),
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "track_length": args.track_length,
        "image_size": args.image_size,
        "center_crop": bool(args.center_crop),
        "device": str(device),
        "seed": args.seed,
        "metrics": {
            "val": evaluate_phase(model=model, loader=loaders["val"], phase="val", device=device, criterion=criterion),
            "gold": evaluate_phase(model=model, loader=loaders["gold"], phase="gold", device=device, criterion=criterion),
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
