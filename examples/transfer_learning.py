"""Minimal GRLNet transfer-learning example on an ImageFolder dataset.

The example loads published ImageNet-1K weights, replaces the classifier heads,
and fine-tunes the whole model by default. Use ``--freeze-backbone`` for linear
probing with only ``main_head`` and ``aux_head`` trainable.

Expected layout:

    train_root/class_name/*.jpg
    val_root/class_name/*.jpg

Example:

    python examples/transfer_learning.py \
      --train-root /path/to/files20/train \
      --val-root /path/to/files20/val \
      --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from grlnet import GRLNetWeights, grlnet_stabhrec40
from grlnet.transforms import imagenet_eval_transform, imagenet_train_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("grlnet_transfer_best.pth"))
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only classifier heads.")
    return parser.parse_args()


def set_backbone_frozen(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        is_head = name.startswith("main_head.") or name.startswith("aux_head.")
        param.requires_grad_(is_head)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_set = ImageFolder(args.train_root, transform=imagenet_train_transform())
    val_set = ImageFolder(args.val_root, transform=imagenet_eval_transform())
    if train_set.classes != val_set.classes:
        raise ValueError("Train and val class folders must match.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = grlnet_stabhrec40(weights=GRLNetWeights.DEFAULT)
    model.reset_classifier(num_classes=len(train_set.classes))
    model.to(device)
    if args.freeze_backbone:
        set_backbone_frozen(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, aux_logits = model(images, return_aux=True)
            loss = criterion(logits, labels)
            if aux_logits:
                loss = loss + 0.2 * torch.stack([criterion(aux, labels) for aux in aux_logits]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch={epoch} loss={running_loss / max(seen, 1):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_set.classes,
                    "val_acc": best_acc,
                    "source_weights": GRLNetWeights.DEFAULT.name,
                },
                args.output,
            )

    print(f"best_val_acc={best_acc:.4f}; checkpoint={args.output}")


if __name__ == "__main__":
    main()
